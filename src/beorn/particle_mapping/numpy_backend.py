"""Pure-NumPy particle-to-mesh mass assignment (with optional Numba acceleration).

Implements NGP, CIC, TSC, and PCS in batched NumPy.  No compiled extensions
required — works on any platform including Apple Silicon without patching.

If Numba is installed (``pip install beorn[numba]``), the inner particle loops
are automatically replaced with JIT-compiled versions for a 10-50× speedup.
No code changes needed — the acceleration is transparent.

Kernel weights (1-D, applied separably in x, y, z)
---------------------------------------------------
Let d = fractional distance from the particle to the cell centre (in grid units).

NGP  (stencil 1):  W = 1
CIC  (stencil 2):  W(d) = 1 - |d|                              for |d| < 1
TSC  (stencil 3):  W(d) = 3/4 - d²                             for |d| < 1/2
                   W(d) = 1/2 (3/2 - |d|)²                     for 1/2 ≤ |d| < 3/2
PCS  (stencil 4):  W(d) = (4 - 6d² + 3|d|³) / 6               for |d| < 1
                   W(d) = (2 - |d|)³ / 6                        for 1 ≤ |d| < 2

The 3-D kernel is the outer product W(dx)·W(dy)·W(dz).

Cell-centered indexing (issue #55)
-----------------------------------
Mesh index ``i`` is the *center* of cell ``i`` — matching the Lagrangian
tracer grid used everywhere upstream (``(arange(N)+0.5)*cell``) and
py21cmfast's ``PerturbField.c`` convention. An undisplaced particle
(sitting exactly at its cell's center) is deposited entirely into that one
cell. All position/weight math in this module works in a *vertex*-centered
convention (``floor``/``round`` with no offset means "nearest/enclosing
mesh vertex"), so every entry point below shifts incoming positions by
``-0.5`` cell before doing anything else, once, rather than patching each
scheme's stencil separately.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Batch size chosen so that the largest intermediate arrays stay below ~1 GB.
# PCS touches 4³=64 cells per particle so gets a smaller batch.
_BATCH_SIZE     = 10_000_000
_BATCH_SIZE_PCS =  3_000_000

_SCHEMES = ('NGP', 'CIC', 'TSC', 'PCS')

# ── Optional Numba acceleration ───────────────────────────────────────────────
# Import the JIT-compiled loops at module load time.  If Numba is not installed
# this silently stays None and the pure-NumPy fallbacks are used instead.
_numba_loops = None
try:
    from .numba_backend import _LOOP_FN as _numba_loops
    logger.debug("numpy_backend: Numba available — using JIT-compiled loops.")
except ImportError:
    # issue #42, O7: the pure-NumPy np.add.at loops below are ~10-50x slower
    # than the Numba-JIT path for large particle counts — warn once so a slow
    # painting stage doesn't look like a hang.
    import warnings
    warnings.warn(
        "beorn.particle_mapping: Numba not installed — falling back to the "
        "pure-NumPy np.add.at painting loop, which is ~10-50x slower than "
        "the Numba-JIT path for large particle counts. "
        "Install with `pip install beorn[numba]` for a large speedup.",
        stacklevel=2,
    )
    logger.debug("numpy_backend: Numba not available — using pure-NumPy loops.")


def map_particles_to_mesh(
    mesh: np.ndarray,
    box_size: float,
    particle_positions: np.ndarray,
    mass_assignment: str = 'CIC',
    weights: np.ndarray = None,
) -> None:
    """Paint particles onto *mesh* in place.

    Uses Numba-JIT loops automatically if Numba is installed; otherwise falls
    back to pure NumPy.

    Args:
        mesh: float32 or float64 3-D array, shape ``(N, N, N)``.  Modified in
            place.  Precision follows ``mesh.dtype`` end to end (issue #52).
        box_size: Side length of the simulation box (same units as positions).
        particle_positions: Array of shape ``(n_parts, 3)``, same dtype as
            ``mesh``. Cell-centered: mesh index ``i`` is the *center* of
            cell ``i`` (see this module's docstring, "Cell-centered
            indexing", issue #55).
        mass_assignment: ``'NGP'``, ``'CIC'``, ``'TSC'``, or ``'PCS'``.
        weights: Per-particle weights, shape ``(n_parts,)``.  ``None`` → 1.
    """
    scheme = mass_assignment.upper()
    if scheme not in _SCHEMES:
        raise ValueError(
            f"NumPy backend: unknown mass_assignment {mass_assignment!r}. "
            f"Choose from {_SCHEMES}."
        )

    assert mesh.dtype in (np.float32, np.float64), \
        f"mesh must be float32 or float64, got {mesh.dtype}"
    assert particle_positions.dtype == mesh.dtype, \
        f"particle_positions dtype ({particle_positions.dtype}) must match mesh dtype ({mesh.dtype})"
    assert mesh.ndim == 3 and mesh.shape[0] == mesh.shape[1] == mesh.shape[2], \
        "mesh must be a cubic 3-D array"

    N      = mesh.shape[0]
    n_part = particle_positions.shape[0]
    scale  = N / box_size

    # Use Numba JIT loops if available, otherwise pure-NumPy batch functions
    if _numba_loops:
        loop      = _numba_loops[scheme]
        batchsize = _BATCH_SIZE
        for start in range(0, n_part, batchsize):
            end = min(start + batchsize, n_part)
            pos = particle_positions[start:end] * mesh.dtype.type(scale) - mesh.dtype.type(0.5)
            wt  = (np.ones(end - start, dtype=mesh.dtype)
                   if weights is None else weights[start:end].astype(mesh.dtype))
            loop(mesh, N, pos, wt)
    else:
        _fn       = _NUMPY_BATCH_FN[scheme]
        batchsize = _BATCH_SIZE_PCS if scheme == 'PCS' else _BATCH_SIZE
        for start in range(0, n_part, batchsize):
            end = min(start + batchsize, n_part)
            pos = particle_positions[start:end] * scale - 0.5
            w   = None if weights is None else weights[start:end]
            _fn(mesh, N, pos[:, 0], pos[:, 1], pos[:, 2], w)


# ── Pure-NumPy kernel helpers ─────────────────────────────────────────────────

def _w_tsc(d: np.ndarray) -> np.ndarray:
    ad = np.abs(d)
    return np.where(ad < 0.5, 0.75 - d * d,
           np.where(ad < 1.5, 0.5 * (1.5 - ad) ** 2, 0.0)).astype(d.dtype)


def _w_pcs(d: np.ndarray) -> np.ndarray:
    ad = np.abs(d)
    return np.where(ad < 1.0, (4.0 - 6.0 * d * d + 3.0 * ad ** 3) / 6.0,
           np.where(ad < 2.0, (2.0 - ad) ** 3 / 6.0, 0.0)).astype(d.dtype)


# ── Pure-NumPy per-scheme batch painters ─────────────────────────────────────
#
# Each takes cell-unit positions as three separate 1-D arrays (px, py, pz)
# rather than one combined (n, 3) array. This lets `paint_displacement_field`
# (issue #47) feed per-axis arrays straight from the grid+displacement
# broadcast, without ever materialising the (N^3, 3) stacked positions array
# that `map_particles_to_mesh`'s callers build; `map_particles_to_mesh` itself
# just slices its own combined `pos` array column-wise into the same shape.

def _ngp_batch(mesh, N, px, py, pz, w):
    ix = np.round(px).astype(np.int32) % N
    iy = np.round(py).astype(np.int32) % N
    iz = np.round(pz).astype(np.int32) % N
    wt = np.ones(len(px), dtype=px.dtype) if w is None else w.astype(px.dtype)
    np.add.at(mesh, (ix, iy, iz), wt)


def _cic_batch(mesh, N, px, py, pz, w):
    wt = np.ones(len(px), dtype=px.dtype) if w is None else w.astype(px.dtype)
    stencils = []
    for p in (px, py, pz):
        i0 = np.floor(p).astype(np.int32)
        d1 = (p - i0).astype(p.dtype)
        d0 = 1.0 - d1
        i0 %= N
        i1 = (i0 + 1) % N
        stencils.append(((i0, d0), (i1, d1)))
    for cx, wx in stencils[0]:
        for cy, wy in stencils[1]:
            for cz, wz in stencils[2]:
                np.add.at(mesh, (cx, cy, cz), wt * wx * wy * wz)


def _tsc_batch(mesh, N, px, py, pz, w):
    # TSC needs the stencil centred on the *nearest* cell (round), not floor.
    # With floor the k=2 contribution is silently dropped for frac >= 0.5,
    # breaking mass conservation by up to ~6%.
    wt = np.ones(len(px), dtype=px.dtype) if w is None else w.astype(px.dtype)
    i_cen_x = np.round(px).astype(np.int32)
    i_cen_y = np.round(py).astype(np.int32)
    i_cen_z = np.round(pz).astype(np.int32)
    for kx in (-1, 0, 1):
        ix = (i_cen_x + kx) % N
        wx = _w_tsc(px - (i_cen_x + kx))
        for ky in (-1, 0, 1):
            iy = (i_cen_y + ky) % N
            wy = _w_tsc(py - (i_cen_y + ky))
            for kz in (-1, 0, 1):
                iz = (i_cen_z + kz) % N
                wz = _w_tsc(pz - (i_cen_z + kz))
                np.add.at(mesh, (ix, iy, iz), wt * wx * wy * wz)


def _pcs_batch(mesh, N, px, py, pz, w):
    wt = np.ones(len(px), dtype=px.dtype) if w is None else w.astype(px.dtype)
    i_cen_x = np.floor(px).astype(np.int32)
    i_cen_y = np.floor(py).astype(np.int32)
    i_cen_z = np.floor(pz).astype(np.int32)
    for kx in (-1, 0, 1, 2):
        ix = (i_cen_x + kx) % N
        wx = _w_pcs(px - (i_cen_x + kx))
        for ky in (-1, 0, 1, 2):
            iy = (i_cen_y + ky) % N
            wy = _w_pcs(py - (i_cen_y + ky))
            for kz in (-1, 0, 1, 2):
                iz = (i_cen_z + kz) % N
                wz = _w_pcs(pz - (i_cen_z + kz))
                np.add.at(mesh, (ix, iy, iz), wt * wx * wy * wz)


_NUMPY_BATCH_FN = {
    'NGP': _ngp_batch,
    'CIC': _cic_batch,
    'TSC': _tsc_batch,
    'PCS': _pcs_batch,
}


# ── Fused grid+displacement painter (issue #47) ──────────────────────────────

def paint_displacement_field(
    mesh: np.ndarray,
    box_size: float,
    psi_x: np.ndarray,
    psi_y: np.ndarray,
    psi_z: np.ndarray,
    mass_assignment: str = 'CIC',
    weights: np.ndarray = None,
) -> None:
    """Paint a regular grid displaced by (psi_x, psi_y, psi_z) onto *mesh*.

    Particle p at flat index (i, j, k) sits at grid position q_ijk + psi_ijk;
    since that structure is known ahead of time, cell indices and weights are
    computed straight from the (q1d, psi) broadcast instead of routing through
    an "arbitrary particle positions" array. This skips the float64 broadcast
    temporaries and the stack/reshape copy that ``LPTBase.get_positions`` +
    :func:`map_particles_to_mesh` require (~11 GB peak at N=512 for
    ``get_positions`` alone, see issue #47 benchmark) — same arithmetic, less
    memory traffic. Scheme-agnostic: NGP/CIC/TSC/PCS all just differ in their
    weight stencil, already factored out into ``_w_tsc``/``_w_pcs``/etc.

    Args:
        mesh: float32 or float64 3-D array, shape ``(N, N, N)``.  Modified in
            place.  Precision follows ``mesh.dtype`` end to end (issue #52) —
            pass a float64 mesh to paint the (float64) displacement field
            without truncating it along the way.
        box_size: Side length of the simulation box (same units as psi_*).
        psi_x, psi_y, psi_z: Displacement field components, each shape
            ``(N, N, N)``, same units as ``box_size``.
        mass_assignment: ``'NGP'``, ``'CIC'``, ``'TSC'``, or ``'PCS'``.
        weights: Per-grid-point weights, shape ``(N, N, N)``.  ``None`` → 1.
    """
    scheme = mass_assignment.upper()
    if scheme not in _SCHEMES:
        raise ValueError(
            f"NumPy backend: unknown mass_assignment {mass_assignment!r}. "
            f"Choose from {_SCHEMES}."
        )

    assert mesh.dtype in (np.float32, np.float64), \
        f"mesh must be float32 or float64, got {mesh.dtype}"
    assert mesh.ndim == 3 and mesh.shape[0] == mesh.shape[1] == mesh.shape[2], \
        "mesh must be a cubic 3-D array"
    N = mesh.shape[0]
    assert psi_x.shape == psi_y.shape == psi_z.shape == (N, N, N), \
        "psi_x/psi_y/psi_z must have shape (N,N,N) matching mesh"

    dtype = mesh.dtype
    scale = dtype.type(N / box_size)
    # Cell-unit Lagrangian grid — (arange(N)+0.5) already *is* the cell-unit
    # position (the L/N cell size and the N/L scale cancel), so it never needs
    # to round-trip through physical units the way get_positions()'s q1d does.
    q1d = (np.arange(N, dtype=dtype) + 0.5)

    # Broadcast + cast to mesh's dtype directly, no forced-float32
    # intermediates, no np.stack/reshape into a combined (N^3,3) array — just
    # three flat views. The -0.5 matches the shift in map_particles_to_mesh
    # above: q1d is cell-centered, the stencils below are vertex-centered
    # (issue #55).
    half = dtype.type(0.5)
    px = (q1d[:, None, None] + psi_x.astype(dtype) * scale - half).ravel()
    py = (q1d[None, :, None] + psi_y.astype(dtype) * scale - half).ravel()
    pz = (q1d[None, None, :] + psi_z.astype(dtype) * scale - half).ravel()

    if weights is not None:
        assert weights.shape == (N, N, N), \
            "weights must have shape (N,N,N) matching mesh"
        weights = weights.ravel()

    n_part = px.size
    if _numba_loops:
        loop      = _numba_loops[scheme]
        batchsize = _BATCH_SIZE
        for start in range(0, n_part, batchsize):
            end = min(start + batchsize, n_part)
            pos = np.stack([px[start:end], py[start:end], pz[start:end]], axis=-1)
            wt  = (np.ones(end - start, dtype=dtype)
                   if weights is None else weights[start:end].astype(dtype))
            loop(mesh, N, pos, wt)
    else:
        _fn       = _NUMPY_BATCH_FN[scheme]
        batchsize = _BATCH_SIZE_PCS if scheme == 'PCS' else _BATCH_SIZE
        for start in range(0, n_part, batchsize):
            end = min(start + batchsize, n_part)
            w = None if weights is None else weights[start:end]
            _fn(mesh, N, px[start:end], py[start:end], pz[start:end], w)
