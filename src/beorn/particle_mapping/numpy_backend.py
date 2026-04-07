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
        mesh: float32 3-D array, shape ``(N, N, N)``.  Modified in place.
        box_size: Side length of the simulation box (same units as positions).
        particle_positions: float32 array, shape ``(n_parts, 3)``.
        mass_assignment: ``'NGP'``, ``'CIC'``, ``'TSC'``, or ``'PCS'``.
        weights: Per-particle weights, shape ``(n_parts,)``.  ``None`` → 1.
    """
    scheme = mass_assignment.upper()
    if scheme not in _SCHEMES:
        raise ValueError(
            f"NumPy backend: unknown mass_assignment {mass_assignment!r}. "
            f"Choose from {_SCHEMES}."
        )

    assert mesh.dtype == np.float32,               "mesh must be float32"
    assert particle_positions.dtype == np.float32, "particle_positions must be float32"
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
            pos = particle_positions[start:end] * np.float32(scale)
            wt  = (np.ones(end - start, dtype=np.float32)
                   if weights is None else weights[start:end].astype(np.float32))
            loop(mesh, N, pos, wt)
    else:
        _fn       = _NUMPY_BATCH_FN[scheme]
        batchsize = _BATCH_SIZE_PCS if scheme == 'PCS' else _BATCH_SIZE
        for start in range(0, n_part, batchsize):
            end = min(start + batchsize, n_part)
            pos = particle_positions[start:end] * scale
            w   = None if weights is None else weights[start:end]
            _fn(mesh, N, pos, w)


# ── Pure-NumPy kernel helpers ─────────────────────────────────────────────────

def _w_tsc(d: np.ndarray) -> np.ndarray:
    ad = np.abs(d)
    return np.where(ad < 0.5, 0.75 - d * d,
           np.where(ad < 1.5, 0.5 * (1.5 - ad) ** 2, 0.0)).astype(np.float32)


def _w_pcs(d: np.ndarray) -> np.ndarray:
    ad = np.abs(d)
    return np.where(ad < 1.0, (4.0 - 6.0 * d * d + 3.0 * ad ** 3) / 6.0,
           np.where(ad < 2.0, (2.0 - ad) ** 3 / 6.0, 0.0)).astype(np.float32)


# ── Pure-NumPy per-scheme batch painters ─────────────────────────────────────

def _ngp_batch(mesh, N, pos, w):
    ix = np.round(pos[:, 0]).astype(np.int32) % N
    iy = np.round(pos[:, 1]).astype(np.int32) % N
    iz = np.round(pos[:, 2]).astype(np.int32) % N
    wt = np.ones(len(pos), dtype=np.float32) if w is None else w.astype(np.float32)
    np.add.at(mesh, (ix, iy, iz), wt)


def _cic_batch(mesh, N, pos, w):
    i0 = np.floor(pos).astype(np.int32)
    d1 = (pos - i0).astype(np.float32)
    d0 = 1.0 - d1
    i0 %= N
    i1  = (i0 + 1) % N
    wt  = np.ones(len(pos), dtype=np.float32) if w is None else w.astype(np.float32)
    for cx, wx in ((i0[:, 0], d0[:, 0]), (i1[:, 0], d1[:, 0])):
        for cy, wy in ((i0[:, 1], d0[:, 1]), (i1[:, 1], d1[:, 1])):
            for cz, wz in ((i0[:, 2], d0[:, 2]), (i1[:, 2], d1[:, 2])):
                np.add.at(mesh, (cx, cy, cz), wt * wx * wy * wz)


def _tsc_batch(mesh, N, pos, w):
    i_cen = np.floor(pos).astype(np.int32)
    wt    = np.ones(len(pos), dtype=np.float32) if w is None else w.astype(np.float32)
    for kx in (-1, 0, 1):
        ix = (i_cen[:, 0] + kx) % N
        wx = _w_tsc(pos[:, 0] - (i_cen[:, 0] + kx))
        for ky in (-1, 0, 1):
            iy = (i_cen[:, 1] + ky) % N
            wy = _w_tsc(pos[:, 1] - (i_cen[:, 1] + ky))
            for kz in (-1, 0, 1):
                iz = (i_cen[:, 2] + kz) % N
                wz = _w_tsc(pos[:, 2] - (i_cen[:, 2] + kz))
                np.add.at(mesh, (ix, iy, iz), wt * wx * wy * wz)


def _pcs_batch(mesh, N, pos, w):
    i_cen = np.floor(pos).astype(np.int32)
    wt    = np.ones(len(pos), dtype=np.float32) if w is None else w.astype(np.float32)
    for kx in (-1, 0, 1, 2):
        ix = (i_cen[:, 0] + kx) % N
        wx = _w_pcs(pos[:, 0] - (i_cen[:, 0] + kx))
        for ky in (-1, 0, 1, 2):
            iy = (i_cen[:, 1] + ky) % N
            wy = _w_pcs(pos[:, 1] - (i_cen[:, 1] + ky))
            for kz in (-1, 0, 1, 2):
                iz = (i_cen[:, 2] + kz) % N
                wz = _w_pcs(pos[:, 2] - (i_cen[:, 2] + kz))
                np.add.at(mesh, (ix, iy, iz), wt * wx * wy * wz)


_NUMPY_BATCH_FN = {
    'NGP': _ngp_batch,
    'CIC': _cic_batch,
    'TSC': _tsc_batch,
    'PCS': _pcs_batch,
}
