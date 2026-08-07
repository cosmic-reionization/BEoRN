"""Numba-JIT particle-to-mesh mass assignment.

Implements NGP, CIC, TSC, and PCS as explicit particle loops compiled by
Numba.  Compared to the NumPy backend, the JIT removes Python overhead from
the inner loop and replaces ``np.add.at`` (which has per-element Python cost)
with direct array indexing in native code — typically 10-50× faster.

Parallelism note
----------------
The loops use ``@njit`` (single-threaded) rather than ``@njit(parallel=True)``
because multiple particles can write to the same cell, creating a race condition
with naive ``+=``.  Single-threaded JIT is already a very large win over pure
NumPy and avoids non-deterministic results.  True parallel assignment would
require either thread-local private meshes or atomic operations and is left as
a future extension.

Requires ``pip install beorn[numba]``.  If Numba is not installed, import of
this module raises ``ImportError`` with an actionable message.  The NumPy
backend (``backend='numpy'``) silently falls back to this module when Numba is
available, so users get the speedup for free without changing any code.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

_SCHEMES = ('NGP', 'CIC', 'TSC', 'PCS')

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


def _require_numba():
    if not _NUMBA_AVAILABLE:
        raise ImportError(
            "Numba is required for backend='numba'. "
            "Install it with: pip install beorn[numba]\n"
            "Note: Numba pins a strict numpy upper bound — check compatibility "
            "with your environment before installing (see project discussion on "
            "issue #18 for context)."
        )


# ── JIT-compiled particle loops ───────────────────────────────────────────────
# Defined at module level so Numba compiles them once and caches the result.
# Each function operates on a single particle batch (already scaled to grid units).

if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _ngp_loop(mesh, N, pos, wt):
        for p in range(pos.shape[0]):
            ix = int(round(pos[p, 0])) % N
            iy = int(round(pos[p, 1])) % N
            iz = int(round(pos[p, 2])) % N
            mesh[ix, iy, iz] += wt[p]

    @njit(cache=True)
    def _cic_loop(mesh, N, pos, wt):
        for p in range(pos.shape[0]):
            i0x = int(pos[p, 0]) % N
            i1x = (i0x + 1) % N
            i0y = int(pos[p, 1]) % N
            i1y = (i0y + 1) % N
            i0z = int(pos[p, 2]) % N
            i1z = (i0z + 1) % N
            d1x = pos[p, 0] - int(pos[p, 0])
            d0x = 1.0 - d1x
            d1y = pos[p, 1] - int(pos[p, 1])
            d0y = 1.0 - d1y
            d1z = pos[p, 2] - int(pos[p, 2])
            d0z = 1.0 - d1z
            w = wt[p]
            mesh[i0x, i0y, i0z] += w * d0x * d0y * d0z
            mesh[i0x, i0y, i1z] += w * d0x * d0y * d1z
            mesh[i0x, i1y, i0z] += w * d0x * d1y * d0z
            mesh[i0x, i1y, i1z] += w * d0x * d1y * d1z
            mesh[i1x, i0y, i0z] += w * d1x * d0y * d0z
            mesh[i1x, i0y, i1z] += w * d1x * d0y * d1z
            mesh[i1x, i1y, i0z] += w * d1x * d1y * d0z
            mesh[i1x, i1y, i1z] += w * d1x * d1y * d1z

    @njit(cache=True)
    def _w_tsc_scalar(d):
        ad = abs(d)
        if ad < 0.5:
            return 0.75 - d * d
        elif ad < 1.5:
            return 0.5 * (1.5 - ad) ** 2
        return 0.0

    @njit(cache=True)
    def _tsc_loop(mesh, N, pos, wt):
        for p in range(pos.shape[0]):
            icx = int(round(pos[p, 0]))
            icy = int(round(pos[p, 1]))
            icz = int(round(pos[p, 2]))
            w = wt[p]
            for kx in (-1, 0, 1):
                ix = (icx + kx) % N
                wx = _w_tsc_scalar(pos[p, 0] - (icx + kx))
                for ky in (-1, 0, 1):
                    iy = (icy + ky) % N
                    wy = _w_tsc_scalar(pos[p, 1] - (icy + ky))
                    for kz in (-1, 0, 1):
                        iz = (icz + kz) % N
                        wz = _w_tsc_scalar(pos[p, 2] - (icz + kz))
                        mesh[ix, iy, iz] += w * wx * wy * wz

    @njit(cache=True)
    def _w_pcs_scalar(d):
        ad = abs(d)
        if ad < 1.0:
            return (4.0 - 6.0 * d * d + 3.0 * ad ** 3) / 6.0
        elif ad < 2.0:
            return (2.0 - ad) ** 3 / 6.0
        return 0.0

    @njit(cache=True)
    def _pcs_loop(mesh, N, pos, wt):
        for p in range(pos.shape[0]):
            icx = int(pos[p, 0])
            icy = int(pos[p, 1])
            icz = int(pos[p, 2])
            w = wt[p]
            for kx in (-1, 0, 1, 2):
                ix = (icx + kx) % N
                wx = _w_pcs_scalar(pos[p, 0] - (icx + kx))
                for ky in (-1, 0, 1, 2):
                    iy = (icy + ky) % N
                    wy = _w_pcs_scalar(pos[p, 1] - (icy + ky))
                    for kz in (-1, 0, 1, 2):
                        iz = (icz + kz) % N
                        wz = _w_pcs_scalar(pos[p, 2] - (icz + kz))
                        mesh[ix, iy, iz] += w * wx * wy * wz

    _LOOP_FN = {
        'NGP': _ngp_loop,
        'CIC': _cic_loop,
        'TSC': _tsc_loop,
        'PCS': _pcs_loop,
    }

else:
    _LOOP_FN = {}


# ── Public entry point ────────────────────────────────────────────────────────

_BATCH_SIZE = 10_000_000


def map_particles_to_mesh(
    mesh: np.ndarray,
    box_size: float,
    particle_positions: np.ndarray,
    mass_assignment: str = 'CIC',
    weights: np.ndarray = None,
) -> None:
    """Paint particles onto *mesh* in place using Numba-JIT loops.

    Args:
        mesh: float32 or float64 3-D array, shape ``(N, N, N)``.  Modified in
            place.  Precision follows ``mesh.dtype`` (issue #52) — Numba's
            lazy JIT compiles a separate specialization per input dtype, so
            float64 works transparently.
        particle_positions: Array of shape ``(n_parts, 3)``, same dtype as
            ``mesh``. Cell-centered: mesh index ``i`` is the *center* of
            cell ``i`` (see :mod:`.numpy_backend`'s module docstring,
            "Cell-centered indexing", issue #55).
        mass_assignment: ``'NGP'``, ``'CIC'``, ``'TSC'``, or ``'PCS'``.
        weights: Per-particle weights, shape ``(n_parts,)``.  ``None`` → 1.
    """
    _require_numba()

    scheme = mass_assignment.upper()
    if scheme not in _SCHEMES:
        raise ValueError(
            f"Numba backend: unknown mass_assignment {mass_assignment!r}. "
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
    scale  = mesh.dtype.type(N / box_size)
    loop   = _LOOP_FN[scheme]

    for start in range(0, n_part, _BATCH_SIZE):
        end = min(start + _BATCH_SIZE, n_part)
        # -0.5: incoming positions are cell-centered, these loops index the
        # mesh in a vertex-centered convention (issue #55).
        pos = particle_positions[start:end] * scale - mesh.dtype.type(0.5)
        wt  = (np.ones(end - start, dtype=mesh.dtype)
               if weights is None else weights[start:end].astype(mesh.dtype))
        loop(mesh, N, pos, wt)
