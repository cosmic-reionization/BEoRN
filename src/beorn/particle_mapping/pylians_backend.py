"""Pylians particle-to-mesh mass assignment.

Wraps :mod:`MAS_library` from Pylians3.  Faster than the NumPy backend
(Fortran-backed, OpenMP) and supports TSC and PCS in addition to NGP/CIC.

Requires ``pip install beorn[pylians]``.  On macOS (Apple Silicon) Pylians
may need manual patching — see issue #18.  Use ``backend='numpy'`` as a
drop-in alternative that works everywhere.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def map_particles_to_mesh(
    mesh: np.ndarray,
    box_size: float,
    particle_positions: np.ndarray,
    mass_assignment: str,
    weights: np.ndarray = None,
) -> None:
    """Paint particles onto *mesh* in place using Pylians MAS_library.

    Supports NGP, CIC, TSC, and PCS mass-assignment schemes.

    Args:
        mesh: float32 3-D array, shape ``(N, N, N)``.  Modified in place.
        box_size: Side length of the simulation box (same units as positions).
        particle_positions: float32 array, shape ``(n_parts, 3)``.
        mass_assignment: ``'NGP'``, ``'CIC'``, ``'TSC'``, or ``'PCS'``.
        weights: Per-particle weights, shape ``(n_parts,)``.  ``None`` → 1.
    """
    try:
        from MAS_library import MASL
    except ImportError as e:
        raise ImportError(
            "Pylians is required for backend='pylians'. "
            "Install it with: pip install beorn[pylians]\n"
            "Note: on macOS (Apple Silicon) Pylians may need manual patching "
            "(see issue #18). Consider using backend='numpy' instead."
        ) from e

    assert mesh.dtype == np.float32,               "mesh must be float32"
    assert particle_positions.dtype == np.float32, "particle_positions must be float32"
    assert mesh.ndim == 3,                         "mesh must be a 3-D array"
    assert particle_positions.ndim == 2,           "particle_positions must have shape (N, 3)"
    assert particle_positions.shape[1] == 3,       "particle_positions must have shape (N, 3)"
    assert box_size > 0,                           "box_size must be positive"

    MASL.MA(
        particle_positions,
        mesh,
        box_size,
        mass_assignment,
        W=weights,
        verbose=False,
    )
