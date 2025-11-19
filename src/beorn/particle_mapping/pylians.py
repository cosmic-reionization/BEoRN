import numpy as np
import logging
from MAS_library import MASL

logger = logging.getLogger(__name__)


# MASL Options are:
# - 'NGP' (nearest grid point),
# - 'CIC' (cloud-in-cell),
# - 'TSC' (triangular-shape cloud),
# - 'PCS' (piecewise cubic spline).

def map_particles_to_mesh(mesh: np.ndarray, box_size: float, particle_positions: np.ndarray, mass_assignment: str, weights: np.ndarray = None) -> None:
    """
    Maps particle positions to an existing mesh using the MASL library.

    Parameters:
    mesh (np.ndarray): The mesh to which the particles will be mapped. Type must be float32.
    box_size (float): The size of the simulation box.
    particle_positions (np.ndarray): The positions of the particles to be mapped. Type must be float32.
    """
    assert mesh.dtype == np.float32, "Mesh must be of type float32"
    assert particle_positions.dtype == np.float32, "Particle positions must be of type float32"
    assert mesh.ndim == 3, "Mesh must be a 3D array"
    assert particle_positions.ndim == 2, "Particle positions must be a 2D array"
    assert particle_positions.shape[1] == 3, "Particle positions must have shape (N, 3)"
    assert box_size > 0, "Box size must be positive"

    MASL.MA(
        particle_positions,
        mesh,
        box_size,
        mass_assignment,
        W = weights,
        verbose = False
    )
