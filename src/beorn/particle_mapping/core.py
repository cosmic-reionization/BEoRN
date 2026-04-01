"""Unified particle-to-mesh mass assignment.

Dispatches to the requested backend.  The public interface is identical
across backends so callers only need to change ``backend``.

Supported backends
------------------
``'numpy'`` (default)
    Pure-NumPy NGP/CIC/TSC/PCS implemented in
    :mod:`beorn.particle_mapping.numpy_backend`.
    No extra dependencies; works everywhere including Apple Silicon.

``'pylians'``
    Delegates to :mod:`MAS_library` from the Pylians3 package.  Faster than
    the NumPy backend (Fortran-backed, OpenMP) for very large particle counts.
    Supports NGP/CIC/TSC/PCS.  Requires ``pip install beorn[pylians]`` (may
    need manual patching on macOS — see issue #18).

``'torch'``
    Differentiable NGP/CIC via PyTorch scatter operations.
    Requires ``pip install beorn[torch]``.

``'numba'``
    Numba-JIT compiled particle loops (single-threaded, but 10-50× faster than
    the pure-NumPy ``add.at`` path).  Requires ``pip install beorn[numba]``.
    Check NumPy version compatibility before installing — Numba pins a strict
    ``numpy<X.Y`` upper bound that changes with each release.

``'jax'``
    Differentiable NGP/CIC/TSC/PCS via JAX scatter operations.
    Requires ``pip install beorn[jax]``.
"""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

_VALID_BACKENDS = ('numpy', 'numba', 'pylians', 'torch', 'jax')


def map_particles_to_mesh(
    mesh: np.ndarray,
    box_size: float,
    particle_positions: np.ndarray,
    mass_assignment: str = 'CIC',
    backend: str = 'numpy',
    weights: np.ndarray = None,
) -> None:
    """Map particle positions onto a 3-D mesh using the requested backend.

    The mesh is modified **in place**.

    Args:
        mesh (np.ndarray): Target 3-D float32 array, shape ``(N, N, N)``.
        box_size (float): Side length of the simulation box (same units as
            ``particle_positions``).
        particle_positions (np.ndarray): float32 array of shape ``(n_parts, 3)``.
        mass_assignment (str): Kernel scheme.  ``'NGP'``, ``'CIC'``,
            ``'TSC'``, ``'PCS'``.  Not all backends support all schemes — see
            notes below.
        backend (str): One of ``'numpy'``, ``'pylians'``, ``'torch'``,
            ``'jax'``.  Defaults to ``'numpy'``.
        weights (np.ndarray, optional): Per-particle weights (e.g. velocities
            for RSD).  Shape ``(n_parts,)``.  ``None`` gives uniform weight 1.

    Backend / scheme support matrix
    --------------------------------
    =========  =====  =====  =====  =====
    backend    NGP    CIC    TSC    PCS
    =========  =====  =====  =====  =====
    numpy      yes    yes    yes    yes
    numba      yes    yes    yes    yes
    pylians    yes    yes    yes    yes
    torch      yes    yes    yes    yes
    jax        yes    yes    yes    yes
    =========  =====  =====  =====  =====
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose from {_VALID_BACKENDS}."
        )

    if backend == 'numba':
        from .numba_backend import map_particles_to_mesh as _fn
        _fn(mesh, box_size, particle_positions, mass_assignment, weights=weights)

    elif backend == 'pylians':
        from .pylians_backend import map_particles_to_mesh as _fn
        _fn(mesh, box_size, particle_positions, mass_assignment, weights=weights)

    elif backend == 'numpy':
        from .numpy_backend import map_particles_to_mesh as _fn
        _fn(mesh, box_size, particle_positions, mass_assignment, weights=weights)

    elif backend == 'torch':
        from .torch_backend import map_particles_to_mesh as _fn
        _fn(mesh, box_size, particle_positions, mass_assignment, weights=weights)

    elif backend == 'jax':
        from .jax_backend import map_particles_to_mesh as _fn
        _fn(mesh, box_size, particle_positions, mass_assignment, weights=weights)
