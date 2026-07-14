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

_VALID_BACKENDS = ('numpy', 'numba', 'pylians', 'torch', 'jax', 'auto')


def _resolve_backend() -> str:
    """Return the fastest backend available on this machine.

    Priority: jax (GPU/TPU) > torch (GPU) > numba (CPU JIT) > numpy (fallback).
    GPU backends are preferred when a device is actually available.
    """
    try:
        import jax
        import jax.numpy  # noqa: F401
        devices = jax.devices()
        if any(d.platform != 'cpu' for d in devices):
            return 'jax'
    except (ImportError, Exception):
        pass

    try:
        import torch
        if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            return 'torch'
    except ImportError:
        pass

    try:
        import numba  # noqa: F401
        return 'numba'
    except ImportError:
        pass

    return 'numpy'


def map_particles_to_mesh(
    mesh: np.ndarray,
    box_size: float,
    particle_positions: np.ndarray,
    mass_assignment: str = 'CIC',
    backend: str = 'auto',
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

    if backend == 'auto':
        backend = _resolve_backend()
        logger.debug("particle_mapping: auto-selected backend=%r", backend)

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


def _infer_functional_backend(particle_positions) -> str:
    """Backend name from the array type: torch tensor → 'torch', jax array →
    'jax', anything else → 'numpy'."""
    mod = type(particle_positions).__module__
    if mod.startswith('torch'):
        return 'torch'
    if mod.startswith('jax'):
        return 'jax'
    return 'numpy'


def paint_mesh(
    particle_positions,
    weights,
    N: int,
    box_size: float,
    mass_assignment: str = 'CIC',
    backend: str = 'auto',
):
    """Functional particle-to-mesh painting: ``mesh = paint_mesh(pos, w, N, L)``.

    The functional paint contract (issue #42, G4): returns the painted mesh
    as an array of the same family as the input positions — a jax array or a
    torch tensor stays on its device with the autograd graph intact (no numpy
    round-trip); numpy input returns a float32 numpy mesh.  The in-place
    :func:`map_particles_to_mesh` API is unchanged.

    Args:
        particle_positions: (n_parts, 3) array/tensor in box units.
        weights:   Per-particle weights, shape (n_parts,), or ``None`` → 1.
        N:         Mesh cells per side.
        box_size:  Box side length (same units as positions).
        mass_assignment: ``'NGP'``, ``'CIC'``, ``'TSC'``, or ``'PCS'``.
        backend:   ``'auto'`` (default — inferred from the input type),
                   ``'numpy'``, ``'numba'``, ``'torch'``, or ``'jax'``.

    Returns:
        Mesh of shape (N, N, N): jax array / torch tensor (device-resident,
        differentiable) or numpy float32 array.
    """
    if backend == 'auto':
        backend = _infer_functional_backend(particle_positions)

    if backend == 'jax':
        from .jax_backend import paint_mesh_jax
        return paint_mesh_jax(particle_positions, N, box_size,
                              mass_assignment=mass_assignment, weights=weights)

    if backend == 'torch':
        from .torch_backend import paint_mesh_torch
        return paint_mesh_torch(particle_positions, N, box_size,
                                mass_assignment=mass_assignment,
                                weights=weights)

    if backend in ('numpy', 'numba', 'pylians'):
        mesh = np.zeros((N, N, N), dtype=np.float32)
        map_particles_to_mesh(
            mesh, box_size,
            np.asarray(particle_positions, dtype=np.float32),
            mass_assignment=mass_assignment, backend=backend,
            weights=None if weights is None else np.asarray(weights),
        )
        return mesh

    raise ValueError(
        f"Unknown backend {backend!r}. Choose from {_VALID_BACKENDS}."
    )
