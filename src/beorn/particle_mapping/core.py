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
    deconvolve: bool = True,
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
        deconvolve (bool): If ``True`` (default), correct the ``mass_assignment``
            window (:func:`beorn.particle_mapping.deconvolve_mas`) in place
            immediately after painting, using the same ``mass_assignment``
            scheme just painted with. This removes the ``sinc^p`` suppression
            near k_Nyquist from *any* mesh built here — the total (or
            per-particle-weight) sum is exactly preserved (the window is 1 at
            k=0), only pointwise structure near k_Nyquist changes. Pass
            ``False`` to get the raw painted mesh (e.g. to test the painting
            mechanics themselves, or when you plan to call
            ``deconvolve_mas``/``power_spectrum_1d(..., deconvolve=True)``
            yourself downstream instead).

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

    if deconvolve:
        from .window import deconvolve_mas
        mesh[...] = np.asarray(
            deconvolve_mas(mesh, box_size, mass_assignment)
        ).astype(mesh.dtype, copy=False)


def paint_displacement_field(
    mesh: np.ndarray,
    box_size: float,
    psi_x: np.ndarray,
    psi_y: np.ndarray,
    psi_z: np.ndarray,
    mass_assignment: str = 'CIC',
    backend: str = 'auto',
    weights: np.ndarray = None,
    deconvolve: bool = True,
) -> None:
    """Paint a regular grid displaced by (psi_x, psi_y, psi_z) onto *mesh*.

    Fused entry point (issue #47): skips building the ``(N^3, 3)`` flat
    position array that :func:`map_particles_to_mesh` expects, since a
    particle at flat index ``(i, j, k)`` is always at ``q_ijk + psi_ijk`` —
    not an arbitrary position. Only the ``'numpy'`` backend implements the
    fusion so far (see :func:`beorn.particle_mapping.numpy_backend.paint_displacement_field`);
    other backends fall back to building the position array the original way
    and delegating to :func:`map_particles_to_mesh` (same result, no fusion
    speedup yet — the GPU backends were flagged in #47 as a separate
    profiling judgement call).

    Args:
        mesh (np.ndarray): Target 3-D float32 array, shape ``(N, N, N)``.
        box_size (float): Side length of the simulation box (same units as
            ``psi_x``/``psi_y``/``psi_z``).
        psi_x, psi_y, psi_z (np.ndarray): Displacement field components, each
            shape ``(N, N, N)``.
        mass_assignment (str): ``'NGP'``, ``'CIC'``, ``'TSC'``, or ``'PCS'``.
        backend (str): One of ``'numpy'``, ``'numba'``, ``'pylians'``,
            ``'torch'``, ``'jax'``, or ``'auto'`` (default).
        weights (np.ndarray, optional): Per-grid-point weights, shape
            ``(N, N, N)``.  ``None`` gives uniform weight 1.
        deconvolve (bool): If ``True`` (default), correct the
            ``mass_assignment`` window in place after painting — see
            :func:`map_particles_to_mesh`.
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose from {_VALID_BACKENDS}."
        )

    if backend == 'auto':
        backend = _resolve_backend()
        logger.debug("particle_mapping: auto-selected backend=%r", backend)

    if backend == 'numpy':
        from .numpy_backend import paint_displacement_field as _fn
        _fn(mesh, box_size, psi_x, psi_y, psi_z,
            mass_assignment=mass_assignment, weights=weights)
    else:
        # Non-fused fallback: build the (N^3,3) position array the original
        # way and hand it to the existing painter.
        N = mesh.shape[0]
        q1d = (np.arange(N) + 0.5) * (box_size / N)
        x = q1d[:, None, None] + psi_x
        y = q1d[None, :, None] + psi_y
        z = q1d[None, None, :] + psi_z
        positions = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1).astype(np.float32)
        w = None if weights is None else np.asarray(weights).ravel()
        map_particles_to_mesh(mesh, box_size, positions, mass_assignment=mass_assignment,
                               backend=backend, weights=w, deconvolve=False)

    if deconvolve:
        from .window import deconvolve_mas
        mesh[...] = np.asarray(
            deconvolve_mas(mesh, box_size, mass_assignment)
        ).astype(mesh.dtype, copy=False)


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
    deconvolve: bool = True,
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
        deconvolve: If ``True`` (default), correct the ``mass_assignment``
            window (:func:`beorn.particle_mapping.deconvolve_mas`) before
            returning — see :func:`map_particles_to_mesh`. Applied via the
            same backend as the returned mesh, so differentiability/device
            residency is preserved.

    Returns:
        Mesh of shape (N, N, N): jax array / torch tensor (device-resident,
        differentiable) or numpy float32 array.
    """
    if backend == 'auto':
        backend = _infer_functional_backend(particle_positions)

    if backend == 'jax':
        from .jax_backend import paint_mesh_jax
        mesh = paint_mesh_jax(particle_positions, N, box_size,
                              mass_assignment=mass_assignment, weights=weights)

    elif backend == 'torch':
        from .torch_backend import paint_mesh_torch
        mesh = paint_mesh_torch(particle_positions, N, box_size,
                                mass_assignment=mass_assignment,
                                weights=weights)

    elif backend in ('numpy', 'numba', 'pylians'):
        mesh = np.zeros((N, N, N), dtype=np.float32)
        map_particles_to_mesh(
            mesh, box_size,
            np.asarray(particle_positions, dtype=np.float32),
            mass_assignment=mass_assignment, backend=backend,
            weights=None if weights is None else np.asarray(weights),
            deconvolve=False,
        )

    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose from {_VALID_BACKENDS}."
        )

    if deconvolve:
        from .window import deconvolve_mas
        mesh = deconvolve_mas(mesh, box_size, mass_assignment)

    return mesh
