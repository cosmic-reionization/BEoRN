"""PyTorch particle-to-mesh mass assignment (differentiable).

Implements NGP, CIC, TSC, and PCS using ``torch.Tensor.index_put`` with
``accumulate=True``, enabling gradients to flow through the mesh.

:func:`paint_mesh_torch` is the functional contract (issue #42, G4): it
accepts torch tensors, paints on their device (CUDA / MPS / CPU) and returns
a device tensor with the autograd graph intact.  :func:`map_particles_to_mesh`
is the in-place numpy shim kept for interface parity with the other backends.
Requires ``pip install beorn[torch]``.
"""
import numpy as np

_SCHEMES = ('NGP', 'CIC', 'TSC', 'PCS')


def paint_mesh_torch(
    particle_positions,
    N: int,
    box_size: float,
    mass_assignment: str = 'CIC',
    weights=None,
    device=None,
):
    """Functional, differentiable particle-to-mesh painting (PyTorch).

    Counterpart of :func:`.jax_backend.paint_mesh_jax`: returns the painted
    mesh as a **torch tensor** on the device of the input positions (or
    *device*), with no numpy round-trip — gradients flow from the mesh back
    to the positions and weights.  Output dtype follows the input positions
    own dtype (numpy inputs are not forced to any particular precision).

    Args:
        particle_positions: (n_parts, 3) array or tensor in box units.
            Cell-centered: mesh index ``i`` is the *center* of cell ``i``
            (see :mod:`.numpy_backend`'s module docstring, "Cell-centered
            indexing", issue #55).
        N:         Mesh cells per side.
        box_size:  Box side length (same units as positions).
        mass_assignment: 'NGP', 'CIC', 'TSC', or 'PCS'.
        weights:   Optional per-particle weights, shape (n_parts,).
        device:    Optional target device (e.g. ``'cuda'``); default is the
                   device of the input tensor, or CPU for numpy input.

    Returns:
        torch tensor of shape (N, N, N).
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for backend='torch'. "
            "Install it with: pip install beorn[torch]"
        ) from e

    scheme = mass_assignment.upper()
    if scheme not in _SCHEMES:
        raise ValueError(
            f"Torch backend: unknown mass_assignment {mass_assignment!r}. "
            f"Choose from {_SCHEMES}."
        )

    if isinstance(particle_positions, torch.Tensor):
        pos = particle_positions
    else:
        pos = torch.as_tensor(np.asarray(particle_positions))
    if device is not None:
        pos = pos.to(device)
    dtype = pos.dtype
    dev = pos.device

    scale = N / box_size
    # -0.5: incoming positions are cell-centered, the stencils below index
    # the mesh in a vertex-centered convention (issue #55).
    pos = pos * scale - 0.5                              # (n, 3)
    if weights is None:
        wt = torch.ones(pos.shape[0], dtype=dtype, device=dev)
    elif isinstance(weights, torch.Tensor):
        wt = weights.to(device=dev, dtype=dtype)
    else:
        wt = torch.as_tensor(np.asarray(weights), dtype=dtype, device=dev)
    t = torch.zeros(N, N, N, dtype=dtype, device=dev)

    def _dt(x):
        return x.to(dtype)

    if scheme == 'NGP':
        ix = pos[:, 0].round().long() % N
        iy = pos[:, 1].round().long() % N
        iz = pos[:, 2].round().long() % N
        t = t.index_put((ix, iy, iz), wt, accumulate=True)

    elif scheme == 'CIC':
        i0 = pos.floor().long()
        d1 = _dt(pos - _dt(i0))
        d0 = 1.0 - d1
        i0 = i0 % N
        i1 = (i0 + 1) % N
        for cx, wx in ((i0[:, 0], d0[:, 0]), (i1[:, 0], d1[:, 0])):
            for cy, wy in ((i0[:, 1], d0[:, 1]), (i1[:, 1], d1[:, 1])):
                for cz, wz in ((i0[:, 2], d0[:, 2]), (i1[:, 2], d1[:, 2])):
                    t = t.index_put((cx, cy, cz), wt * wx * wy * wz,
                                    accumulate=True)

    elif scheme == 'TSC':
        # Stencil centred on the *nearest* cell (round), not floor: with
        # floor the k=2 contribution is silently dropped for frac >= 0.5,
        # breaking mass conservation by up to ~6% (matches the numpy/numba
        # backends' TSC centring).
        i_cen = pos.round().long()
        for kx in (-1, 0, 1):
            ix = (i_cen[:, 0] + kx) % N
            wx = _w_tsc_torch(pos[:, 0] - _dt(i_cen[:, 0] + kx))
            for ky in (-1, 0, 1):
                iy = (i_cen[:, 1] + ky) % N
                wy = _w_tsc_torch(pos[:, 1] - _dt(i_cen[:, 1] + ky))
                for kz in (-1, 0, 1):
                    iz = (i_cen[:, 2] + kz) % N
                    wz = _w_tsc_torch(pos[:, 2] - _dt(i_cen[:, 2] + kz))
                    t = t.index_put((ix, iy, iz), wt * wx * wy * wz,
                                    accumulate=True)

    else:  # PCS
        i_cen = pos.floor().long()
        for kx in (-1, 0, 1, 2):
            ix = (i_cen[:, 0] + kx) % N
            wx = _w_pcs_torch(pos[:, 0] - _dt(i_cen[:, 0] + kx))
            for ky in (-1, 0, 1, 2):
                iy = (i_cen[:, 1] + ky) % N
                wy = _w_pcs_torch(pos[:, 1] - _dt(i_cen[:, 1] + ky))
                for kz in (-1, 0, 1, 2):
                    iz = (i_cen[:, 2] + kz) % N
                    wz = _w_pcs_torch(pos[:, 2] - _dt(i_cen[:, 2] + kz))
                    t = t.index_put((ix, iy, iz), wt * wx * wy * wz,
                                    accumulate=True)

    return t


def map_particles_to_mesh(
    mesh: np.ndarray,
    box_size: float,
    particle_positions: np.ndarray,
    mass_assignment: str = 'CIC',
    weights: np.ndarray = None,
) -> None:
    """Paint particles onto *mesh* in place using PyTorch.

    The result is written back into the NumPy *mesh* array so the interface
    matches the other backends.  Use :func:`paint_mesh_torch` directly for a
    device-resident, gradient-carrying mesh.

    Args:
        mesh: float32 or float64 3-D NumPy array, shape ``(N, N, N)``.
            Modified in place.  Precision follows ``mesh.dtype`` (issue #52).
        box_size: Side length of the simulation box (same units as positions).
        particle_positions: Array of shape ``(n_parts, 3)``.
        mass_assignment: ``'NGP'``, ``'CIC'``, ``'TSC'``, or ``'PCS'``.
        weights: Per-particle weights, shape ``(n_parts,)``.  ``None`` → 1.
    """
    N = mesh.shape[0]
    pos_typed = np.asarray(particle_positions, dtype=mesh.dtype)
    t = paint_mesh_torch(pos_typed, N, box_size,
                         mass_assignment=mass_assignment, weights=weights,
                         device='cpu')
    mesh[:] += t.numpy().astype(mesh.dtype, copy=False)


def _w_tsc_torch(d):
    """TSC 1-D kernel weights."""
    import torch
    ad = d.abs()
    return torch.where(ad < 0.5, 0.75 - d * d,
           torch.where(ad < 1.5, 0.5 * (1.5 - ad) ** 2,
                       torch.zeros_like(d)))


def _w_pcs_torch(d):
    """PCS 1-D kernel weights."""
    import torch
    ad = d.abs()
    return torch.where(ad < 1.0, (4.0 - 6.0 * d * d + 3.0 * ad ** 3) / 6.0,
           torch.where(ad < 2.0, (2.0 - ad) ** 3 / 6.0,
                       torch.zeros_like(d)))
