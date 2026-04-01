"""PyTorch particle-to-mesh mass assignment (differentiable).

Implements NGP, CIC, TSC, and PCS using ``torch.Tensor.index_put_`` with
``accumulate=True``, enabling gradients to flow through the mesh.
Requires ``pip install beorn[torch]``.
"""
import numpy as np

_SCHEMES = ('NGP', 'CIC', 'TSC', 'PCS')


def map_particles_to_mesh(
    mesh: np.ndarray,
    box_size: float,
    particle_positions: np.ndarray,
    mass_assignment: str = 'CIC',
    weights: np.ndarray = None,
) -> None:
    """Paint particles onto *mesh* in place using PyTorch.

    The result is written back into the NumPy *mesh* array so the interface
    matches the other backends.  Pass a ``torch.Tensor`` for *mesh* directly
    if you need to keep gradients attached.

    Args:
        mesh: float32 3-D NumPy array, shape ``(N, N, N)``.  Modified in place.
        box_size: Side length of the simulation box (same units as positions).
        particle_positions: float32 array, shape ``(n_parts, 3)``.
        mass_assignment: ``'NGP'``, ``'CIC'``, ``'TSC'``, or ``'PCS'``.
        weights: Per-particle weights, shape ``(n_parts,)``.  ``None`` → 1.
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

    N     = mesh.shape[0]
    scale = N / box_size

    pos = torch.from_numpy(particle_positions) * scale   # (n, 3)
    wt  = torch.ones(len(pos), dtype=torch.float32) if weights is None \
          else torch.from_numpy(weights).float()
    t   = torch.zeros(N, N, N, dtype=torch.float32)

    if scheme == 'NGP':
        ix = pos[:, 0].round().long() % N
        iy = pos[:, 1].round().long() % N
        iz = pos[:, 2].round().long() % N
        t.index_put_((ix, iy, iz), wt, accumulate=True)

    elif scheme == 'CIC':
        i0 = pos.floor().long()
        d1 = (pos - i0.float()).float()
        d0 = 1.0 - d1
        i0 = i0 % N
        i1 = (i0 + 1) % N
        for cx, wx in ((i0[:, 0], d0[:, 0]), (i1[:, 0], d1[:, 0])):
            for cy, wy in ((i0[:, 1], d0[:, 1]), (i1[:, 1], d1[:, 1])):
                for cz, wz in ((i0[:, 2], d0[:, 2]), (i1[:, 2], d1[:, 2])):
                    t.index_put_((cx, cy, cz), wt * wx * wy * wz, accumulate=True)

    elif scheme == 'TSC':
        i_cen = pos.floor().long()
        for kx in (-1, 0, 1):
            ix = (i_cen[:, 0] + kx) % N
            wx = _w_tsc_torch(pos[:, 0] - (i_cen[:, 0] + kx).float())
            for ky in (-1, 0, 1):
                iy = (i_cen[:, 1] + ky) % N
                wy = _w_tsc_torch(pos[:, 1] - (i_cen[:, 1] + ky).float())
                for kz in (-1, 0, 1):
                    iz = (i_cen[:, 2] + kz) % N
                    wz = _w_tsc_torch(pos[:, 2] - (i_cen[:, 2] + kz).float())
                    t.index_put_((ix, iy, iz), wt * wx * wy * wz, accumulate=True)

    else:  # PCS
        i_cen = pos.floor().long()
        for kx in (-1, 0, 1, 2):
            ix = (i_cen[:, 0] + kx) % N
            wx = _w_pcs_torch(pos[:, 0] - (i_cen[:, 0] + kx).float())
            for ky in (-1, 0, 1, 2):
                iy = (i_cen[:, 1] + ky) % N
                wy = _w_pcs_torch(pos[:, 1] - (i_cen[:, 1] + ky).float())
                for kz in (-1, 0, 1, 2):
                    iz = (i_cen[:, 2] + kz) % N
                    wz = _w_pcs_torch(pos[:, 2] - (i_cen[:, 2] + kz).float())
                    t.index_put_((ix, iy, iz), wt * wx * wy * wz, accumulate=True)

    mesh[:] += t.numpy()


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
