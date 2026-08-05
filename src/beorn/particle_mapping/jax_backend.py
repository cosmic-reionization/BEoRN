"""JAX particle-to-mesh mass assignment (differentiable).

Implements NGP, CIC, TSC, and PCS using ``jax.numpy`` scatter operations,
enabling ``jax.grad`` / ``jax.jit`` to differentiate through the mesh.
Requires ``pip install beorn[jax]``.
"""
import numpy as np

_SCHEMES = ('NGP', 'CIC', 'TSC', 'PCS')


def paint_mesh_jax(
    particle_positions,
    N: int,
    box_size: float,
    mass_assignment: str = 'CIC',
    weights=None,
):
    """Functional, differentiable particle-to-mesh painting (JAX).

    Unlike :func:`map_particles_to_mesh` this returns the painted mesh as a
    **jax array** without any numpy round-trip, so ``jax.grad``/``jax.jit``
    flow through it — use it to differentiate a measured statistic of the
    painted field with respect to the particle positions or weights (and
    through them, the cosmology).  Output dtype follows the input positions
    (pass float64 with x64 enabled for gradient checks).

    Args:
        particle_positions: (n_parts, 3) array (numpy or jax) in box units.
        N:         Mesh cells per side.
        box_size:  Box side length (same units as positions).
        mass_assignment: 'NGP', 'CIC', 'TSC', or 'PCS'.
        weights:   Optional per-particle weights, shape (n_parts,).

    Returns:
        jax array of shape (N, N, N).
    """
    try:
        import jax.numpy as jnp
    except ImportError as e:
        raise ImportError(
            "JAX is required for paint_mesh_jax. "
            "Install it with: pip install beorn[jax]"
        ) from e

    scheme = mass_assignment.upper()
    if scheme not in _SCHEMES:
        raise ValueError(
            f"JAX backend: unknown mass_assignment {mass_assignment!r}. "
            f"Choose from {_SCHEMES}."
        )

    scale = N / box_size
    pos = jnp.asarray(particle_positions) * scale   # (n, 3)
    dtype = pos.dtype
    wt = jnp.ones(pos.shape[0], dtype=dtype) if weights is None \
        else jnp.asarray(weights, dtype=dtype)
    t = jnp.zeros((N, N, N), dtype=dtype)

    def _scatter(t, ix, iy, iz, vals):
        flat_idx = ix * N * N + iy * N + iz
        return t.reshape(-1).at[flat_idx].add(vals).reshape(N, N, N)

    if scheme == 'NGP':
        ix = jnp.round(pos[:, 0]).astype(jnp.int32) % N
        iy = jnp.round(pos[:, 1]).astype(jnp.int32) % N
        iz = jnp.round(pos[:, 2]).astype(jnp.int32) % N
        t = _scatter(t, ix, iy, iz, wt)

    elif scheme == 'CIC':
        i0 = jnp.floor(pos).astype(jnp.int32)
        d1 = (pos - i0.astype(dtype))
        d0 = 1.0 - d1
        i0 = i0 % N
        i1 = (i0 + 1) % N
        for cx, wx in ((i0[:, 0], d0[:, 0]), (i1[:, 0], d1[:, 0])):
            for cy, wy in ((i0[:, 1], d0[:, 1]), (i1[:, 1], d1[:, 1])):
                for cz, wz in ((i0[:, 2], d0[:, 2]), (i1[:, 2], d1[:, 2])):
                    t = _scatter(t, cx, cy, cz, wt * wx * wy * wz)

    elif scheme == 'TSC':
        # Stencil centred on the *nearest* cell (round), not floor: with
        # floor the k=2 contribution is silently dropped for frac >= 0.5,
        # breaking mass conservation by up to ~6% (matches the numpy/numba
        # backends' TSC centring).
        i_cen = jnp.round(pos).astype(jnp.int32)
        for kx in (-1, 0, 1):
            ix = (i_cen[:, 0] + kx) % N
            wx = _w_tsc_jax(pos[:, 0] - (i_cen[:, 0] + kx).astype(dtype), jnp)
            for ky in (-1, 0, 1):
                iy = (i_cen[:, 1] + ky) % N
                wy = _w_tsc_jax(pos[:, 1] - (i_cen[:, 1] + ky).astype(dtype), jnp)
                for kz in (-1, 0, 1):
                    iz = (i_cen[:, 2] + kz) % N
                    wz = _w_tsc_jax(pos[:, 2] - (i_cen[:, 2] + kz).astype(dtype), jnp)
                    t = _scatter(t, ix, iy, iz, wt * wx * wy * wz)

    else:  # PCS
        i_cen = jnp.floor(pos).astype(jnp.int32)
        for kx in (-1, 0, 1, 2):
            ix = (i_cen[:, 0] + kx) % N
            wx = _w_pcs_jax(pos[:, 0] - (i_cen[:, 0] + kx).astype(dtype), jnp)
            for ky in (-1, 0, 1, 2):
                iy = (i_cen[:, 1] + ky) % N
                wy = _w_pcs_jax(pos[:, 1] - (i_cen[:, 1] + ky).astype(dtype), jnp)
                for kz in (-1, 0, 1, 2):
                    iz = (i_cen[:, 2] + kz) % N
                    wz = _w_pcs_jax(pos[:, 2] - (i_cen[:, 2] + kz).astype(dtype), jnp)
                    t = _scatter(t, ix, iy, iz, wt * wx * wy * wz)

    return t


def map_particles_to_mesh(
    mesh: np.ndarray,
    box_size: float,
    particle_positions: np.ndarray,
    mass_assignment: str = 'CIC',
    weights: np.ndarray = None,
) -> None:
    """Paint particles onto *mesh* in place using JAX.

    The result is written back into the NumPy *mesh* array so the interface
    matches the other backends.

    Args:
        mesh: float32 or float64 3-D NumPy array, shape ``(N, N, N)``.
            Modified in place.  Precision follows ``mesh.dtype`` (issue #52) —
            note float64 additionally requires jax's x64 mode to be enabled
            (``jax.config.update('jax_enable_x64', True)``), otherwise jax
            silently canonicalises back to float32 (see
            :class:`beorn.lpt.backends.JaxBackend`).
        box_size: Side length of the simulation box (same units as positions).
        particle_positions: Array of shape ``(n_parts, 3)``.
        mass_assignment: ``'NGP'``, ``'CIC'``, ``'TSC'``, or ``'PCS'``.
        weights: Per-particle weights, shape ``(n_parts,)``.  ``None`` → 1.
    """
    N = mesh.shape[0]
    pos_typed = np.asarray(particle_positions, dtype=mesh.dtype)
    t = paint_mesh_jax(pos_typed, N, box_size,
                       mass_assignment=mass_assignment, weights=weights)
    mesh[:] += np.asarray(t).astype(mesh.dtype, copy=False)


def _w_tsc_jax(d, jnp):
    """TSC 1-D kernel weights."""
    ad = jnp.abs(d)
    return jnp.where(ad < 0.5, 0.75 - d * d,
           jnp.where(ad < 1.5, 0.5 * (1.5 - ad) ** 2,
                     jnp.zeros_like(d)))


def _w_pcs_jax(d, jnp):
    """PCS 1-D kernel weights."""
    ad = jnp.abs(d)
    return jnp.where(ad < 1.0, (4.0 - 6.0 * d * d + 3.0 * ad ** 3) / 6.0,
           jnp.where(ad < 2.0, (2.0 - ad) ** 3 / 6.0,
                     jnp.zeros_like(d)))
