"""Differentiable radiation painting (issue #42, Phase 2: G7, G8, G11).

Pure, backend-generic (numpy/jax/torch) counterparts of the kernel-convolution
painting in :mod:`.coordinator` / :mod:`.helpers`:

- **G7** — kernels built analytically in Fourier space: the ionization bubble
  is a closed-form top-hat transform, differentiable in R_bubble; extended
  profiles use the exact radial transform K̂(k) = 4π ∫ ρ(r) sinc(kr) r² dr on
  static radial nodes, differentiable in the profile values.  No real-space
  kernel grids, no interp1d, no renormalisation step (the analytic transform
  has no sampling error to correct).
- **G8** — overlap spreading as an iterative FFT-diffusion surrogate: excess
  ionization relu(x−1) diffuses outward into neutral capacity, a fixed number
  of iterations, all smooth ops.  The exact label+EDT algorithm
  (:func:`.spread.spreading_excess_fast`) remains the default and the
  validation reference.
- **G11** — in-memory contract: :func:`paint_fields_diff` maps a halo mesh to
  (xHII, x_al kernel field, ΔT) device arrays in one graph — no HDF5, no
  ProcessPool/MPI — so gradients flow from the painted fields back through
  the halo mesh (and, via :func:`beorn.lpt.chmf.halo_field_diff`, to the
  cosmology).

The CPU production path in the coordinator is unchanged; this module is the
opt-in gpu/diff mode.

:func:`paint_fields_population_diff` (issue #59, Phase D) generalizes
:func:`paint_fields_diff` from one global bubble/profile to a real
Fourier-accumulation loop over mass bins, mirroring
:meth:`.coordinator.PaintingCoordinator.paint_single_mass_bin`/
:meth:`~.coordinator.PaintingCoordinator.paint_full`'s own algorithm: each
bin contributes ``FFT(halo_mesh_bin) × kernel_bin`` to a running Fourier-space
sum, and only the sum is inverse-transformed — one inverse FFT per field
regardless of how many mass bins there are, the same O(n_bins) saving the
production coordinator gets from returning Fourier-space contributions from
``paint_single_mass_bin``.
"""
from __future__ import annotations

import numpy as np

from ..cosmo.differentiable import get_backend, device_of, as_array, as_const

__all__ = [
    'bubble_kernel_fourier',
    'profile_kernel_fourier',
    'spreading_excess_diff',
    'paint_fields_diff',
    'paint_fields_population_diff',
]


def _k_mag_rfft(N, L, name, xp, device=None):
    """|k| on the rfftn grid (N, N, N//2+1) as a static backend constant."""
    dk = 2.0 * np.pi / L
    kx = (np.fft.fftfreq(N, d=1.0 / N) * dk)[:, None, None]
    ky = (np.fft.fftfreq(N, d=1.0 / N) * dk)[None, :, None]
    kz = (np.fft.rfftfreq(N, d=1.0 / N) * dk)[None, None, :]
    k = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    return as_const(k, name, xp, device)


def _rfftn(x, name, xp):
    return xp.fft.rfftn(x) if name != 'numpy' else np.fft.rfftn(x)


def _irfftn(xk, shape, name, xp):
    if name == 'torch':
        return xp.fft.irfftn(xk, s=shape)
    if name == 'jax':
        return xp.fft.irfftn(xk, s=shape).real
    return np.fft.irfftn(xk, s=shape)


# ──────────────────────────────────────────────────────────────────────────────
# G7 — analytic Fourier kernels
# ──────────────────────────────────────────────────────────────────────────────

def bubble_kernel_fourier(k_mag, R_bubble, backend='numpy'):
    """Fourier transform of a unit top-hat ionization bubble — closed form.

    K̂(k) = (4π/3) R³ · Ŵ(kR),   Ŵ(x) = 3 (sin x − x cos x)/x³,

    i.e. the continuous FT of x_HII(r) = 1 for r < R.  Smooth in R_bubble, so
    gradients flow from the painted xHII field back to the bubble radius (and
    through the R_bubble ODE to the ionizing-photon parameters).

    Args:
        k_mag:    |k| grid (backend array), h/Mpc — comoving.
        R_bubble: Bubble radius (comoving Mpc/h) — scalar or 0-dim tensor,
                  may carry gradients.
        backend:  'numpy' (default), 'jax' or 'torch'.

    Returns:
        K̂(k) in (Mpc/h)³, same shape as ``k_mag``.
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, k_mag, R_bubble)
    R = as_array(R_bubble, name, xp, device)
    x = k_mag * R
    small = x < 1e-4
    x_safe = xp.where(small, xp.ones_like(x), x)
    W = xp.where(small,
                 1.0 - x ** 2 / 10.0,
                 3.0 * (xp.sin(x_safe) - x_safe * xp.cos(x_safe)) / x_safe ** 3)
    return (4.0 * np.pi / 3.0) * R ** 3 * W


def profile_kernel_fourier(k_mag, r_nodes, profile_values, backend='numpy',
                           chunk=262144):
    """Radial Fourier transform of a 1-D profile — K̂(k) = 4π ∫ ρ(r) sinc(kr) r² dr.

    Trapezoid over the static radial nodes, batched over k in chunks (the
    (n_k, n_r) products are the only large intermediates).  Differentiable in
    ``profile_values`` — the gradient path from painted fields back to the
    radiation-profile solver outputs (G12).

    Args:
        k_mag:          |k| grid (backend array), h/Mpc — any shape.
        r_nodes:        Static radial nodes (numpy), comoving Mpc/h,
                        matching the coordinator's ``radial_grid * (1+z)``.
        profile_values: Profile ρ(r) on the nodes (backend array, may carry
                        gradients), shape (n_r,).
        backend:        'numpy' (default), 'jax' or 'torch'.
        chunk:          k-points per batch.

    Returns:
        K̂(k) in profile-units × (Mpc/h)³, same shape as ``k_mag``.
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, k_mag, profile_values)
    r_np = np.asarray(r_nodes, dtype=float)
    r = as_const(r_np, name, xp, device)
    dr = as_const(np.diff(r_np), name, xp, device)
    prof = as_array(profile_values, name, xp, device)

    kflat = k_mag.reshape(-1)
    n_k = kflat.shape[0]
    pieces = []
    for i0 in range(0, n_k, chunk):
        kc = kflat[i0:i0 + chunk]
        x = kc[:, None] * r[None, :]
        small = x < 1e-8
        x_safe = xp.where(small, xp.ones_like(x), x)
        sinc = xp.where(small, xp.ones_like(x), xp.sin(x_safe) / x_safe)
        integrand = prof[None, :] * sinc * r[None, :] ** 2   # (chunk, n_r)
        seg = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * dr[None, :]
        pieces.append(4.0 * np.pi * seg.sum(1))
    if name == 'torch':
        out = xp.cat(pieces)
    elif name == 'jax':
        out = xp.concatenate(pieces)
    else:
        out = np.concatenate(pieces)
    return out.reshape(k_mag.shape)


# ──────────────────────────────────────────────────────────────────────────────
# G8 — differentiable overlap-spreading surrogate
# ──────────────────────────────────────────────────────────────────────────────

def spreading_excess_diff(x, L, R_diffuse=None, n_iter=8, backend='numpy'):
    """Smooth surrogate for photon-conserving excess spreading.

    Each iteration removes the local excess relu(x−1), diffuses it with a
    Gaussian of scale ``R_diffuse`` (FFT, periodic) and adds it back — excess
    photons migrate outward into neutral regions while every operation stays
    differentiable.  Residual excess after ``n_iter`` iterations is clamped
    (documented photon loss; grows with the globally over-ionized fraction).

    The exact connected-component + EDT algorithm
    (:func:`beorn.painting.spread.spreading_excess_fast`) remains the default
    and the accuracy reference.

    Args:
        x:         Ionization field (backend array, shape (N, N, N)), any
                   values ≥ 0 (may exceed 1 where bubbles overlap).
        L:         Box size (Mpc/h).
        R_diffuse: Gaussian diffusion scale per iteration (Mpc/h).  Default:
                   2 cells.
        n_iter:    Fixed iteration count (static — jit-friendly).
        backend:   'numpy' (default), 'jax' or 'torch'.

    Returns:
        Field with values in [0, 1], same shape/device as ``x``.
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, x)
    x = as_array(x, name, xp, device)
    N = x.shape[0]
    if R_diffuse is None:
        R_diffuse = 2.0 * L / N

    k = _k_mag_rfft(N, L, name, xp, device)
    gauss_k = xp.exp(-0.5 * (k * R_diffuse) ** 2)

    def relu(v):
        return xp.where(v > 0, v, xp.zeros_like(v))

    for _ in range(n_iter):
        excess = relu(x - 1.0)
        x = x - excess
        spread = _irfftn(_rfftn(excess, name, xp) * gauss_k,
                         tuple(excess.shape), name, xp)
        x = x + spread

    # clamp the residual overshoot; keep everything ≥ 0
    x = x - relu(x - 1.0)
    return relu(x)


# ──────────────────────────────────────────────────────────────────────────────
# G11 — in-memory painting contract
# ──────────────────────────────────────────────────────────────────────────────

def paint_fields_diff(
    halo_mesh,
    z,
    L,
    R_bubble=None,
    r_alpha=None, prof_alpha=None,
    r_temp=None, prof_temp=None,
    backend='numpy',
    xHII_floor=0.0,
    spread_iter=0,
    R_diffuse=None,
):
    """Paint (xHII, x_al kernel field, ΔT) from a halo mesh — one graph, in memory.

    The differentiable, device-resident counterpart of one
    :meth:`PaintingCoordinator.paint_single_mass_bin` bin: one forward FFT of
    the halo mesh, per-field multiplication with the analytic G7 kernels,
    one inverse FFT per requested field.  Returns device arrays — no HDF5
    round-trip, no process pool (G11).  Gradients flow to R_bubble, the
    profile values, and the halo mesh itself.

    All radii comoving Mpc/h (pass the coordinator's ``radial_grid*(1+z)``);
    the x_al field is the raw kernel field — apply the 1.81e11/(1+z) scaling
    and S_alpha outside, as the coordinator does.

    Args:
        halo_mesh:  Halo-count (or λ-intensity, see ``halo_field_diff``) grid,
                    shape (N, N, N), backend array.
        z:          Redshift (static float).
        L:          Box size (Mpc/h).
        R_bubble:   Comoving bubble radius (Mpc/h, scalar; may carry grads).
                    None → skip xHII.
        r_alpha, prof_alpha: Comoving nodes + values of the Lyman-α profile.
                    None → skip.
        r_temp, prof_temp:   Comoving nodes + values of the ΔT profile.
                    None → skip.
        backend:    'numpy' (default), 'jax' or 'torch'.
        xHII_floor: Minimum ionized fraction (applied as ``maximum`` — the V9
                    floor, smooth a.e.).
        spread_iter: If > 0, apply :func:`spreading_excess_diff` with this
                    iteration count to the xHII field.
        R_diffuse:  Diffusion scale forwarded to the spreading surrogate.

    Returns:
        (Grid_xHII, Grid_xal_kernel, Grid_dT) — backend arrays or None for
        skipped fields.
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, halo_mesh, R_bubble, prof_alpha, prof_temp)
    halo_mesh = as_array(halo_mesh, name, xp, device)
    N = halo_mesh.shape[0]
    V_cell = (float(L) / N) ** 3
    shape = (N, N, N)

    fa = _rfftn(halo_mesh, name, xp)
    k = _k_mag_rfft(N, L, name, xp, device)

    grid_xHII = None
    if R_bubble is not None:
        kern = bubble_kernel_fourier(k, R_bubble, backend=backend)
        grid_xHII = _irfftn(fa * kern, shape, name, xp) / V_cell
        if spread_iter > 0:
            grid_xHII = spreading_excess_diff(grid_xHII, L,
                                              R_diffuse=R_diffuse,
                                              n_iter=spread_iter,
                                              backend=backend)
        else:
            over = xp.where(grid_xHII > 1.0, grid_xHII - 1.0,
                            xp.zeros_like(grid_xHII))
            grid_xHII = grid_xHII - over
        floor = xp.zeros_like(grid_xHII) + xHII_floor
        grid_xHII = xp.where(grid_xHII > floor, grid_xHII, floor)

    grid_xal = None
    if prof_alpha is not None:
        kern = profile_kernel_fourier(k, r_alpha, prof_alpha, backend=backend)
        grid_xal = _irfftn(fa * kern, shape, name, xp) / V_cell

    grid_temp = None
    if prof_temp is not None:
        kern = profile_kernel_fourier(k, r_temp, prof_temp, backend=backend)
        grid_temp = _irfftn(fa * kern, shape, name, xp) / V_cell

    return grid_xHII, grid_xal, grid_temp


# ──────────────────────────────────────────────────────────────────────────────
# Full per-bin Fourier-accumulation painting (issue #59, Phase D)
# ──────────────────────────────────────────────────────────────────────────────

def paint_fields_population_diff(
    halo_mesh_bins,
    z,
    L,
    R_bubble_bins=None,
    r_alpha=None, prof_alpha_bins=None,
    r_temp=None, prof_temp_bins=None,
    backend='numpy',
    xHII_floor=0.0,
    spread_iter=0,
    R_diffuse=None,
):
    """Paint (xHII, x_al kernel field, ΔT) from a *stack of per-mass-bin*
    halo meshes — the real, per-bin generalisation of :func:`paint_fields_diff`.

    Each mass bin gets its own R_bubble/profile (e.g. one row of
    :func:`beorn.precomputation.differentiable.bubble_radius_diff`'s output
    for a whole mass grid) and contributes ``FFT(mesh_bin) × kernel_bin`` to
    a running Fourier-space sum; only the accumulated sum is
    inverse-transformed, so this is **exactly 3 inverse FFTs total**,
    independent of ``n_bins`` — mirroring
    :meth:`.coordinator.PaintingCoordinator.paint_single_mass_bin`'s own
    algorithmic saving (see the module docstring). ``n_bins=1`` reduces to
    :func:`paint_fields_diff` exactly (same kernels, same accumulation, just
    one term).

    All radii comoving Mpc/h (pass ``radial_grid*(1+z)``, matching
    :func:`paint_fields_diff`); the x_al field is the raw kernel field —
    apply the ``1.81e11/(1+z)`` scaling and ``S_alpha`` outside, as the
    coordinator does.

    Args:
        halo_mesh_bins: Per-bin halo-count (or λ-intensity) grids, shape
            ``(n_bins, N, N, N)`` (backend array) — e.g.
            :func:`beorn.lpt.chmf.halo_field_diff`'s ``n_b_bins`` output
            (``return_bins=True``).
        z:          Redshift (static float).
        L:          Box size (Mpc/h).
        R_bubble_bins: Comoving bubble radius per bin (Mpc/h), shape
            ``(n_bins,)``; may carry gradients. ``None`` → skip xHII.
        r_alpha, prof_alpha_bins: Comoving nodes (shared across bins), shape
            ``(n_r,)``, and per-bin Lyman-α profile values, shape
            ``(n_bins, n_r)``. ``None`` → skip.
        r_temp, prof_temp_bins:   Same, for the ΔT profile.
        backend:    'numpy' (default), 'jax' or 'torch'.
        xHII_floor, spread_iter, R_diffuse: Forwarded to
            :func:`spreading_excess_diff`/the same post-processing
            :func:`paint_fields_diff` applies — done once, on the
            accumulated field, not per bin.

    Returns:
        (Grid_xHII, Grid_xal_kernel, Grid_dT) — backend arrays or None for
        skipped fields.
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, halo_mesh_bins, R_bubble_bins,
                       prof_alpha_bins, prof_temp_bins)
    halo_mesh_bins = as_array(halo_mesh_bins, name, xp, device)
    n_bins = halo_mesh_bins.shape[0]
    N = halo_mesh_bins.shape[1]
    V_cell = (float(L) / N) ** 3
    shape = (N, N, N)
    k = _k_mag_rfft(N, L, name, xp, device)

    fa_xHII_sum = fa_xal_sum = fa_temp_sum = None
    for b in range(n_bins):
        fa = _rfftn(halo_mesh_bins[b], name, xp)

        if R_bubble_bins is not None:
            contrib = fa * bubble_kernel_fourier(k, R_bubble_bins[b], backend=backend)
            fa_xHII_sum = contrib if fa_xHII_sum is None else fa_xHII_sum + contrib

        if prof_alpha_bins is not None:
            contrib = fa * profile_kernel_fourier(k, r_alpha, prof_alpha_bins[b],
                                                   backend=backend)
            fa_xal_sum = contrib if fa_xal_sum is None else fa_xal_sum + contrib

        if prof_temp_bins is not None:
            contrib = fa * profile_kernel_fourier(k, r_temp, prof_temp_bins[b],
                                                   backend=backend)
            fa_temp_sum = contrib if fa_temp_sum is None else fa_temp_sum + contrib

    grid_xHII = None
    if fa_xHII_sum is not None:
        grid_xHII = _irfftn(fa_xHII_sum, shape, name, xp) / V_cell
        if spread_iter > 0:
            grid_xHII = spreading_excess_diff(grid_xHII, L, R_diffuse=R_diffuse,
                                              n_iter=spread_iter, backend=backend)
        else:
            over = xp.where(grid_xHII > 1.0, grid_xHII - 1.0,
                            xp.zeros_like(grid_xHII))
            grid_xHII = grid_xHII - over
        floor = xp.zeros_like(grid_xHII) + xHII_floor
        grid_xHII = xp.where(grid_xHII > floor, grid_xHII, floor)

    grid_xal = None
    if fa_xal_sum is not None:
        grid_xal = _irfftn(fa_xal_sum, shape, name, xp) / V_cell

    grid_temp = None
    if fa_temp_sum is not None:
        grid_temp = _irfftn(fa_temp_sum, shape, name, xp) / V_cell

    return grid_xHII, grid_xal, grid_temp
