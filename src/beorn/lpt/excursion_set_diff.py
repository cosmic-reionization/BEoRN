"""Differentiable surrogate for the deterministic excursion-set halo tier.

Smooth counterpart of :class:`~beorn.lpt.excursion_set.ExcursionSetFinder`
(the same "exact algorithm vs. differentiable surrogate" split as
``spreading_method='exact'|'diffusion'`` in :mod:`beorn.painting`, see
:func:`beorn.painting.differentiable.spreading_excess_diff`): instead of
connected-component labelling of a hard barrier crossing, each fine cell gets
a continuous, per-scale soft-crossing weight, combined across scales via a
survival ("stick-breaking") product so that a cell already claimed at a
coarser (larger) scale contributes negligibly at finer scales -- a smooth
relaxation of the exact tier's "first crossing wins, then permanently
removed" rule.

Everything here is a pure, backend-generic (numpy/jax/torch) function of its
explicit arguments, following :func:`beorn.lpt.chmf.conditional_dndlnm_diff`'s
own conventions -- differentiable w.r.t. ``delta``, the cosmology and
``delta_c``.

Known, documented gap vs. the exact tier: the survival product has no analogue
of connected-component merging, so as ``T -> 0`` this converges to a
per-fine-cell first-crossing mass field, *not* the exact tier's
merged-patch masses -- by design, since the surrogate's contract is a
continuous per-cell mass field (mirroring :func:`~beorn.lpt.chmf.halo_field_diff`'s
own expectation-mode philosophy), not a discrete halo catalog.
"""
from __future__ import annotations

import numpy as np

from ..constants import rhoc0
from ..cosmo.differentiable import get_backend, device_of, as_array, as_const

__all__ = ['excursion_set_field_diff']


def _rfftn(x, name, xp):
    return xp.fft.rfftn(x) if name != 'numpy' else np.fft.rfftn(x)


def _irfftn(xk, shape, name, xp):
    if name == 'torch':
        return xp.fft.irfftn(xk, s=shape)
    if name == 'jax':
        return xp.fft.irfftn(xk, s=shape).real
    return np.fft.irfftn(xk, s=shape)


def _k_mag_rfft(N, L, name, xp, device=None):
    """|k| on the rfftn grid (N, N, N//2+1) as a static backend constant."""
    dk = 2.0 * np.pi / L
    kx = (np.fft.fftfreq(N, d=1.0 / N) * dk)[:, None, None]
    ky = (np.fft.fftfreq(N, d=1.0 / N) * dk)[None, :, None]
    kz = (np.fft.rfftfreq(N, d=1.0 / N) * dk)[None, None, :]
    k = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    return as_const(k, name, xp, device)


def _tophat_window(x, xp):
    """Top-hat Fourier window W(x) = 3(sin x - x cos x)/x^3, Taylor-expanded
    (safe, and gradient-friendly) for x -> 0 -- matches
    :func:`beorn.painting.differentiable.bubble_kernel_fourier`'s own small-x
    guard, needed here because the DC mode (k=0) always appears in a real
    FFT grid, unlike :func:`beorn.mass_function.differentiable.sigma2_M`'s
    ``k`` nodes (which start away from zero)."""
    small = x < 1e-4
    x_safe = xp.where(small, xp.ones_like(x), x)
    return xp.where(
        small,
        1.0 - x_safe ** 2 / 10.0,
        3.0 * (xp.sin(x_safe) - x_safe * xp.cos(x_safe)) / x_safe ** 3,
    )


def _barrier_diff(M, z, delta_c, Om, Ob, h0, ns, sigma_8, backend,
                  chmf_recipe, n_k, n_nodes):
    """Differentiable port of :meth:`beorn.lpt.chmf.CHMF.barrier` -- same
    two recipes, same native (z-evolved) convention, w.r.t. the cosmology
    (via ``sigma_8`` through ``sigma2_M``) and ``delta_c``."""
    from ..mass_function.differentiable import sigma2_M
    from ..cosmo.differentiable import growth_factor

    name, xp = get_backend(backend)
    device = device_of(name, xp, delta_c, Om, Ob, h0, ns, sigma_8)
    dc = as_array(delta_c, name, xp, device)

    recipe_key = chmf_recipe.replace('_', '').lower()
    if recipe_key == 'barkanaloeb2004':
        return dc
    if recipe_key != 'movingbarrier':
        raise ValueError(
            f"Unknown chmf_recipe {chmf_recipe!r}. "
            f"Choose 'BarkanaLoeb2004' or 'MovingBarrier' (case-insensitive)."
        )

    a_mb, alpha_mb, beta_mb = 0.7, 0.81, 0.34  # Jenkins et al. (2001)
    D1 = as_array(
        growth_factor(1.0 / (1.0 + z), Om, backend=backend, n_nodes=n_nodes),
        name, xp, device,
    )
    delta_crit_z = dc / D1
    x = as_array(
        sigma2_M(M, 0.0, Om, Ob, h0, ns, sigma_8, backend=backend,
                n_k=n_k, n_nodes=n_nodes),
        name, xp, device,
    )
    B_x = xp.sqrt(a_mb) * delta_crit_z * (1.0 + beta_mb * (a_mb * delta_crit_z ** 2 / x) ** (-alpha_mb))
    return B_x * D1  # native (z-evolved) convention, matching CHMF.barrier


def excursion_set_field_diff(
    delta,
    Lbox,
    M_split,
    z,
    Om, Ob, h0, ns, sigma_8,
    delta_c=1.686,
    backend='numpy',
    n_scales=32,
    T=0.1,
    M_max=None,
    chmf_recipe='BarkanaLoeb2004',
    n_k=1000,
    n_nodes=512,
):
    """Continuous, differentiable deterministic-mass field (the ``'soft'``
    counterpart of :meth:`~beorn.lpt.excursion_set.ExcursionSetFinder.find`).

    Walks the same log-spaced mass/radius hierarchy as the exact tier, large
    scale to small, top-hat-smoothing ``delta`` at each radius and combining
    a soft barrier-crossing indicator across scales via a survival
    ("stick-breaking") product to avoid double counting::

        p_i = sigmoid((delta_smoothed_i - barrier_i) / T)
        S_0 = 1;  S_i = S_{i-1} * (1 - p_{i-1})
        w_i = S_i * p_i
        M_det(x) = sum_i w_i(x) * M(R_i)

    ``1 - sum_i w_i`` is the "never (softly) crossed" residual mass fraction
    that should flow untouched into the stochastic CHMF tier (the
    differentiable analogue of :class:`~beorn.lpt.chmf.CHMFSampler`'s
    ``deterministic_mass_fraction`` kwarg: divide the returned field by
    ``rho_m * cell_volume`` and clip to ``[0, 1]``).

    Args:
        delta:   Linear overdensity field, shape (N, N, N), any backend array.
        Lbox:    Box size in Mpc/h.
        M_split: Mass threshold where the walk stops (the deterministic/
                 stochastic boundary) in M_sun.
        z:       Redshift.
        Om, Ob, h0, ns, sigma_8: Cosmological parameters (scalars or 0-dim
                 tensors carrying gradients).
        delta_c: Linear collapse threshold.
        backend: 'numpy' (default), 'jax' or 'torch'.
        n_scales: Number of log-spaced mass/radius nodes from ``M_max`` down
                 to ``M_split``.
        T:       Softness temperature of the sigmoid crossing indicator
                 (dimensionless, same units as ``delta``). Smaller -> sharper,
                 closer to the exact tier's hard threshold (per fine cell,
                 without the merge step -- see module docstring).
        M_max:   Top of the mass hierarchy in M_sun. ``None`` (default) ->
                 ``0.1 * rho_m * Lbox**3`` (matches
                 :meth:`ExcursionSetFinder.find`'s own default).
        chmf_recipe: ``'BarkanaLoeb2004'`` or ``'MovingBarrier'`` (case-
                 insensitive) -- which barrier :func:`_barrier_diff` uses,
                 mirroring :meth:`beorn.lpt.chmf.CHMF.barrier`.
        n_k, n_nodes: Forwarded to :func:`~beorn.mass_function.differentiable.sigma2_M`
                 / :func:`~beorn.cosmo.differentiable.growth_factor`.

    Returns:
        Deterministically-collapsed mass per fine cell, in M_sun -- same
        shape/backend/device as ``delta``.

    Note:
        ``M_max``/``M_split``/``n_scales`` build a **static** mass/radius
        hierarchy from the plain float values of ``Om``/``h0`` (mirroring
        :func:`~beorn.lpt.chmf.halo_field_diff`'s own static ``M_edges``/
        ``M_centers`` convention) -- gradients w.r.t. ``Om``/``h0``
        themselves are not supported through this hierarchy; ``sigma_8``
        (this module's primary differentiation target, via the barrier's
        ``sigma2_M`` dependence) is unaffected.
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, delta, delta_c, sigma_8)
    delta = as_array(delta, name, xp, device)
    N = delta.shape[0]

    rho_m = float(Om) * rhoc0 / float(h0)
    if M_max is None:
        M_max = 0.1 * rho_m * float(Lbox) ** 3
    if M_max <= M_split:
        raise ValueError(
            f"M_max ({M_max:.3e} Msun) must exceed M_split ({M_split:.3e} Msun)."
        )

    M_values = np.logspace(np.log10(M_max), np.log10(M_split), n_scales)
    R_values = (3.0 * M_values / (4.0 * np.pi * rho_m)) ** (1.0 / 3.0)

    k_mag = _k_mag_rfft(N, float(Lbox), name, xp, device)
    delta_k = _rfftn(delta, name, xp)
    shape = tuple(delta.shape)

    S = xp.ones_like(delta)
    M_det = xp.zeros_like(delta)

    for M_R, R in zip(M_values, R_values):
        barrier_val = _barrier_diff(
            float(M_R), z, delta_c, Om, Ob, h0, ns, sigma_8, backend,
            chmf_recipe, n_k, n_nodes,
        )
        W = _tophat_window(k_mag * R, xp)
        smoothed = _irfftn(delta_k * W, shape, name, xp)

        p_i = 1.0 / (1.0 + xp.exp(-(smoothed - barrier_val) / T))
        w_i = S * p_i
        M_det = M_det + w_i * float(M_R)
        S = S * (1.0 - p_i)

    return M_det
