"""Differentiable radiation-profile building blocks (issue #42, Phase 2: G12, G13).

Both profile ODEs in :class:`.solver.RadiationProfileSolver` are linear scalar
ODEs, so they admit closed-form integrating-factor solutions — cumulative
trapezoids over the static scale-factor nodes replace ``scipy.solve_ivp``,
making the solutions differentiable w.r.t. the tabulated source terms (and,
through them, the astro parameters f_st, Nion, cX, ...):

- bubble volume:  dV/da = A(a) − B(a)·V   →  :func:`linear_ode_solution`
- heating:        dy/da = γ(a) − 2y/a     →  :func:`heat_ode_solution`
  (exact integrating factor a²: y(a) = a⁻² ∫ γ a′² da′)

:func:`bubble_radius_diff` wraps the first with the same physics constants as
:meth:`.solver.RadiationProfileSolver.R_bubble` (Barkana & Loeb eq. 65).

G13 — stochastic stellar fractions with the noise outside the graph:
:func:`sample_fst_reparam` (mean-preserving lognormal, reparameterised) and
:func:`interp_profiles_fst` (linear interpolation between precomputed f_st
profile slices, replacing the argmin snap of the default painting path).

Ngam_dot(z) builder — real astro-parameter gradients into
:func:`bubble_radius_diff` (previously only reachable via a hand-supplied
toy array):
:func:`mass_accretion_diff`/:func:`mass_accretion_derivative_diff` (ports of
:mod:`.massaccretion`'s exponential accretion track, for one
``(Mh_center, alpha_center)`` bin) and :func:`ngam_dot_ion_diff` (port of
:func:`.helpers.Ngdot_ion`'s ``'SED'`` branch, composing those with
:func:`beorn.astro_differentiable.f_star_halo_diff`/:func:`~beorn.astro_differentiable.f_esc_diff`) —
so gradients flow from R_bubble(z) back to ``Nion``, the star-formation
efficiency shape (``f_st``, ``g1-g4``, ``Mp``, ``Mt``) and the escape
fraction shape (``f0_esc``, ``Mp_esc``, ``pl_esc``).

All functions are numpy/jax/torch backend-generic and complement (never
replace) the solve_ivp defaults. As with :mod:`.astro_differentiable`,
``backend='numpy'`` is not differentiable (plain NumPy has no autodiff) — it
exists only so the same code path can be checked against the production
functions without jax/torch installed; use ``backend='jax'``/``'torch'`` for
actual gradients.
"""
from __future__ import annotations

import numpy as np

from ..cosmo.differentiable import (
    get_backend, device_of, as_array, as_const, hubble_E, hubble_per_yr,
)
from ..constants import rhoc0, m_p_in_Msun, km_per_Mpc, cm_per_Mpc, sec_per_year, m_H, M_sun
from ..astro_differentiable import f_star_halo_diff, f_esc_diff

__all__ = [
    'cumtrapz_static',
    'linear_ode_solution',
    'heat_ode_solution',
    'bubble_radius_diff',
    'sample_fst_reparam',
    'interp_profiles_fst',
    'mass_accretion_diff',
    'mass_accretion_derivative_diff',
    'ngam_dot_ion_diff',
]


def cumtrapz_static(y, x_np, name, xp, initial=0.0):
    """Cumulative trapezoid of ``y`` over static numpy nodes ``x_np`` (last axis).

    Returns an array of the same shape as ``y`` whose first sample along the
    last axis is ``initial``.
    """
    dx = as_const(np.diff(np.asarray(x_np, dtype=float)), name, xp,
                  device=y.device if name == 'torch' else None)
    seg = 0.5 * (y[..., 1:] + y[..., :-1]) * dx
    if name == 'torch':
        csum = xp.cumsum(seg, dim=-1)
        zero = xp.zeros_like(csum[..., :1]) + initial
        return xp.cat([zero, csum], dim=-1)
    csum = xp.cumsum(seg, axis=-1)
    zero = xp.zeros_like(csum[..., :1]) + initial
    return xp.concatenate([zero, csum], axis=-1)


def linear_ode_solution(a_nodes, A, B, backend='numpy'):
    """Closed-form solution of dV/da = A(a) − B(a)·V,  V(a₀) = 0.

    Integrating factor: V(a) = e^{−Φ(a)} ∫_{a₀}^{a} A(a′) e^{Φ(a′)} da′ with
    Φ = ∫ B da — two cumulative trapezoids on the static nodes.  Differentiable
    w.r.t. the tabulated A and B (broadcasts over leading axes; the last axis
    is the a-axis).

    Args:
        a_nodes: Static, increasing scale-factor nodes (numpy), shape (n_a,).
        A, B:    Source and damping terms on the nodes (backend arrays,
                 shape (..., n_a); may carry gradients).
        backend: 'numpy' (default), 'jax' or 'torch'.

    Returns:
        V on the nodes, shape (..., n_a).
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, A, B)
    A = as_array(A, name, xp, device)
    B = as_array(B, name, xp, device)

    phi = cumtrapz_static(B, a_nodes, name, xp)
    inner = cumtrapz_static(A * xp.exp(phi), a_nodes, name, xp)
    return xp.exp(-phi) * inner


def heat_ode_solution(a_nodes, gamma, backend='numpy'):
    """Closed-form solution of dy/da = γ(a) − 2y/a,  y(a₀) = 0.

    Exact integrating factor a²:  y(a) = a⁻² ∫_{a₀}^{a} γ(a′) a′² da′.
    Replaces the ``rho_heat`` solve_ivp call on the diff path; differentiable
    w.r.t. the tabulated γ (last axis = a-axis, broadcasts over the rest).
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, gamma)
    gamma = as_array(gamma, name, xp, device)
    a = as_const(np.asarray(a_nodes, dtype=float), name, xp, device)
    inner = cumtrapz_static(gamma * a ** 2, a_nodes, name, xp)
    return inner / a ** 2


def bubble_radius_diff(z_nodes, Ngam_dot, Om, Ob, h0, clumping=1.0,
                       backend='numpy', T_recomb=1e4):
    """Comoving ionized-bubble radius R(z) — differentiable counterpart of
    :meth:`.solver.RadiationProfileSolver.R_bubble` (Barkana & Loeb eq. 65).

    Same physics: dV/da = pref·(Ṅγ/n̄_b − α_HII·C·n_b(a)·V) with
    pref = km_per_Mpc/(H(z)·a), solved by :func:`linear_ode_solution` instead
    of solve_ivp.  Differentiable w.r.t. ``Ngam_dot`` (which carries Nion/f_st
    gradients from upstream) and the cosmology scalars.

    Args:
        z_nodes:  Static, decreasing redshift nodes (numpy), shape (n_z,) —
                  the solver's ``z_bins`` convention (a increasing).
        Ngam_dot: Ionizing photon rate [s⁻¹] on the nodes, shape (..., n_z)
                  (backend array; may carry gradients).
        Om, Ob, h0: Cosmological parameters (scalars or 0-dim tensors).
        clumping: Clumping factor C.
        backend:  'numpy' (default), 'jax' or 'torch'.
        T_recomb: Temperature for the case-B recombination coefficient.

    Returns:
        R_bubble on the nodes (comoving Mpc/h), shape (..., n_z).
    """
    from ..cross_sections import alpha_HII

    name, xp = get_backend(backend)
    device = device_of(name, xp, Ngam_dot, Om, Ob, h0)
    Ngam_dot = as_array(Ngam_dot, name, xp, device)
    Ob = as_array(Ob, name, xp, device)
    h0 = as_array(h0, name, xp, device)

    z_np = np.asarray(z_nodes, dtype=float)
    a_np = 1.0 / (1.0 + z_np)
    a = as_const(a_np, name, xp, device)

    # mean comoving baryon number density [ (Mpc/h)^-3 ]
    n_bar = Ob * rhoc0 / (m_p_in_Msun * h0)
    Hz = 100.0 * h0 * hubble_E(as_const(z_np, name, xp, device), Om,
                               backend=backend)          # km/s/Mpc
    pref = km_per_Mpc / (Hz * a)                          # [s / (Mpc/h) * ...]

    alpha_rec = float(alpha_HII(T_recomb))                # cm^3/s
    A = pref * Ngam_dot / n_bar
    B = pref * alpha_rec * clumping / cm_per_Mpc ** 3 * h0 ** 3 * n_bar / a ** 3

    V = linear_ode_solution(a_np, A, B, backend=backend)
    # where-guarded cube root: d(V^{1/3})/dV diverges at V = 0 (the pre-onset
    # nodes), which would poison the whole backward pass with NaNs.
    pos = V > 0
    V_safe = xp.where(pos, V, xp.ones_like(V))
    return xp.where(pos, (V_safe * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0),
                    xp.zeros_like(V))


# ──────────────────────────────────────────────────────────────────────────────
# G13 — stochastic stellar fractions, reparameterised
# ──────────────────────────────────────────────────────────────────────────────

def sample_fst_reparam(f_st_center, sigma, eps, f_st_min=None, f_st_max=1.0,
                       backend='numpy'):
    """Mean-preserving lognormal f_st draws with the noise outside the graph.

    f = f_center · exp(σ·ε − σ²/2) — identical distribution to the coordinator's
    ``rng.lognormal(mean=ln f_c − σ²/2, sigma=σ)`` sampling, but with ε an
    explicit standard-normal input (reparameterisation), so the draws are
    differentiable w.r.t. ``f_st_center`` and ``sigma``.  Clipping to
    [f_st_min, f_st_max] uses min/max (smooth a.e., matching np.clip).

    Args:
        f_st_center: Central stellar fraction (scalar; may carry gradients).
        sigma:       Lognormal width (scalar; may carry gradients).
        eps:         Static standard-normal noise, shape (n_halos,).
        f_st_min, f_st_max: Clip bounds (None → no bound on that side).
        backend:     'numpy' (default), 'jax' or 'torch'.
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, f_st_center, sigma)
    eps = as_const(np.asarray(eps, dtype=float), name, xp, device)
    f_st_center = as_array(f_st_center, name, xp, device)
    sigma = as_array(sigma, name, xp, device)

    f = f_st_center * xp.exp(sigma * eps - sigma ** 2 / 2.0)
    if f_st_max is not None:
        hi = xp.zeros_like(f) + f_st_max
        f = xp.where(f < hi, f, hi)
    if f_st_min is not None:
        lo = xp.zeros_like(f) + f_st_min
        f = xp.where(f > lo, f, lo)
    return f


def interp_profiles_fst(f_st, f_st_grid, profile_stack, backend='numpy'):
    """Linearly interpolate precomputed profiles between f_st grid slices.

    Differentiable replacement for the argmin snap of
    ``PaintingCoordinator._nearest_f_st_indices``: each queried f_st mixes its
    two bracketing profile slices, so gradients flow through f_st into the
    painted fields (G13).  Queries are clamped to the grid range.

    Args:
        f_st:          Query values, shape (n,) (backend array; may carry
                       gradients — e.g. from :func:`sample_fst_reparam`).
        f_st_grid:     Static, increasing f_st grid (numpy), shape (n_grid,).
        profile_stack: Profiles per grid point — backend array of shape
                       (n_grid, ...) (e.g. R_bubble or ρ profiles per slice).
        backend:       'numpy' (default), 'jax' or 'torch'.

    Returns:
        Interpolated profiles, shape (n, ...).
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, f_st, profile_stack)
    f_st = as_array(f_st, name, xp, device)
    stack = as_array(profile_stack, name, xp, device)
    grid_np = np.asarray(f_st_grid, dtype=float)
    grid = as_const(grid_np, name, xp, device)

    lo, hi = float(grid_np[0]), float(grid_np[-1])
    fq = xp.where(f_st < lo, xp.zeros_like(f_st) + lo, f_st)
    fq = xp.where(fq > hi, xp.zeros_like(fq) + hi, fq)

    if name == 'torch':
        idx = xp.searchsorted(grid, fq.contiguous(), right=True)
        idx = xp.clamp(idx, 1, len(grid_np) - 1)
    elif name == 'jax':
        idx = xp.searchsorted(grid, fq, side='right')
        idx = xp.clip(idx, 1, len(grid_np) - 1)
    else:
        idx = np.clip(np.searchsorted(grid, fq, side='right'), 1,
                      len(grid_np) - 1)

    g0, g1 = grid[idx - 1], grid[idx]
    w = (fq - g0) / (g1 - g0)
    w = w.reshape(w.shape + (1,) * (stack.ndim - 1))
    return stack[idx - 1] * (1.0 - w) + stack[idx] * w


# ──────────────────────────────────────────────────────────────────────────────
# Ngam_dot(z) builder — real astro-parameter gradients (issue #42 follow-up)
# ──────────────────────────────────────────────────────────────────────────────

def mass_accretion_diff(z_bins, Mh_center, alpha_center, backend='numpy'):
    """Exponential halo-mass accretion track M(z) for one (mass, alpha) bin.

    Differentiable counterpart of :func:`.massaccretion.mass_accretion`
    (``M(z) = M0 · exp(α·(z_initial − z))``), for a single
    ``(Mh_center, alpha_center)`` pair rather than the full mass/alpha grid —
    broadcasts over leading axes like :func:`linear_ode_solution`, so batching
    multiple bins is just a matter of passing array-shaped ``Mh_center``/
    ``alpha_center``. ``z_initial = z_bins.min()``, matching production
    exactly (not a choice made here).

    Args:
        z_bins: Static redshift grid (numpy), shape (n_z,).
        Mh_center: Halo mass at ``z_initial``, Msun/h (scalar or array; may
            carry gradients).
        alpha_center: Accretion-rate exponent (scalar or array; may carry
            gradients).
        backend: 'numpy' (default, not differentiable), 'jax' or 'torch'.

    Returns:
        M(z) on ``z_bins``, shape ``(..., n_z)``.
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, Mh_center, alpha_center)
    z_np = np.asarray(z_bins, dtype=float)
    z_initial = float(z_np.min())
    z = as_const(z_np, name, xp, device)
    Mh_center = as_array(Mh_center, name, xp, device)
    alpha_center = as_array(alpha_center, name, xp, device)
    return Mh_center * xp.exp(alpha_center * (z_initial - z))


def mass_accretion_derivative_diff(Mh, alpha_center, Om, h0, z_bins,
                                   backend='numpy'):
    """dM/dt for the accretion track in :func:`mass_accretion_diff`.

    Differentiable counterpart of
    :func:`.massaccretion.mass_accretion_derivative`
    (``dMh/dt = Mh · α · H(z) · (1+z)``), using :func:`~beorn.cosmo.differentiable.hubble_per_yr`
    (flat LCDM only — the same pre-existing limitation :func:`bubble_radius_diff`
    already has via :func:`~beorn.cosmo.differentiable.hubble_E`).

    Args:
        Mh: Halo mass on ``z_bins``, Msun/h (backend array, e.g. from
            :func:`mass_accretion_diff`; may carry gradients).
        alpha_center: Accretion-rate exponent (scalar or array; may carry
            gradients).
        Om, h0: Cosmology (scalars; may carry gradients).
        z_bins: Static redshift grid (numpy), shape (n_z,), matching ``Mh``'s
            last axis.
        backend: 'numpy' (default, not differentiable), 'jax' or 'torch'.

    Returns:
        dM/dt on ``z_bins``, same shape as ``Mh``.
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, Mh, alpha_center, Om, h0)
    z_np = np.asarray(z_bins, dtype=float)
    z = as_const(z_np, name, xp, device)
    Mh = as_array(Mh, name, xp, device)
    alpha_center = as_array(alpha_center, name, xp, device)
    H = hubble_per_yr(z, Om, h0, backend=backend)
    return Mh * alpha_center * (1.0 + z) * H


def ngam_dot_ion_diff(z_bins, Mh_center, alpha_center, Om, Ob, h0, Nion,
                      f_st, Mp, g1, g2, Mt, g3, g4, halo_mass_min,
                      f0_esc, Mp_esc, pl_esc, backend='numpy'):
    """Ionizing photon rate Ngam_dot(z) for one (mass, alpha) bin.

    Differentiable counterpart of the ``'SED'`` branch of
    :func:`.helpers.Ngdot_ion` (the only production branch that's actually
    tested — the others are marked TODO/untested there), composing
    :func:`mass_accretion_diff`/:func:`mass_accretion_derivative_diff` with
    :func:`beorn.astro_differentiable.f_star_halo_diff`/
    :func:`beorn.astro_differentiable.f_esc_diff`. A drop-in, differentiable
    replacement for a hand-supplied ``Ngam_dot`` array — same shape contract
    as :func:`bubble_radius_diff`'s ``Ngam_dot`` argument (broadcastable to
    ``(..., n_z)`` matching ``z_bins``).

    Gradients flow to ``Nion`` and every star-formation-efficiency/escape-
    fraction shape parameter (``f_st``, ``Mp``, ``g1``, ``g2``, ``Mt``,
    ``g3``, ``g4``, ``f0_esc``, ``Mp_esc``, ``pl_esc``) when
    ``backend='jax'``/``'torch'``. ``halo_mass_min`` is a static cutoff, not
    differentiated (see :func:`beorn.astro_differentiable.f_star_halo_diff`).

    Args:
        z_bins: Static redshift grid (numpy), shape (n_z,).
        Mh_center, alpha_center: Mass-accretion-track bin center (scalars;
            may carry gradients) — see :func:`mass_accretion_diff`.
        Om, Ob, h0: Cosmology (scalars; may carry gradients).
        Nion: Ionizing photons per baryon in stars (scalar; may carry
            gradients).
        f_st, Mp, g1, g2, Mt, g3, g4: Star-formation-efficiency shape
            parameters — see :func:`beorn.astro_differentiable.f_star_halo_diff`.
        halo_mass_min: Static mass floor, Msun/h (not differentiated).
        f0_esc, Mp_esc, pl_esc: Escape-fraction shape parameters — see
            :func:`beorn.astro_differentiable.f_esc_diff`.
        backend: 'numpy' (default, not differentiable), 'jax' or 'torch'.

    Returns:
        Ngam_dot on ``z_bins`` [s⁻¹], shape (n_z,).
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, Mh_center, alpha_center, Om, Ob, h0, Nion)

    Mh = mass_accretion_diff(z_bins, Mh_center, alpha_center, backend=backend)
    dMh_dt = mass_accretion_derivative_diff(Mh, alpha_center, Om, h0, z_bins,
                                            backend=backend)
    fstar = f_star_halo_diff(Mh, f_st, Mp, g1, g2, Mt, g3, g4, halo_mass_min,
                             backend=backend)
    fesc = f_esc_diff(Mh, f0_esc, Mp_esc, pl_esc, backend=backend)

    Ob = as_array(Ob, name, xp, device)
    Om = as_array(Om, name, xp, device)
    h0 = as_array(h0, name, xp, device)
    Nion = as_array(Nion, name, xp, device)

    return dMh_dt / h0 * fstar * Ob / Om * fesc * Nion / sec_per_year / m_H * M_sun
