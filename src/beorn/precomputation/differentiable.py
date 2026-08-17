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

X-ray heating (issue #59, Phase A) — :func:`eps_xray_diff` (port of
:func:`.astro.eps_xray`) and :func:`rho_xray_diff` (port of
:meth:`.solver.RadiationProfileSolver.rho_xray`) unblock gradients through
the X-ray channel, which was previously a fixed toy proxy; :func:`rho_heat_diff`
is a thin wrapper applying :func:`heat_ode_solution` to ``rho_xray_diff``'s
output, following :meth:`~.solver.RadiationProfileSolver.rho_heat`'s exact
prefactor construction. ``rho_xray_diff``'s data-dependent, per-redshift
lookback window is replaced by a fixed-size quadrature grid (``n_zprime``)
and its radial/optical-depth geometry is evaluated at static (non-
differentiated) cosmology — see its docstring's "Scope for this phase" for
the precise, deliberate list of what does and doesn't carry gradients.

Lyman-alpha coupling (issue #59, Phase B) — :func:`eps_lyal_diff` (port of
:func:`.couplings.eps_lyal`) and :func:`rho_alpha_profile_diff` (port of
:func:`.helpers.rho_alpha_profile`) unblock gradients through the Lyman-α
channel the same way; it sums the same fixed set of Lyman-series
recombination transitions production does, each with its own fixed-size
lookback-time quadrature, under the identical scope-for-this-phase
restrictions as :func:`rho_xray_diff` (static cosmology/flat ΛCDM, always
``z_source_start``, fixed node counts).

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
    get_backend, device_of, as_array, as_const, trapz_static, hubble_E, hubble_per_yr,
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
    'eps_xray_diff',
    'rho_xray_diff',
    'rho_heat_diff',
    'eps_lyal_diff',
    'rho_alpha_profile_diff',
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


# ──────────────────────────────────────────────────────────────────────────────
# X-ray heating (issue #59, Phase A) — real per-bin rho_xray/rho_heat
# ──────────────────────────────────────────────────────────────────────────────

def _linear_interp_weights(x_nodes, x_query, mode='extrapolate'):
    """Static linear-interpolation weight matrix: ``y_query ≈ W @ y_nodes``.

    ``x_nodes``/``x_query`` are plain numpy arrays (never carry gradients);
    the returned ``(n_query, n_nodes)`` matrix contracts against the
    *values* at the nodes, which may carry gradients — this is how
    interpolation from a differentiable-valued, but node-position-static,
    function is done throughout this module (a fixed-node counterpart of
    ``scipy.interpolate.interp1d``).

    ``mode='extrapolate'`` continues the boundary segment's slope past the
    ends (matches ``interp1d``'s default ``fill_value='extrapolate'``);
    ``mode='zero'`` zeroes out-of-range queries (matches
    ``fill_value=0.0, bounds_error=False``).
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    x_query = np.asarray(x_query, dtype=float)
    order = np.argsort(x_nodes)
    xs = x_nodes[order]
    n = xs.size
    idx = np.clip(np.searchsorted(xs, x_query, side='right') - 1, 0, n - 2)
    x0, x1 = xs[idx], xs[idx + 1]
    w = (x_query - x0) / (x1 - x0)
    W = np.zeros((x_query.size, n))
    rows = np.arange(x_query.size)
    W[rows, order[idx]] += (1.0 - w)
    W[rows, order[idx + 1]] += w
    if mode == 'zero':
        out_of_range = (x_query < xs[0]) | (x_query > xs[-1])
        W[out_of_range, :] = 0.0
    return W


def eps_xray_diff(nu_, xray_normalisation, alS_xray, energy_min_sed_xray,
                  energy_max_sed_xray, h0, backend='numpy'):
    """X-ray SED ε_X(ν) — differentiable counterpart of :func:`.astro.eps_xray`
    (power law, arXiv:1406.4120 Eq. 2).

    Differentiable w.r.t. ``xray_normalisation`` (i.e. ``log10_Lx``),
    ``alS_xray``, ``energy_min_sed_xray``, ``energy_max_sed_xray`` and ``h0``
    when ``backend='jax'``/``'torch'``.

    Args:
        nu_: Photon frequency [Hz] — a static numpy grid (the usual case
            inside :func:`rho_xray_diff`) or a backend array carrying
            gradients.
        xray_normalisation, alS_xray, energy_min_sed_xray, energy_max_sed_xray, h0:
            SED shape/normalisation parameters (scalars; may carry
            gradients).
        backend: 'numpy' (default, not differentiable), 'jax' or 'torch'.

    Returns:
        ε_X(ν) [photons/Hz/s/SFR], same shape as ``nu_``.
    """
    from ..constants import h_eV_sec, eV_per_erg

    name, xp = get_backend(backend)
    device = device_of(name, xp, xray_normalisation, alS_xray,
                       energy_min_sed_xray, energy_max_sed_xray, h0)
    nu_ = (as_const(np.asarray(nu_, dtype=float), name, xp, device)
          if isinstance(nu_, np.ndarray) else as_array(nu_, name, xp, device))
    xray_normalisation = as_array(xray_normalisation, name, xp, device)
    alS_xray = as_array(alS_xray, name, xp, device)
    energy_min_sed_xray = as_array(energy_min_sed_xray, name, xp, device)
    energy_max_sed_xray = as_array(energy_max_sed_xray, name, xp, device)
    h0 = as_array(h0, name, xp, device)

    norm_xray = (1.0 - alS_xray) / (
        (energy_max_sed_xray / h_eV_sec) ** (1.0 - alS_xray)
        - (energy_min_sed_xray / h_eV_sec) ** (1.0 - alS_xray)
    )
    return (xray_normalisation / h0 * eV_per_erg * norm_xray
            * nu_ ** (-alS_xray) / (nu_ * h_eV_sec))


def rho_xray_diff(z_bins, rr, Mh_center, alpha_center, Om, Ob, h0,
                  f_st, Mp, g1, g2, Mt, g3, g4, halo_mass_min,
                  xray_normalisation, alS_xray, energy_min_sed_xray,
                  energy_max_sed_xray, energy_cutoff_min_xray,
                  energy_cutoff_max_xray, HI_frac, xe, z_source_start,
                  backend='numpy', n_nu=50, n_zprime=64):
    """X-ray energy-deposition profile ρ_xray(r, z) — differentiable
    counterpart of :meth:`.solver.RadiationProfileSolver.rho_xray`, for one
    ``(Mh_center, alpha_center)`` mass-accretion bin (see
    :func:`mass_accretion_diff`; broadcasts over leading batch axes the same
    way).

    Gradients flow to the star-formation-efficiency shape parameters
    (``f_st``, ``Mp``, ``g1``, ``g2``, ``Mt``, ``g3``, ``g4``), the
    mass-accretion track (``Mh_center``, ``alpha_center``) and the X-ray SED
    shape (``xray_normalisation`` i.e. ``log10_Lx``, ``alS_xray``,
    ``energy_min_sed_xray``, ``energy_max_sed_xray``) when
    ``backend='jax'``/``'torch'``.

    Scope for this phase (documented, not silently assumed):

    - ``Om``, ``Ob``, ``h0`` are treated as **static** Python floats for the
      radial geometry (comoving distances, Hubble rate, optical depth) — the
      exit test targets X-ray/Lyα source parameters, not cosmology, and
      threading gradients through the dynamic (node-position) radial
      interpolation below would need a materially more complex primitive.
    - that geometry assumes flat ΛCDM (``w0=-1, wa=0``), the same
      pre-existing limitation :func:`bubble_radius_diff` already has via
      :func:`~beorn.cosmo.differentiable.hubble_E`.
    - the source lifetime is always ``z_source_start`` (i.e. the
      ``t_source_age=None`` default of
      :meth:`~.solver.RadiationProfileSolver._z_star_at`) — the variable,
      data-dependent lookback window ``source_age`` implies isn't
      jit-friendly and is out of scope here.
    - ``energy_cutoff_min_xray``/``energy_cutoff_max_xray`` (the frequency
      *integration range*) are static configuration, not differentiated —
      unlike ``energy_min_sed_xray``/``energy_max_sed_xray`` (the SED's own
      normalisation bounds inside :func:`eps_xray_diff`), which do carry
      gradients.
    - the lookback-time quadrature uses a fixed ``n_zprime`` node count
      (rather than production's data-dependent ``N_prime_i``) — the same
      fixed-node-quadrature trick :func:`~beorn.mass_function.differentiable.sigma2_M`
      already uses for σ²(M) — changing the discretisation slightly but
      keeping every traced shape static.

    Args:
        z_bins: Static, decreasing redshift grid (numpy), shape (n_z,).
        rr: Static radial grid (comoving cMpc/h) (numpy), shape (n_r,).
        Mh_center, alpha_center: Mass-accretion-track bin center — see
            :func:`mass_accretion_diff`.
        Om, Ob, h0: Cosmology (Python floats — see Scope above).
        f_st, Mp, g1, g2, Mt, g3, g4, halo_mass_min: Star-formation-efficiency
            shape — see :func:`~beorn.astro_differentiable.f_star_halo_diff`.
        xray_normalisation, alS_xray, energy_min_sed_xray, energy_max_sed_xray:
            X-ray SED shape — see :func:`eps_xray_diff`.
        energy_cutoff_min_xray, energy_cutoff_max_xray: Frequency
            integration bounds, eV (Python floats, static).
        HI_frac: Hydrogen fraction by number (Python float, static).
        xe: Free-electron-fraction history on ``z_bins`` (numpy, static),
            shape (n_z,).
        z_source_start: Source lifetime cutoff, static (see Scope above).
        backend: 'numpy' (default, not differentiable), 'jax' or 'torch'.
        n_nu: Number of frequency-integration nodes (static).
        n_zprime: Number of lookback-time quadrature nodes per redshift
            (static — see Scope above).

    Returns:
        rho_xray on ``(rr, z_bins)``, shape ``(..., n_r, n_z)`` — note this
        differs from :meth:`.solver.RadiationProfileSolver.rho_xray`'s
        ``(n_r, mass_bins, alpha_bins, n_z)`` axis order: batch dims come
        first here, matching this module's convention elsewhere (e.g.
        :func:`bubble_radius_diff`).
    """
    import types

    from ..constants import h_eV_sec, E_HI, E_HeI
    from ..cross_sections import sigma_HI, sigma_HeI
    from ..astro import f_Xh
    from ..cosmo import comoving_distance
    from .helpers import cum_optical_depth

    name, xp = get_backend(backend)
    device = device_of(name, xp, Mh_center, alpha_center, f_st, Mp, Mt,
                       xray_normalisation, alS_xray)

    z_np = np.asarray(z_bins, dtype=float)
    rr_np = np.asarray(rr, dtype=float)
    xe_np = np.asarray(xe, dtype=float)
    Om_f, Ob_f, h0_f = float(Om), float(Ob), float(h0)
    HI_frac_f = float(HI_frac)
    z_star = float(z_source_start)

    # ---- mass-accretion track (differentiable) ---------------------------
    Mh = mass_accretion_diff(z_np, Mh_center, alpha_center, backend=backend)
    dMh_dt = mass_accretion_derivative_diff(Mh, alpha_center, Om, h0, z_np,
                                            backend=backend)
    fstar = f_star_halo_diff(Mh, f_st, Mp, g1, g2, Mt, g3, g4, halo_mass_min,
                             backend=backend)
    Ob_b = as_array(Ob, name, xp, device)
    Om_b = as_array(Om, name, xp, device)
    M_star_dot = (Ob_b / Om_b) * fstar * dMh_dt                # (..., n_z)
    batch_shape = tuple(M_star_dot.shape[:-1])

    anchor = xp.zeros_like(M_star_dot[..., :1])
    M_star_dot_aug = (xp.cat([anchor, M_star_dot], dim=-1) if name == 'torch'
                      else xp.concatenate([anchor, M_star_dot], axis=-1))
    x_nodes_star = np.concatenate(([z_star], z_np))  # matches augmentation order

    def _zeros(shape):
        if name == 'torch':
            return xp.zeros(shape, dtype=M_star_dot.dtype, device=M_star_dot.device)
        return xp.zeros(shape)

    # ---- static frequency grid & prefactor (energy-only, no gradients) --
    nu_min = float(energy_cutoff_min_xray) / h_eV_sec
    nu_max = float(energy_cutoff_max_xray) / h_eV_sec
    nu_np = np.logspace(np.log(nu_min), np.log(nu_max), n_nu, base=np.e)

    f_He_bynumb = 1.0 - HI_frac_f
    nb0 = rhoc0 * Ob_f / (m_p_in_Msun * h0_f)
    nH0 = (1.0 - f_He_bynumb) * nb0
    nHe0 = f_He_bynumb * nb0
    prefactor_nu_np = (
        (nH0 / nb0) * sigma_HI(nu_np * h_eV_sec) * (nu_np * h_eV_sec - E_HI)
        + (nHe0 / nb0) * sigma_HeI(nu_np * h_eV_sec) * (nu_np * h_eV_sec - E_HeI)
    )
    prefactor_nu = as_const(prefactor_nu_np, name, xp, device)

    # Reuse the exact production geometry (comoving_distance/cum_optical_depth)
    # for the static (non-differentiated) radial/optical-depth pieces — a
    # lightweight stand-in for Parameters exposing only what they read.
    fake_params = types.SimpleNamespace(
        cosmology=types.SimpleNamespace(Om=Om_f, Ob=Ob_f, h0=h0_f, w0=-1.0, wa=0.0),
        solver=types.SimpleNamespace(HI_frac=HI_frac_f),
    )

    rho_slices = []
    for i, z in enumerate(z_np):
        if z > z_star:
            rho_slices.append(_zeros(batch_shape + (rr_np.size,)))
            continue

        z_prime = np.logspace(np.log(z), np.log(z_star), n_zprime, base=np.e)
        rcom_prime = comoving_distance(z_prime, fake_params) * h0_f
        tau_prime = cum_optical_depth(z_prime, nu_np * h_eV_sec, fake_params)   # (n_nu, n_zprime)
        nu_prime = nu_np[:, None] * (1.0 + z_prime)[None, :] / (1.0 + z)

        eps_X = eps_xray_diff(nu_prime, xray_normalisation, alS_xray,
                              energy_min_sed_xray, energy_max_sed_xray, h0,
                              backend=backend)                                 # (n_nu, n_zprime)
        atten = as_const(np.exp(-tau_prime), name, xp, device) * eps_X         # (n_nu, n_zprime)

        W_z = as_const(_linear_interp_weights(x_nodes_star, z_prime, mode='extrapolate'),
                       name, xp, device)                                       # (n_zprime, n_z+1)
        M_star_dot_zprime = xp.einsum('qn,...n->...q', W_z, M_star_dot_aug)    # (..., n_zprime)

        integral_factors = (
            atten.reshape((n_nu,) + (1,) * len(batch_shape) + (n_zprime,))
            * M_star_dot_zprime.reshape((1,) + batch_shape + (n_zprime,))
        )                                                                      # (n_nu, ..., n_zprime)

        W_r = as_const(_linear_interp_weights(rcom_prime, rr_np, mode='zero'),
                       name, xp, device)                                       # (n_r, n_zprime)
        integral_factors_r = xp.einsum('rq,n...q->n...r', W_r, integral_factors)  # (n_nu, ..., n_r)

        integrand = (prefactor_nu.reshape((n_nu,) + (1,) * len(batch_shape) + (1,))
                    * integral_factors_r)
        heat = trapz_static(integrand, nu_np, name, xp, axis=0)                # (..., n_r)

        fXh = float(f_Xh(xe_np[i]))
        prefac_r_np = (fXh / (4.0 * np.pi * (rr_np / (1.0 + z)) ** 2)
                      / (cm_per_Mpc / h0_f) ** 2)
        prefac_r = as_const(prefac_r_np.reshape((1,) * len(batch_shape) + (rr_np.size,)),
                            name, xp, device)
        rho_slices.append(heat * prefac_r)

    return xp.stack(rho_slices, dim=-1) if name == 'torch' else xp.stack(rho_slices, axis=-1)


def rho_heat_diff(z_bins, rho_xray, Om, h0, z_decoupling, backend='numpy'):
    """Heating-rate integral ρ_heat(r, z) — differentiable counterpart of
    :meth:`.solver.RadiationProfileSolver.rho_heat`.

    A thin wrapper around :func:`heat_ode_solution`:
    :meth:`~.solver.RadiationProfileSolver.rho_heat` solves
    ``dy/da = γ(a) − 2y/a`` via ``solve_ivp`` — exactly the ODE
    :func:`heat_ode_solution` already solves in closed form (see the module
    docstring) — so this just builds ``γ(a)`` on the same node grid
    (``z_decoupling`` prepended, matching production's zero-heating anchor)
    and calls it. Differentiable w.r.t. ``rho_xray`` (e.g. the output of
    :func:`rho_xray_diff`) when ``backend='jax'``/``'torch'``.

    ``Om``, ``h0`` are treated as static Python floats, matching
    :func:`rho_xray_diff`'s scope for this phase.

    Args:
        z_bins: Static, decreasing redshift grid (numpy), shape (n_z,) —
            must match the grid ``rho_xray`` was computed on.
        rho_xray: Output of :func:`rho_xray_diff` (or any backend array with
            the same ``(..., n_z)`` trailing shape).
        Om, h0: Cosmology (Python floats — see Scope above).
        z_decoupling: Redshift anchored as the zero-heating initial
            condition, static (``parameters.solver.z_decoupling``).
        backend: 'numpy' (default, not differentiable), 'jax' or 'torch'.

    Returns:
        rho_heat on ``z_bins``, same shape as ``rho_xray``.
    """
    from ..constants import kb_eV_per_K

    name, xp = get_backend(backend)
    device = device_of(name, xp, rho_xray)
    rho_xray = as_array(rho_xray, name, xp, device)

    z_np = np.asarray(z_bins, dtype=float)
    zz = np.insert(z_np, 0, float(z_decoupling))
    a_nodes = 1.0 / (1.0 + zz)

    anchor = xp.zeros_like(rho_xray[..., :1])
    rho_xray_full = (xp.cat([anchor, rho_xray], dim=-1) if name == 'torch'
                     else xp.concatenate([anchor, rho_xray], axis=-1))

    Hz_np = 100.0 * float(h0) * np.sqrt(float(Om) * (1.0 + zz) ** 3 + (1.0 - float(Om)))
    a_b = as_const(a_nodes, name, xp, device)
    Hz_b = as_const(Hz_np, name, xp, device)

    gamma = 2.0 * rho_xray_full / (3.0 * kb_eV_per_K * a_b * Hz_b) * km_per_Mpc
    y = heat_ode_solution(a_nodes, gamma, backend=backend)
    return y[..., 1:]


# ──────────────────────────────────────────────────────────────────────────────
# Lyman-alpha coupling (issue #59, Phase B) — real per-bin rho_alpha
# ──────────────────────────────────────────────────────────────────────────────

_REC_FRAC_CACHE = None


def _rec_frac():
    """Load & cache ``input_data/recfrac.dat`` (static recombination-fraction
    table) — the same file :func:`.helpers.rho_alpha_profile` reads."""
    global _REC_FRAC_CACHE
    if _REC_FRAC_CACHE is None:
        import importlib.util
        from pathlib import Path
        path_to_file = (Path(importlib.util.find_spec('beorn').origin).parent
                        / 'input_data' / 'recfrac.dat')
        _REC_FRAC_CACHE = np.genfromtxt(path_to_file, usecols=(0, 1),
                                        comments='#', dtype=float, names='n, f')
    return _REC_FRAC_CACHE


def eps_lyal_diff(nu_, n_lyman_alpha_photons, lyman_alpha_power_law, h0,
                  backend='numpy'):
    """Lyman-α SED ε_α(ν) — differentiable counterpart of
    :func:`.couplings.eps_lyal` (power law, BEoRN paper Eq. 8).

    Differentiable w.r.t. ``n_lyman_alpha_photons``, ``lyman_alpha_power_law``
    and ``h0`` when ``backend='jax'``/``'torch'``.

    Args:
        nu_: Photon frequency [Hz] — a static numpy grid (the usual case
            inside :func:`rho_alpha_profile_diff`) or a backend array
            carrying gradients.
        n_lyman_alpha_photons, lyman_alpha_power_law, h0: SED
            shape/normalisation parameters (scalars; may carry gradients).
        backend: 'numpy' (default, not differentiable), 'jax' or 'torch'.

    Returns:
        ε_α(ν) [photons/yr/Hz/SFR], same shape as ``nu_``.
    """
    from ..constants import nu_al, nu_LL

    name, xp = get_backend(backend)
    device = device_of(name, xp, n_lyman_alpha_photons, lyman_alpha_power_law, h0)
    nu_ = (as_const(np.asarray(nu_, dtype=float), name, xp, device)
          if isinstance(nu_, np.ndarray) else as_array(nu_, name, xp, device))
    N_al = as_array(n_lyman_alpha_photons, name, xp, device)
    alS = as_array(lyman_alpha_power_law, name, xp, device)
    h0 = as_array(h0, name, xp, device)

    Anorm = (1.0 - alS) / (nu_LL ** (1.0 - alS) - nu_al ** (1.0 - alS))
    return Anorm * nu_ ** (-alS) * N_al / (m_p_in_Msun * h0)


def rho_alpha_profile_diff(z_bins, r_grid, Mh_center, alpha_center, Om, Ob, h0,
                           f_st, Mp, g1, g2, Mt, g3, g4, halo_mass_min,
                           n_lyman_alpha_photons, lyman_alpha_power_law,
                           z_source_start, backend='numpy', n_zprime=64,
                           rectrunc=23):
    """Lyman-alpha coupling profile ρ_alpha(r, z) — differentiable
    counterpart of :func:`.helpers.rho_alpha_profile`, for one
    ``(Mh_center, alpha_center)`` mass-accretion bin (see
    :func:`mass_accretion_diff`; broadcasts over leading batch axes the same
    way as :func:`rho_xray_diff`).

    Gradients flow to the star-formation-efficiency shape parameters
    (``f_st``, ``Mp``, ``g1``, ``g2``, ``Mt``, ``g3``, ``g4``), the
    mass-accretion track (``Mh_center``, ``alpha_center``) and the Lyman-α
    SED shape (``n_lyman_alpha_photons``, ``lyman_alpha_power_law``) when
    ``backend='jax'``/``'torch'``.

    Sums the same ``rectrunc - 2`` Lyman-series recombination transitions
    production does (``n = 2 .. rectrunc - 1``, weighted by
    ``input_data/recfrac.dat``'s tabulated, non-differentiated recombination
    fractions); each transition gets its own fixed-size (``n_zprime``)
    lookback-time quadrature. See :func:`rho_xray_diff`'s "Scope for this
    phase" docstring section, which applies here identically: ``Om``/``Ob``/
    ``h0`` are static floats (flat ΛCDM geometry), the source lifetime is
    always ``z_source_start``, and quadrature node counts are fixed rather
    than production's data-dependent ones.

    Args:
        z_bins: Static, decreasing redshift grid (numpy), shape (n_z,).
        r_grid: Static *physical* radial grid (pMpc/h) (numpy), shape
            (n_r,) — matches :meth:`.solver.RadiationProfileSolver.solve`'s
            ``r_lyal``, not the comoving grid :func:`rho_xray_diff` uses.
        Mh_center, alpha_center: Mass-accretion-track bin center — see
            :func:`mass_accretion_diff`.
        Om, Ob, h0: Cosmology (Python floats — static, see
            :func:`rho_xray_diff`'s Scope).
        f_st, Mp, g1, g2, Mt, g3, g4, halo_mass_min: Star-formation-efficiency
            shape — see :func:`~beorn.astro_differentiable.f_star_halo_diff`.
        n_lyman_alpha_photons, lyman_alpha_power_law: Lyman-α SED shape —
            see :func:`eps_lyal_diff`.
        z_source_start: Source lifetime cutoff, static.
        backend: 'numpy' (default, not differentiable), 'jax' or 'torch'.
        n_zprime: Number of lookback-time quadrature nodes per transition
            per redshift (static).
        rectrunc: Number of Lyman-series transitions to sum (``n = 2 ..
            rectrunc - 1``), matching production's default of 23.

    Returns:
        rho_alpha on ``(r_grid, z_bins)`` [pcm⁻².s⁻¹.Hz⁻¹], shape
        ``(..., n_r, n_z)`` — see :func:`rho_xray_diff`'s Returns note on
        axis-order convention.
    """
    import types

    from ..constants import nu_LL
    from ..cosmo import comoving_distance

    name, xp = get_backend(backend)
    device = device_of(name, xp, Mh_center, alpha_center, f_st, Mp, Mt,
                       n_lyman_alpha_photons, lyman_alpha_power_law)

    z_np = np.asarray(z_bins, dtype=float)
    r_np = np.asarray(r_grid, dtype=float)
    Om_f, Ob_f, h0_f = float(Om), float(Ob), float(h0)
    z_star = float(z_source_start)

    rec = _rec_frac()
    nu_n = nu_LL * (1.0 - 1.0 / rec['n'] ** 2)
    nu_n = np.where(nu_n == 0, np.inf, nu_n)

    # ---- mass-accretion track (differentiable), same construction as
    # rho_xray_diff's M_star_dot --------------------------------------------
    Mh = mass_accretion_diff(z_np, Mh_center, alpha_center, backend=backend)
    dMh_dt = mass_accretion_derivative_diff(Mh, alpha_center, Om, h0, z_np,
                                            backend=backend)
    fstar = f_star_halo_diff(Mh, f_st, Mp, g1, g2, Mt, g3, g4, halo_mass_min,
                             backend=backend)
    Ob_b = as_array(Ob, name, xp, device)
    Om_b = as_array(Om, name, xp, device)
    M_star_dot = (Ob_b / Om_b) * fstar * dMh_dt                # (..., n_z)
    batch_shape = tuple(M_star_dot.shape[:-1])

    anchor = xp.zeros_like(M_star_dot[..., :1])
    M_star_dot_aug = (xp.cat([anchor, M_star_dot], dim=-1) if name == 'torch'
                      else xp.concatenate([anchor, M_star_dot], axis=-1))
    x_nodes_star = np.concatenate(([z_star], z_np))

    def _zeros(shape):
        if name == 'torch':
            return xp.zeros(shape, dtype=M_star_dot.dtype, device=M_star_dot.device)
        return xp.zeros(shape)

    fake_params = types.SimpleNamespace(
        cosmology=types.SimpleNamespace(Om=Om_f, Ob=Ob_f, h0=h0_f, w0=-1.0, wa=0.0),
    )

    n_transitions = rectrunc - 2
    rho_slices = []
    for i, z in enumerate(z_np):
        if z > z_star:
            rho_slices.append(_zeros(batch_shape + (r_np.size,)))
            continue

        flux = _zeros(batch_shape + (r_np.size,))
        r_query = r_np * (1.0 + z)
        for k in range(n_transitions):
            n_k = float(rec['n'][k + 2])
            z_max_k = min((1.0 - (n_k + 1.0) ** -2) / (1.0 - n_k ** -2) * (1.0 + z) - 1.0,
                         z_star)
            if z_max_k <= z:
                continue

            z_prime = np.logspace(np.log(z), np.log(z_max_k), n_zprime, base=np.e)
            rcom_prime = comoving_distance(z_prime, fake_params) * h0_f
            nu_prime = nu_n[k + 2] * (1.0 + z_prime) / (1.0 + z)

            W_z = as_const(_linear_interp_weights(x_nodes_star, z_prime, mode='extrapolate'),
                           name, xp, device)
            M_star_dot_zprime = xp.einsum('qn,...n->...q', W_z, M_star_dot_aug)  # (..., n_zprime)

            eps_al = eps_lyal_diff(nu_prime, n_lyman_alpha_photons,
                                   lyman_alpha_power_law, h0, backend=backend)
            eps_al = (eps_al.reshape((1,) * len(batch_shape) + (n_zprime,))
                     * M_star_dot_zprime)                                       # (..., n_zprime)

            W_r = as_const(_linear_interp_weights(rcom_prime, r_query, mode='zero'),
                           name, xp, device)                                    # (n_r, n_zprime)
            flux_k = xp.einsum('rq,...q->...r', W_r, eps_al) * float(rec['f'][k + 2])
            flux = flux + flux_k

        prefac_r_np = (1.0 / (4.0 * np.pi * r_np ** 2) * (h0_f / cm_per_Mpc) ** 2
                      / sec_per_year)
        prefac_r = as_const(prefac_r_np.reshape((1,) * len(batch_shape) + (r_np.size,)),
                            name, xp, device)
        rho_slices.append(flux * prefac_r)

    return xp.stack(rho_slices, dim=-1) if name == 'torch' else xp.stack(rho_slices, axis=-1)
