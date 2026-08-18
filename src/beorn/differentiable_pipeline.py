"""Differentiable multi-redshift driver — global dTb(z)/xHII(z)/Tk(z) history.

Loops the single-z differentiable chain (ICs → LPT density → EPS halo field
→ Ngam_dot(z)/R_bubble(z) → painting → couplings → dTb) over a redshift grid
and reduces each snapshot to its spatial mean — the differentiable,
backend-generic counterpart of
:meth:`beorn.structs.temporal_cube.TemporalCube.global_mean`. A plain Python
loop, no ``lax.scan``/``vmap`` — matching every other multi-z loop in this
codebase (production's ``PaintingCoordinator.paint_simple_loop``, every
existing test): issue #42's own stated policy is to extend the existing
pattern, not invent a new one.

``Ngam_dot(z)``/``R_bubble(z)`` (:func:`beorn.precomputation.differentiable.ngam_dot_ion_population_diff`/
:func:`~beorn.precomputation.differentiable.bubble_radius_diff`, issue #59
Phase C) are solved **once** over the whole redshift grid and indexed per
snapshot — not re-solved every iteration. ``Ngam_dot(z)`` is an
abundance-weighted mean over a static halo-mass grid (the differentiable
HMF as the weight), not one hand-picked representative halo — see
:func:`~beorn.precomputation.differentiable.ngam_dot_ion_population_diff`'s
docstring for exactly what that does and doesn't average over.

:func:`paint_snapshot_diff`/:func:`dtb_global_signal_diff` above stay the
*interim* single-global-bubble chain: real Ly-alpha
(:func:`beorn.precomputation.helpers.rho_alpha_profile`) and X-ray heating
(:meth:`beorn.precomputation.solver.RadiationProfileSolver.rho_xray`) source
terms are **not** used there — both channels still use the same toy proxies
as the issue #42 exit test (a fixed radial heating profile, a Ly-alpha
coupling proportional to the halo mesh).

:func:`paint_snapshot_population_diff`/:func:`dtb_global_signal_population_diff`
(issue #59, Phase D) are the **true differentiable twin**: every mass bin
gets its own bubble radius (:func:`~beorn.precomputation.differentiable.bubble_radius_diff`),
X-ray heating profile (:func:`~beorn.precomputation.differentiable.rho_xray_diff`/
:func:`~beorn.precomputation.differentiable.rho_heat_diff`, Phase A) and
Lyman-alpha profile (:func:`~beorn.precomputation.differentiable.rho_alpha_profile_diff`,
Phase B), Fourier-accumulated by
:func:`~beorn.painting.differentiable.paint_fields_population_diff` instead
of one hand-picked representative halo painted with a toy heating/coupling
proxy — unifying Phases A-C into the real ``paint_full`` per-bin
architecture. The interim functions above are kept, unchanged, for the
existing issue #42 exit test and anything else that only needs the
ionization channel.

``backend='numpy'`` is the default and is **not** differentiable (plain
NumPy has no autodiff); use ``backend='jax'``/``'torch'`` for gradients.
"""
from __future__ import annotations

import numpy as np

from .cosmo.differentiable import get_backend, dTb_factor
from .constants import rhoc0, Tcmb0, M_sun, cm_per_Mpc, m_H
from .lpt import lpt_density, lpt_linear_density
from .lpt.chmf import halo_field_diff, chmf_mass_bins
from .precomputation.differentiable import (
    ngam_dot_ion_diff, ngam_dot_ion_population_diff, bubble_radius_diff,
    rho_xray_diff, rho_heat_diff, rho_alpha_profile_diff,
)
from .painting.differentiable import paint_fields_diff, paint_fields_population_diff
from .couplings import x_coll_diff, s_alpha_diff, dtb_diff

__all__ = [
    'paint_snapshot_diff', 'dtb_global_signal_diff',
    'paint_snapshot_population_diff', 'dtb_global_signal_population_diff',
]

_DEFAULT_R_NODES = np.linspace(1e-3, 20.0, 60)


def paint_snapshot_diff(
    z, dk, R_bubble, L, N, Om, Ob, h0, ns, sigma_8,
    Mh_center=1e10, cell_volume=None, M_min=1e9, n_mass_bins=8,
    weights='counts', eps_halo=None,
    r_temp=None, prof_temp=None, xHII_floor=1e-4, spread_iter=4,
    xal_coupling=1e-2, backend='numpy',
):
    """One redshift snapshot of the differentiable 21-cm chain.

    Painted density → EPS halo field → ionization/heating painting →
    couplings → dTb. Shared by :func:`dtb_global_signal_diff` (the multi-z
    driver) and the issue #42 exit test
    (``tests/test_differentiable_21cm.py::_delta2_21``), so both run
    provably identical per-snapshot physics rather than two independently
    maintained copies of the same formulas.

    Args:
        z: Redshift (static float).
        dk: LPT initial-conditions Fourier field (backend array, from
            :func:`beorn.lpt.lpt_ics` — computed once, outside any z-loop).
        R_bubble: Comoving bubble radius at this z (Mpc/h; may carry
            gradients — e.g. one element of
            :func:`~beorn.precomputation.differentiable.bubble_radius_diff`'s
            output).
        L, N: Box size (Mpc/h) and cells per side.
        Om, Ob, h0, ns, sigma_8: Cosmology (may carry gradients).
        Mh_center: Representative EPS environment mass, Msun/h — passed as
            :func:`beorn.lpt.chmf.halo_field_diff`'s ``M_env``.
        cell_volume: Cell volume (Mpc/h)³ — ``None`` → ``(L/N)**3``.
        M_min, n_mass_bins, weights, eps_halo: Forwarded to
            :func:`beorn.lpt.chmf.halo_field_diff`.
        r_temp, prof_temp: Radial nodes/values of the (toy) heating profile —
            ``None`` → the same fixed toy profile as the issue #42 exit test
            (``100 * exp(-r/3)`` on 60 nodes out to 20 Mpc/h). Real X-ray
            heating is not implemented (see module docstring).
        xHII_floor, spread_iter: Forwarded to
            :func:`beorn.painting.differentiable.paint_fields_diff`.
        xal_coupling: Toy Ly-alpha coupling proportionality constant
            (``x_al = xal_coupling * halo_mesh``) — real Ly-alpha is not
            implemented (see module docstring).
        backend: 'numpy' (default, not differentiable), 'jax' or 'torch'.

    Returns:
        (delta_b, dTb, xHII, Tk) — each a backend array of shape (N, N, N).
    """
    name, xp = get_backend(backend)
    cell = float(L) / N
    if cell_volume is None:
        cell_volume = cell ** 3
    R_cell = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * cell
    if r_temp is None:
        r_temp = _DEFAULT_R_NODES
    if prof_temp is None:
        prof_temp = 100.0 * np.exp(-np.asarray(r_temp) / 3.0)

    delta_b = lpt_density(dk, L, z, Om, backend=backend)
    dlin = lpt_linear_density(dk, L, z, Om, backend=backend, R_tophat=R_cell)

    M_env = 4.0 / 3.0 * np.pi * R_cell ** 3 * rhoc0 * Om
    hmesh, _ = halo_field_diff(dlin, M_env, z, Om, Ob, h0, ns, sigma_8,
                               cell_volume=cell_volume, M_min=M_min,
                               n_mass_bins=n_mass_bins, weights=weights,
                               eps=eps_halo, backend=backend)

    xhii, _, dT = paint_fields_diff(hmesh, z, L, R_bubble=R_bubble,
                                    r_temp=r_temp, prof_temp=prof_temp,
                                    backend=backend, xHII_floor=xHII_floor,
                                    spread_iter=spread_iter)

    dT_pos = xp.where(dT > 0, dT, xp.zeros_like(dT))
    Tk = dT_pos + 2.0                                # adiabatic floor
    xal = xal_coupling * hmesh                        # toy Ly-alpha coupling
    xtot = xal * s_alpha_diff(z, Tk, 1 - xhii, backend=backend) \
        + x_coll_diff(z, Tk, 1 - xhii, 1e-3, backend=backend)
    dTb = dtb_diff(z, Tk, xtot, delta_b, xhii, factor=27.0, backend=backend)

    return delta_b, dTb, xhii, Tk


def dtb_global_signal_diff(
    z_grid, dk, L, N, Om, Ob, h0, ns, sigma_8,
    Nion, f_st, Mp, g1, g2, Mt, g3, g4, halo_mass_min,
    f0_esc, Mp_esc, pl_esc,
    Mh_center=1e10, alpha_center=0.79, ngam_mass_bins=None,
    cell_volume=None, M_min=1e9, n_mass_bins=8, weights='counts',
    eps_halo=None, r_temp=None, prof_temp=None, xHII_floor=1e-4,
    spread_iter=4, xal_coupling=1e-2, backend='numpy',
):
    """Differentiable global dTb(z)/xHII(z)/Tk(z) history.

    The differentiable, backend-generic counterpart of
    :meth:`beorn.structs.temporal_cube.TemporalCube.global_mean`: loops
    :func:`paint_snapshot_diff` over ``z_grid`` and reduces each snapshot to
    its spatial mean. Gradients reach every cosmology parameter and every
    astro parameter
    :func:`beorn.precomputation.differentiable.ngam_dot_ion_population_diff`
    reaches (``Nion``, ``f_st``, ``Mp``, ``g1``, ``g2``, ``Mt``, ``g3``,
    ``g4``, ``f0_esc``, ``Mp_esc``, ``pl_esc``, plus ``ns``/``sigma_8``
    through the HMF weight) through the ionization channel, when
    ``backend='jax'``/``'torch'``.

    The same fixed ``dk`` and the same ``eps_halo`` shot-noise realization
    are reused at every redshift — the pragmatic first pass; independent
    per-z noise is a trivial later change, not addressed here.

    Args:
        z_grid: Static redshift grid (numpy), **decreasing** — matching
            :func:`~beorn.precomputation.differentiable.bubble_radius_diff`'s/
            production's ``solver.z_bins`` convention (a increasing);
            passing an increasing grid integrates the photon-rate ODE
            backward in time and gives wrong physics. Shape (n_z,).
        dk: LPT initial-conditions Fourier field — see :func:`paint_snapshot_diff`.
        L, N: Box size (Mpc/h) and cells per side.
        Om, Ob, h0, ns, sigma_8: Cosmology (may carry gradients).
        Nion, f_st, Mp, g1, g2, Mt, g3, g4, halo_mass_min, f0_esc, Mp_esc, pl_esc:
            Astro parameters — see
            :func:`beorn.precomputation.differentiable.ngam_dot_ion_population_diff`
            (may carry gradients).
        Mh_center: Representative EPS environment mass for the halo *field*
            (painting) only — unrelated to the ionizing-budget calculation
            below, see :func:`paint_snapshot_diff`.
        alpha_center: Representative accretion-rate exponent used for every
            mass bin in the ionizing-budget calculation — see
            :func:`~beorn.precomputation.differentiable.ngam_dot_ion_population_diff`.
        ngam_mass_bins: Static halo-mass quadrature grid for the ionizing
            budget, Msun/h — ``None`` → ``np.logspace(np.log10(halo_mass_min),
            13.0, 50)`` (the star-forming range up to a generous bright-end
            cutoff). See
            :func:`~beorn.precomputation.differentiable.ngam_dot_ion_population_diff`.
        cell_volume, M_min, n_mass_bins, weights, eps_halo, r_temp, prof_temp,
            xHII_floor, spread_iter, xal_coupling: Forwarded to
            :func:`paint_snapshot_diff` at every redshift.
        backend: 'numpy' (default, not differentiable), 'jax' or 'torch'.

    Returns:
        (dTb_mean, xHII_mean, Tk_mean) — each a backend array of shape
        (n_z,), in ``z_grid``'s order.
    """
    name, xp = get_backend(backend)

    if ngam_mass_bins is None:
        ngam_mass_bins = np.logspace(np.log10(halo_mass_min), 13.0, 50)

    Ngam = ngam_dot_ion_population_diff(
        z_grid, ngam_mass_bins, alpha_center, Om, Ob, h0, ns, sigma_8, Nion,
        f_st, Mp, g1, g2, Mt, g3, g4, halo_mass_min, f0_esc, Mp_esc, pl_esc,
        backend=backend,
    )
    R_b = bubble_radius_diff(z_grid, Ngam, Om, Ob, h0, backend=backend)

    dTb_hist, xHII_hist, Tk_hist = [], [], []
    for i, z in enumerate(np.asarray(z_grid, dtype=float)):
        _, dTb, xhii, Tk = paint_snapshot_diff(
            float(z), dk, R_b[i], L, N, Om, Ob, h0, ns, sigma_8,
            Mh_center=Mh_center, cell_volume=cell_volume, M_min=M_min,
            n_mass_bins=n_mass_bins, weights=weights, eps_halo=eps_halo,
            r_temp=r_temp, prof_temp=prof_temp, xHII_floor=xHII_floor,
            spread_iter=spread_iter, xal_coupling=xal_coupling,
            backend=backend,
        )
        dTb_hist.append(xp.mean(dTb))
        xHII_hist.append(xp.mean(xhii))
        Tk_hist.append(xp.mean(Tk))

    return xp.stack(dTb_hist), xp.stack(xHII_hist), xp.stack(Tk_hist)


# ──────────────────────────────────────────────────────────────────────────────
# Full per-bin differentiable twin (issue #59, Phase D)
# ──────────────────────────────────────────────────────────────────────────────

def paint_snapshot_population_diff(
    z, dk, L, N, Om, Ob, h0, ns, sigma_8,
    R_bubble_bins, prof_temp_bins, r_temp,
    prof_alpha_bins, r_alpha,
    z_decoupling,
    M_env=1e10, cell_volume=None, M_min=1e9, n_mass_bins=8,
    eps_halo=None, xHII_floor=1e-4, spread_iter=4, backend='numpy',
):
    """One redshift snapshot of the full per-bin differentiable 21-cm chain
    (issue #59, Phase D) — the real-physics counterpart of
    :func:`paint_snapshot_diff`.

    Density → EPS halo field, decomposed **per mass bin**
    (:func:`beorn.lpt.chmf.halo_field_diff` with ``return_bins=True``) →
    per-bin Fourier-accumulated painting
    (:func:`beorn.painting.differentiable.paint_fields_population_diff`,
    using each bin's own ``R_bubble``/X-ray-heating/Lyman-α profile, solved
    once over the whole redshift grid by
    :func:`dtb_global_signal_population_diff` and indexed in here) →
    couplings → dTb, with every constant/scaling the interim
    :func:`paint_snapshot_diff` used a toy value for computed for real:
    the adiabatic baseline (:func:`~beorn.cosmo.background.T_adiab_fluctu`'s
    formula), the collisional-coupling baryon density
    (:func:`~beorn.structs.derived_quantities.GridDerivedPropertiesMixin.coef`'s
    formula), the ``S_alpha``/4π normalisation production applies after
    painting, and the cosmology-dependent ``dTb_factor``
    (:func:`~beorn.cosmo.differentiable.dTb_factor`) instead of a fixed 27.0.

    Args:
        z: Redshift (static float).
        dk: LPT initial-conditions Fourier field.
        L, N: Box size (Mpc/h) and cells per side.
        Om, Ob, h0, ns, sigma_8: Cosmology (may carry gradients).
        R_bubble_bins: Comoving bubble radius per mass bin (Mpc/h), shape
            (n_mass_bins,) — one z-slice of
            :func:`~beorn.precomputation.differentiable.bubble_radius_diff`'s
            output; may carry gradients.
        prof_temp_bins: X-ray heating/temperature profile per mass bin on
            ``r_temp``, shape (n_mass_bins, n_r) — one z-slice of
            :func:`~beorn.precomputation.differentiable.rho_heat_diff`'s
            output; may carry gradients.
        r_temp: Comoving radial nodes (Mpc/h) for ``prof_temp_bins`` — the
            solver's ``r_grid_cell``.
        prof_alpha_bins: Lyman-α profile per mass bin on ``r_alpha``, shape
            (n_mass_bins, n_r_lyal) — one z-slice of
            :func:`~beorn.precomputation.differentiable.rho_alpha_profile_diff`'s
            output; may carry gradients.
        r_alpha: Comoving radial nodes (Mpc/h) for ``prof_alpha_bins`` — note
            this is **not** the same grid ``prof_alpha_bins`` was computed on:
            :func:`~beorn.precomputation.differentiable.rho_alpha_profile_diff`'s
            ``r_grid`` (the solver's ``r_lyal``) is *physical*, so the caller
            must pass ``r_lyal * (1 + z)`` here — exactly what
            :func:`dtb_global_signal_population_diff` does — mirroring
            :meth:`.coordinator.PaintingCoordinator.paint_single_mass_bin`'s
            own ``r_lyal * (1 + z)`` conversion before building its
            Lyman-alpha kernel.
        z_decoupling: Redshift where the gas adiabatically decouples from
            the CMB (``parameters.solver.z_decoupling``) — sets the
            adiabatic-cooling baseline.
        M_env, cell_volume, M_min, n_mass_bins, eps_halo: Forwarded to
            :func:`beorn.lpt.chmf.halo_field_diff` — ``n_mass_bins`` and
            ``M_min``/``M_env`` must match whatever was used to build
            ``R_bubble_bins``/``prof_temp_bins``/``prof_alpha_bins``'s mass
            grid (:func:`beorn.lpt.chmf.chmf_mass_bins`) — mismatched grids
            silently pair the wrong bin with the wrong profile.
        xHII_floor, spread_iter: Forwarded to
            :func:`~beorn.painting.differentiable.paint_fields_population_diff`.
        backend: 'numpy' (default, not differentiable), 'jax' or 'torch'.

    Returns:
        (delta_b, dTb, xHII, Tk) — each a backend array of shape (N, N, N).
    """
    name, xp = get_backend(backend)
    cell = float(L) / N
    if cell_volume is None:
        cell_volume = cell ** 3
    R_cell = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * cell

    delta_b = lpt_density(dk, L, z, Om, backend=backend)
    dlin = lpt_linear_density(dk, L, z, Om, backend=backend, R_tophat=R_cell)

    _, _, n_b_bins = halo_field_diff(
        dlin, M_env, z, Om, Ob, h0, ns, sigma_8, cell_volume=cell_volume,
        M_min=M_min, n_mass_bins=n_mass_bins, weights='counts', eps=eps_halo,
        backend=backend, return_bins=True,
    )

    grid_xhii, grid_xal, grid_temp = paint_fields_population_diff(
        n_b_bins, z, L, R_bubble_bins=R_bubble_bins,
        r_alpha=r_alpha, prof_alpha_bins=prof_alpha_bins,
        r_temp=r_temp, prof_temp_bins=prof_temp_bins,
        backend=backend, xHII_floor=xHII_floor, spread_iter=spread_iter,
    )

    # T_adiab_fluctu(z, z_decoupling, delta_b) -- beorn.cosmo.background's
    # formula, ported inline (2-line closed form, not worth a new module fn).
    T_adiab = Tcmb0 * (1.0 + z) ** 2 / (1.0 + z_decoupling)
    Tk = T_adiab * (1.0 + delta_b) ** (2.0 / 3.0) + grid_temp

    # coef/rho_b -- beorn.structs.derived_quantities.GridDerivedPropertiesMixin.coef's
    # formula, ported inline for the same reason. ``coef`` is assembled from
    # scalars *before* touching ``delta_b`` (matching production's own
    # scalar-then-array structure) -- delta_b is float32 (from lpt_density's
    # particle-mesh painting), and the constant chain here spans ~1e-44
    # (M_sun/cm_per_Mpc**3) to ~1e30 (M_sun): folding a float32 array into
    # that chain overflows intermediate float32 terms to inf, and inf/inf
    # gives NaN. Computed as plain scalars this chain stays in float64/Python
    # float precision throughout, and only the final, moderate (~1e-4-1e-3)
    # coefficient ever multiplies the array.
    coef = rhoc0 * h0 ** 2 * Ob * (1.0 + z) ** 3 * M_sun / cm_per_Mpc ** 3 / m_H
    rho_b = (1.0 + delta_b) * coef
    x_alpha = (1.81e11 / (1.0 + z)) * grid_xal
    xtot = (x_alpha * s_alpha_diff(z, Tk, 1.0 - grid_xhii, backend=backend)
           / (4.0 * np.pi)
           + x_coll_diff(z, Tk, 1.0 - grid_xhii, rho_b, backend=backend))

    factor = dTb_factor(Om, Ob, h0, backend=backend)
    dTb = dtb_diff(z, Tk, xtot, delta_b, grid_xhii, factor=factor, backend=backend)

    return delta_b, dTb, grid_xhii, Tk


def dtb_global_signal_population_diff(
    z_grid, dk, L, N, Om, Ob, h0, ns, sigma_8,
    Nion, f_st, Mp, g1, g2, Mt, g3, g4, halo_mass_min,
    f0_esc, Mp_esc, pl_esc,
    xray_normalisation, alS_xray, energy_min_sed_xray, energy_max_sed_xray,
    energy_cutoff_min_xray, energy_cutoff_max_xray, HI_frac,
    n_lyman_alpha_photons, lyman_alpha_power_law,
    z_source_start, z_decoupling,
    alpha_center=0.79, M_env=1e10, cell_volume=None, M_min=1e9,
    n_mass_bins=8, eps_halo=None, r_grid_cell=None, r_lyal=None,
    xHII_floor=1e-4, spread_iter=4, backend='numpy',
):
    """Full per-bin differentiable global dTb(z)/xHII(z)/Tk(z) history —
    the true differentiable twin (issue #59, Phase D) of
    :func:`dtb_global_signal_diff`.

    Every mass bin (:func:`beorn.lpt.chmf.chmf_mass_bins`, the same static
    grid :func:`beorn.lpt.chmf.halo_field_diff` bins its expected halo
    counts into) gets its own ``Ngam_dot(z)``/``R_bubble(z)``
    (:func:`~beorn.precomputation.differentiable.ngam_dot_ion_diff`/
    :func:`~beorn.precomputation.differentiable.bubble_radius_diff`),
    X-ray heating profile
    (:func:`~beorn.precomputation.differentiable.rho_xray_diff`/
    :func:`~beorn.precomputation.differentiable.rho_heat_diff`) and
    Lyman-α profile
    (:func:`~beorn.precomputation.differentiable.rho_alpha_profile_diff`) —
    all solved **once** over the whole ``z_grid`` (batched over the mass
    axis, the same "solved once, indexed per snapshot" architecture
    :func:`dtb_global_signal_diff` already uses for its single global
    ``Ngam_dot``/``R_bubble``) and then painted per snapshot by
    :func:`paint_snapshot_population_diff`.

    Scope: ``alpha_center`` stays one representative accretion-rate exponent
    for every mass bin, for exactly the reason
    :func:`~beorn.precomputation.differentiable.ngam_dot_ion_population_diff`
    documents (no accretion-rate-scatter analogue for a smooth analytic/CHMF
    halo field). ``xe`` (the free-electron-fraction history feeding
    :func:`~beorn.precomputation.differentiable.rho_xray_diff`) is fixed at
    the ``fXh='constant'`` default (2e-4) — see
    :meth:`~beorn.precomputation.solver.RadiationProfileSolver.solve`.

    Args:
        z_grid: Static, decreasing redshift grid (numpy), shape (n_z,).
        dk: LPT initial-conditions Fourier field.
        L, N: Box size (Mpc/h) and cells per side.
        Om, Ob, h0, ns, sigma_8: Cosmology (may carry gradients).
        Nion, f_st, Mp, g1, g2, Mt, g3, g4, halo_mass_min, f0_esc, Mp_esc, pl_esc:
            Star-formation/escape-fraction astro parameters — see
            :func:`~beorn.precomputation.differentiable.ngam_dot_ion_diff`.
        xray_normalisation, alS_xray, energy_min_sed_xray, energy_max_sed_xray,
        energy_cutoff_min_xray, energy_cutoff_max_xray, HI_frac:
            X-ray SED parameters — see
            :func:`~beorn.precomputation.differentiable.rho_xray_diff`.
        n_lyman_alpha_photons, lyman_alpha_power_law: Lyman-α SED
            parameters — see
            :func:`~beorn.precomputation.differentiable.rho_alpha_profile_diff`.
        z_source_start, z_decoupling: Source lifetime / decoupling redshift
            — see :func:`~beorn.precomputation.differentiable.rho_xray_diff`/
            :func:`paint_snapshot_population_diff`.
        alpha_center: Representative accretion-rate exponent — see Scope
            above.
        M_env, cell_volume, M_min, n_mass_bins, eps_halo: Forwarded to
            :func:`beorn.lpt.chmf.halo_field_diff` at every redshift.
        r_grid_cell, r_lyal: Radial quadrature grids — ``None`` → the
            solver's own defaults (``np.logspace(-2, log10(600), 200)``
            comoving Mpc/h; ``np.logspace(-5, 2, 1000)`` physical Mpc/h).
        xHII_floor, spread_iter: Forwarded to
            :func:`paint_snapshot_population_diff` at every redshift.
        backend: 'numpy' (default, not differentiable), 'jax' or 'torch'.

    Returns:
        (dTb_mean, xHII_mean, Tk_mean) — each a backend array of shape
        (n_z,), in ``z_grid``'s order.
    """
    name, xp = get_backend(backend)

    if r_grid_cell is None:
        r_grid_cell = np.logspace(-2, np.log10(600.0), 200)
    if r_lyal is None:
        r_lyal = np.logspace(-5, 2, 1000, base=10)

    z_np = np.asarray(z_grid, dtype=float)
    xe = np.full(z_np.size, 2e-4)

    M_centers, _ = chmf_mass_bins(M_env, M_min, n_mass_bins=n_mass_bins)

    Ngam_bins = ngam_dot_ion_diff(
        z_np, M_centers[:, None], alpha_center, Om, Ob, h0, Nion, f_st, Mp,
        g1, g2, Mt, g3, g4, halo_mass_min, f0_esc, Mp_esc, pl_esc,
        backend=backend,
    )  # (n_mass_bins, n_z)
    R_bubble_bins = bubble_radius_diff(z_np, Ngam_bins, Om, Ob, h0, backend=backend)

    rho_xray_bins = rho_xray_diff(
        z_np, r_grid_cell, M_centers[:, None], alpha_center, Om, Ob, h0,
        f_st, Mp, g1, g2, Mt, g3, g4, halo_mass_min, xray_normalisation,
        alS_xray, energy_min_sed_xray, energy_max_sed_xray,
        energy_cutoff_min_xray, energy_cutoff_max_xray, HI_frac, xe,
        z_source_start, backend=backend,
    )  # (n_mass_bins, n_r, n_z)
    rho_heat_bins = rho_heat_diff(z_np, rho_xray_bins, Om, h0, z_decoupling,
                                  backend=backend)  # (n_mass_bins, n_r, n_z)

    rho_alpha_bins = rho_alpha_profile_diff(
        z_np, r_lyal, M_centers[:, None], alpha_center, Om, Ob, h0, f_st,
        Mp, g1, g2, Mt, g3, g4, halo_mass_min, n_lyman_alpha_photons,
        lyman_alpha_power_law, z_source_start, backend=backend,
    )  # (n_mass_bins, n_r_lyal, n_z)

    dTb_hist, xHII_hist, Tk_hist = [], [], []
    for i, z in enumerate(z_np):
        # rho_alpha_profile_diff (and RadiationProfileSolver.r_lyal, whose
        # convention it mirrors) is built on a *physical* radial grid; the
        # painting kernel (like r_temp/prof_temp_bins) needs comoving nodes
        # -- PaintingCoordinator.paint_single_mass_bin converts with this
        # same r_lyal * (1 + z) before building its Lyman-alpha kernel.
        # Missing this left the Lyman-alpha channel starved everywhere except
        # right next to each halo (~14x too small a comoving extent at z=13).
        r_alpha = r_lyal * (1.0 + z)
        _, dTb, xhii, Tk = paint_snapshot_population_diff(
            float(z), dk, L, N, Om, Ob, h0, ns, sigma_8,
            R_bubble_bins[:, i], rho_heat_bins[:, :, i], r_grid_cell,
            rho_alpha_bins[:, :, i], r_alpha, z_decoupling,
            M_env=M_env, cell_volume=cell_volume, M_min=M_min,
            n_mass_bins=n_mass_bins, eps_halo=eps_halo,
            xHII_floor=xHII_floor, spread_iter=spread_iter, backend=backend,
        )
        dTb_hist.append(xp.mean(dTb))
        xHII_hist.append(xp.mean(xhii))
        Tk_hist.append(xp.mean(Tk))

    return xp.stack(dTb_hist), xp.stack(xHII_hist), xp.stack(Tk_hist)
