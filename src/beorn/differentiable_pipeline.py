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

Real Ly-alpha (:func:`beorn.precomputation.helpers.rho_alpha_profile`) and
X-ray heating (:meth:`beorn.precomputation.solver.RadiationProfileSolver.rho_xray`)
source terms are **not** implemented here — both channels still use the same
toy proxies as the issue #42 exit test (a fixed radial heating profile, a
Ly-alpha coupling proportional to the halo mesh); differentiable ports of
both now exist (:func:`beorn.precomputation.differentiable.rho_xray_diff`/
:func:`~beorn.precomputation.differentiable.rho_alpha_profile_diff`, issue #59
Phases A/B) but aren't wired into this driver yet — that's Phase D. Only the
ionization channel is real astro-parameter physics end to end.
``backend='numpy'`` is the default and is **not** differentiable (plain
NumPy has no autodiff); use ``backend='jax'``/``'torch'`` for gradients.
"""
from __future__ import annotations

import numpy as np

from .cosmo.differentiable import get_backend
from .constants import rhoc0
from .lpt import lpt_density, lpt_linear_density
from .lpt.chmf import halo_field_diff
from .precomputation.differentiable import ngam_dot_ion_population_diff, bubble_radius_diff
from .painting.differentiable import paint_fields_diff
from .couplings import x_coll_diff, s_alpha_diff, dtb_diff

__all__ = ['paint_snapshot_diff', 'dtb_global_signal_diff']

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
