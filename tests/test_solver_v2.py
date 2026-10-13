"""nonuniform-fix S2: the v2 (continuous-onset) profile solver.

Small grids so the suite stays fast. See fst_stochastic/docs/src_fix_plan_2026-10-01.md (D1-D4)."""
import copy
import sys
import types

if "MAS_library" not in sys.modules:
    sys.modules["MAS_library"] = types.SimpleNamespace(MASL=None)

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from beorn.constants import km_per_Mpc, kb_eV_per_K
from beorn.cosmo import hubble
from beorn.precomputation.helpers import emission_history_interpolator, emission_rate_fn
from beorn.precomputation.massaccretion import mass_accretion, trajectory_at
from beorn.astro import f_star_Halo
from beorn.precomputation.solver import RadiationProfileSolver
from beorn.structs.parameters import Parameters


def small_parameters(numerics="v2"):
    p = Parameters()
    p.solver.numerics = numerics
    p.solver.redshifts = np.array([20.0, 16.0, 13.0, 11.0, 9.5, 8.5])
    p.solver.halo_mass_bin_min = 1e8
    p.solver.halo_mass_bin_max = 1e12
    p.solver.halo_mass_nbin = 9
    p.solver.halo_mass_accretion_alpha = np.array([0.45, 0.55, 0.65])
    p.solver.z_source_start = 20.0
    p.solver.bubble_substeps = 50
    p.solver.onset_internal_refine = 2
    p.source.halo_mass_min = 1e9
    return p


def test_trajectory_at_matches_mass_accretion_on_the_profile_grid():
    p = small_parameters()
    zb = p.solver.redshifts
    m, a = p.solver.halo_mass_bin_centers, p.solver.halo_mass_accretion_alpha_bin_centers
    Mh, dM = mass_accretion(p, zb, m, a)
    Mt, dMt = trajectory_at(p, zb, m, a, float(zb.min()))
    np.testing.assert_allclose(Mt, Mh, rtol=1e-14)
    np.testing.assert_allclose(dMt, dM, rtol=1e-14)


def test_rate_fn_matches_tabulated_rate_at_grid_redshifts():
    p = small_parameters()
    zb = p.solver.redshifts
    m, a = p.solver.halo_mass_bin_centers, p.solver.halo_mass_accretion_alpha_bin_centers
    Mh, dM = mass_accretion(p, zb, m, a)
    tab = (p.cosmology.Ob / p.cosmology.Om) * f_star_Halo(p, Mh) * dM
    tab[np.where(Mh < p.source.halo_mass_min)] = 0
    ana = emission_rate_fn(p, m, a, float(zb.min()))(zb)
    np.testing.assert_allclose(ana, tab, rtol=1e-12)
    assert np.array_equal(ana > 0, tab > 0)


def test_rate_fn_switches_on_exactly_at_the_cutoff():
    p = small_parameters()
    zb = p.solver.redshifts
    m, a = np.array([2e9]), np.array([0.5])
    z_ref = float(zb.min())
    z_on = z_ref + np.log(2e9 / p.source.halo_mass_min) / 0.5     # M(z_on) = M_min
    rate = emission_rate_fn(p, m, a, z_ref)
    assert rate(np.array([z_on + 1e-6]))[0, 0, 0] == 0.0
    assert rate(np.array([z_on - 1e-6]))[0, 0, 0] > 0.0


def test_internal_grid_keeps_profile_redshifts():
    p = small_parameters()
    s = RadiationProfileSolver(p, p.solver.redshifts)
    for k in (1, 3):
        z_int, idx = s._internal_grid(k)
        assert z_int.size == (p.solver.redshifts.size - 1) * k + 1
        assert np.array_equal(z_int[idx], p.solver.redshifts)
        assert np.all(np.diff(z_int) < 0)
    z1, _ = s._internal_grid(1)
    assert np.array_equal(z1, p.solver.redshifts)


def test_heat_quadrature_matches_a_tight_ode_and_is_never_negative():
    p = small_parameters()
    zb = np.linspace(20, 8, 13)
    rng = np.random.default_rng(0)
    rho = np.abs(rng.normal(size=(3, 2, zb.size))) * 1e-12 * np.linspace(0, 1, zb.size)
    T = RadiationProfileSolver._heat_quadrature(p, zb, rho, nsub=256)
    aa = 1 / (1 + zb)
    rho_i = interp1d(aa, rho, axis=-1, fill_value="extrapolate")

    def rhs(a, y):
        g = 2 * rho_i(a) / (3 * kb_eV_per_K * a * hubble(1 / a - 1, p)) * km_per_Mpc
        return g.ravel() - 2 * y / a

    sol = solve_ivp(rhs, [aa[0], aa[-1]], np.zeros(rho[..., 0].size), t_eval=aa, rtol=1e-10, atol=1e-10 * T.max())
    assert sol.success, sol.message
    T_ode = sol.y.reshape(T.shape)
    np.testing.assert_allclose(T[..., 1:], T_ode[..., 1:], rtol=1e-5)
    assert T.min() >= 0
    assert np.all(RadiationProfileSolver._heat_quadrature(p, zb, np.zeros_like(rho)) == 0)


def test_rho_xray_on_reproduces_legacy_rho_xray():
    """The general X-ray routine with the tabulated rate, the profile grid and dz' = 0.1 is the
    legacy rho_xray bit for bit."""
    p = small_parameters(numerics="legacy")
    zb = p.solver.redshifts
    s = RadiationProfileSolver(p, zb)
    m, a = p.solver.halo_mass_bin_centers, p.solver.halo_mass_accretion_alpha_bin_centers
    s.halo_mass_evolution, s.halo_mass_derivative = mass_accretion(p, zb, m, a)
    xe = np.full(zb.size, 2e-4)
    legacy = s.rho_xray(s.r_grid, xe)
    Mstar = (p.cosmology.Ob / p.cosmology.Om) * f_star_Halo(p, s.halo_mass_evolution) * s.halo_mass_derivative
    Mstar[np.where(s.halo_mass_evolution < p.source.halo_mass_min)] = 0
    rate = emission_history_interpolator(zb, Mstar, p.solver.z_source_start)
    general = s._rho_xray_on(p, s.r_grid, zb, xe, rate, m.size, a.size, 0.1)
    assert np.array_equal(general, legacy)


def test_v2_solve_preserves_the_cutoff_mass():
    p = small_parameters()
    zb = p.solver.redshifts
    prof = RadiationProfileSolver(p, zb).solve()
    edges = np.asarray(prof.halo_mass_bins)
    centres = np.sqrt(edges[:-1] * edges[1:])
    below = centres <= p.source.halo_mass_min
    assert np.all(prof.R_bubble[below] == 0)
    assert np.all(prof.rho_heat.max(axis=0)[below] == 0)
    assert np.all(prof.rho_alpha.max(axis=0)[below] == 0)
    above = ~below
    above[..., 0] = False                        # z = 20 = z_source_start: nothing has emitted yet
    assert np.all(prof.R_bubble[above] > 0)
    assert prof.rho_heat.min() >= 0


def test_v2_and_legacy_agree_far_from_the_cutoff():
    """Far above M_min the onset quantisation is irrelevant: v2 and legacy agree to the
    accuracy of the legacy time interpolation (a few per cent on this coarse grid)."""
    zb = small_parameters().solver.redshifts
    v2 = RadiationProfileSolver(small_parameters("v2"), zb).solve()
    lg = RadiationProfileSolver(small_parameters("legacy"), zb).solve()
    ratio = v2.R_bubble[-1, :, -1] ** 3 / lg.R_bubble[-1, :, -1] ** 3
    assert np.all(np.abs(ratio - 1) < 0.1)


def test_fstar_branch_uses_per_f_st_parameters():
    p = small_parameters()
    zb = p.solver.redshifts
    s = RadiationProfileSolver(p, zb)
    r_lyal = np.logspace(-5, 2, 50)
    lo, hi = copy.deepcopy(p), copy.deepcopy(p)
    lo.source.f_st, hi.source.f_st = 0.01, 0.04
    Rlo = s._continuous_profiles(lo, r_lyal)[0]
    Rhi = s._continuous_profiles(hi, r_lyal)[0]
    emitting = Rlo > 0
    assert np.all(Rhi[emitting] > Rlo[emitting])


def test_direct_integrator_matches_loop_away_from_the_onset():
    """Step B (profile_integrator='direct', default) vs the first v2 scheme ('loop'): same
    physics, the rate evaluated exactly at z'(r) instead of interpolated between lookback points.
    Agreement to the dz' discretisation; zero/non-zero changes only at the switch-on."""
    zb = small_parameters().solver.redshifts
    loop_p = small_parameters()
    loop_p.solver.profile_integrator = "loop"
    d = RadiationProfileSolver(small_parameters(), zb).solve()
    l = RadiationProfileSolver(loop_p, zb).solve()
    assert np.array_equal(d.R_bubble, l.R_bubble)                     # bubble solve unchanged
    for name in ("rho_xray", "rho_heat", "rho_alpha"):
        a, b = getattr(d, name), getattr(l, name)
        assert np.abs(a - b).max() <= 1e-4 * np.abs(b).max(), name
        big = np.abs(b) > 1e-3 * np.abs(b).max()
        assert np.all(np.abs(a[big] / b[big] - 1) < 0.05), name
    assert d.rho_heat.min() >= 0
