"""Phase 2 (issue #42) differentiable 21-cm tests: G7/G8/G9/G12/G13/G14 + exit test.

The exit test differentiates Δ²₂₁(k) — the power spectrum of the dTb field
from a full differentiable chain (ICs → LPT density → EPS halo field →
Ngam_dot(z) from Nion/f_star/f_esc → bubble/profile painting → couplings →
dTb) — w.r.t. one real astro parameter (``Nion``, via G14's
:func:`~beorn.precomputation.differentiable.ngam_dot_ion_diff`) and one
cosmology parameter (σ₈), in jax and torch, against central finite
differences.
"""
import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from beorn.structs import Parameters
from beorn.couplings import x_coll, x_coll_diff, S_alpha, s_alpha_diff, dtb_diff
from beorn.painting.differentiable import (
    bubble_kernel_fourier, profile_kernel_fourier, spreading_excess_diff,
    paint_fields_diff,
)
from beorn.precomputation.differentiable import (
    linear_ode_solution, heat_ode_solution, bubble_radius_diff,
    sample_fst_reparam, interp_profiles_fst,
    mass_accretion_diff, mass_accretion_derivative_diff, ngam_dot_ion_diff,
    ngam_dot_ion_population_diff,
)
from beorn.astro_differentiable import f_star_halo_diff, f_esc_diff
from beorn.differentiable_pipeline import paint_snapshot_diff, dtb_global_signal_diff
from beorn.lpt import lpt_ics

jax = pytest.importorskip('jax', reason='differentiable 21-cm tests need jax')
import jax.numpy as jnp  # noqa: E402

jax.config.update('jax_enable_x64', True)

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    import inspect
    import tools21cm as t2c
    _T2C_BACKEND = 'backend' in str(
        inspect.signature(t2c.power_spectrum.apply_window))
except Exception:
    _T2C_BACKEND = False


# ─────────────────────────────────────────────────────────────────────────────
# G12 — closed-form ODE cores vs solve_ivp
# ─────────────────────────────────────────────────────────────────────────────

def test_linear_ode_matches_solve_ivp():
    a = np.linspace(0.05, 0.15, 400)
    A = np.sin(a * 40) ** 2 * 3.0 + 1.0
    B = 5.0 + 20 * a
    ref = solve_ivp(
        lambda t, y: np.interp(t, a, A) - np.interp(t, a, B) * y,
        [a[0], a[-1]], [0.0], t_eval=a, rtol=1e-10, atol=1e-14).y[0]
    V = linear_ode_solution(a, A, B)
    np.testing.assert_allclose(V[1:], ref[1:], rtol=1e-4)


def test_heat_ode_matches_solve_ivp():
    a = np.linspace(0.05, 0.15, 400)
    g = np.cos(a * 30) ** 2 * 2.0
    ref = solve_ivp(
        lambda t, y: np.interp(t, a, g) - 2 * y / t,
        [a[0], a[-1]], [0.0], t_eval=a, rtol=1e-10, atol=1e-14).y[0]
    y = heat_ode_solution(a, g)
    np.testing.assert_allclose(y[1:], ref[1:], rtol=2e-3)


def test_bubble_radius_diff_gradient():
    """R(z) grows with the photon amplitude, and jax grads match FD."""
    z_nodes = np.linspace(20.0, 6.0, 40)
    base = np.full(z_nodes.size, 1e52)

    def R_last(amp):
        R = bubble_radius_diff(z_nodes, amp * jnp.asarray(base),
                               0.31, 0.045, 0.68, backend='jax')
        return R[-1]

    g = jax.grad(R_last)(1.0)
    h = 1e-4
    fd = (R_last(1.0 + h) - R_last(1.0 - h)) / (2 * h)
    assert float(g) == pytest.approx(float(fd), rel=1e-6)
    assert float(R_last(1.0)) > 0


# ─────────────────────────────────────────────────────────────────────────────
# G7 — analytic kernels
# ─────────────────────────────────────────────────────────────────────────────

def test_bubble_kernel_volume_normalisation():
    """DC mode: the raw convolution deposits exactly n·V_bubble of volume."""
    N, L, R = 32, 100.0, 4.0
    V_cell = (L / N) ** 3
    h = np.zeros((N, N, N))
    h[4, 4, 4] = 1.0
    h[20, 22, 9] = 1.0
    k = np.sqrt(sum(np.meshgrid(
        np.fft.fftfreq(N, d=1 / N) ** 2, np.fft.fftfreq(N, d=1 / N) ** 2,
        np.fft.rfftfreq(N, d=1 / N) ** 2, indexing='ij'))) * 2 * np.pi / L
    field = np.fft.irfftn(np.fft.rfftn(h) * bubble_kernel_fourier(k, R),
                          s=(N, N, N)) / V_cell
    ionized = field.sum() * V_cell           # unclamped — DC mode is exact
    assert ionized == pytest.approx(2 * 4 * np.pi / 3 * R ** 3, rel=1e-10)


def test_paint_fields_diff_bubble_bounds():
    """The painted (clamped) xHII field is bounded and close to n·V_bubble.

    Values in [floor, 1]; clamping the Gibbs ringing of the sharp bubble edge
    inflates the volume at marginal R/cell resolution — R ≈ 4 cells here keeps
    it within a few per cent.
    """
    N, L, R = 32, 50.0, 6.0
    h = np.zeros((N, N, N))
    h[8, 8, 8] = 1.0
    xhii, _, _ = paint_fields_diff(h, 8.0, L, R_bubble=R, xHII_floor=1e-4)
    assert float(xhii.max()) <= 1.0 + 1e-12
    assert float(xhii.min()) >= 1e-4 - 1e-12
    ionized = float(xhii.sum()) * (L / N) ** 3
    expect = 4 * np.pi / 3 * R ** 3 + 1e-4 * L ** 3   # bubble + floor
    assert ionized == pytest.approx(expect, rel=0.05)


def test_profile_kernel_matches_bubble_closed_form():
    """The generic radial transform reproduces the top-hat closed form."""
    k = np.logspace(-2, 0.5, 50)
    R = 3.0
    r = np.linspace(1e-4, 8.0, 4000)
    prof = (r < R).astype(float)
    K_num = profile_kernel_fourier(k, r, prof)
    K_ana = bubble_kernel_fourier(k, R)
    np.testing.assert_allclose(K_num, K_ana, rtol=5e-3, atol=1e-3)


def test_bubble_kernel_gradient_in_R():
    def vol(R):
        return bubble_kernel_fourier(jnp.asarray([1e-6]), R, backend='jax')[0]
    g = jax.grad(vol)(3.0)
    assert float(g) == pytest.approx(4 * np.pi * 3.0 ** 2, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# G8 — spreading surrogate vs exact algorithm
# ─────────────────────────────────────────────────────────────────────────────

def test_spreading_surrogate_conserves_and_bounds():
    p = Parameters()
    p.simulation.Ncell = 32
    p.simulation.Lbox = 50.0
    x = np.zeros((32, 32, 32))
    zz, yy, xx = np.ogrid[:32, :32, :32]
    # two overlapping bubble pairs + one isolated bubble
    for c in [(10, 10, 10), (10, 10, 16), (22, 22, 22), (22, 25, 22),
              (5, 26, 6)]:
        x[(xx - c[0]) ** 2 + (yy - c[1]) ** 2 + (zz - c[2]) ** 2 < 25] += 1.0
    assert x.max() > 1.0  # bubbles overlap

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        exact = spreading_excess_fast_ref(p, x.copy())
    sur = spreading_excess_diff(x, 50.0, n_iter=12)

    assert float(np.max(sur)) <= 1.0 + 1e-12
    assert float(np.min(sur)) >= 0.0
    # photon conservation and the exact algorithm's global mean
    assert float(np.sum(sur)) == pytest.approx(x.sum(), rel=1e-3)
    assert float(np.mean(sur)) == pytest.approx(exact.mean(), rel=1e-3)


def spreading_excess_fast_ref(p, x):
    from beorn.painting.spread import spreading_excess_fast
    return spreading_excess_fast(p, x)


# ─────────────────────────────────────────────────────────────────────────────
# G9 — couplings
# ─────────────────────────────────────────────────────────────────────────────

def test_x_coll_diff_matches_splev():
    Tk = np.logspace(0, 3.5, 500)
    ref = x_coll(8.0, Tk, 0.9, 1e-3)
    new = x_coll_diff(8.0, Tk, 0.9, 1e-3)
    np.testing.assert_allclose(new, ref, rtol=1e-2)  # log-log linear vs spline


def test_s_alpha_and_dtb_diff_exact():
    Tk = np.logspace(0, 3, 100)
    np.testing.assert_allclose(s_alpha_diff(8.0, Tk, 0.5),
                               S_alpha(8.0, Tk, 0.5), rtol=1e-14)
    rng = np.random.default_rng(0)
    d, xh = rng.random(100), rng.random(100) * 0.5
    xt = rng.random(100)
    from beorn.cosmo import T_cmb
    ref = 27.0 * np.sqrt(9.0) * (1 - T_cmb(8.0) / Tk) * (1 - xh) \
        * xt / (1 + xt) * (1 + d)
    np.testing.assert_allclose(
        dtb_diff(8.0, Tk, xt, d, xh, factor=27.0), ref, rtol=1e-14)


def test_x_coll_diff_gradient():
    def f(logT):
        return x_coll_diff(8.0, jnp.exp(logT), 0.9, 1e-3,
                           backend='jax').sum()
    g = jax.grad(f)(jnp.log(jnp.asarray([10.0, 100.0]))).sum()
    assert np.isfinite(float(g)) and float(g) != 0.0


# ─────────────────────────────────────────────────────────────────────────────
# G13 — stochastic f_st
# ─────────────────────────────────────────────────────────────────────────────

def test_fst_reparam_mean_preserving_and_differentiable():
    eps = np.random.default_rng(1).standard_normal(100000)
    f = sample_fst_reparam(0.1, 0.5, eps, f_st_max=None)
    assert float(np.mean(f)) == pytest.approx(0.1, rel=5e-3)

    def mean_f(fc):
        return sample_fst_reparam(fc, 0.5, eps[:1000], f_st_max=None,
                                  backend='jax').mean()
    g = jax.grad(mean_f)(0.1)
    fd = (mean_f(0.1 + 1e-5) - mean_f(0.1 - 1e-5)) / 2e-5
    assert float(g) == pytest.approx(float(fd), rel=1e-6)


def test_interp_profiles_fst_matches_numpy():
    grid = np.linspace(0.01, 1.0, 11)
    stack = np.stack([np.full(5, g ** 2) for g in grid])
    q = np.array([0.055, 0.5, 2.0])  # incl. out-of-range clamp
    out = interp_profiles_fst(q, grid, stack)
    ref = np.interp(np.clip(q, grid[0], grid[-1]), grid, grid ** 2)
    np.testing.assert_allclose(out[:, 0], ref, rtol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# G14 — real Ngam_dot(z) source terms (astro-parameter gradients)
#
# f_star_halo_diff/f_esc_diff/mass_accretion_diff/mass_accretion_derivative_diff/
# ngam_dot_ion_diff are numpy/jax/torch ports of astro.f_star_Halo/astro.f_esc/
# massaccretion.mass_accretion/massaccretion.mass_accretion_derivative/
# helpers.Ngdot_ion's 'SED' branch — checked against those production
# functions directly (not just "runs"), then gradient-checked in jax and
# torch against central finite differences.
# ─────────────────────────────────────────────────────────────────────────────

_ASTRO_PARAM_DEFAULTS = dict(
    Om=0.315, Ob=0.049, h0=0.673, Nion=5000.0,
    f_st=0.05, Mp=2.8e11 * 0.68, g1=0.49, g2=-0.61, Mt=1e8, g3=4.0, g4=-1.0,
    halo_mass_min=1e8, f0_esc=0.2, Mp_esc=1e10, pl_esc=0.0,
)


def _astro_params():
    from beorn.structs import Parameters
    p = Parameters()
    for k, v in _ASTRO_PARAM_DEFAULTS.items():
        if k in ('Om', 'Ob', 'h0'):
            setattr(p.cosmology, k, v)
        else:
            setattr(p.source, k, v)
    return p


def test_f_star_halo_diff_matches_production():
    from beorn.astro import f_star_Halo
    p = _astro_params()
    Mh = np.logspace(7, 13, 200)
    ref = f_star_Halo(p, Mh.copy())
    d = _ASTRO_PARAM_DEFAULTS
    out = f_star_halo_diff(Mh, d['f_st'], d['Mp'], d['g1'], d['g2'], d['Mt'],
                           d['g3'], d['g4'], d['halo_mass_min'])
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-12)


def test_f_esc_diff_matches_production():
    from beorn.astro import f_esc
    p = _astro_params()
    Mh = np.logspace(7, 13, 200)
    ref = f_esc(p, Mh.copy())
    d = _ASTRO_PARAM_DEFAULTS
    out = f_esc_diff(Mh, d['f0_esc'], d['Mp_esc'], d['pl_esc'])
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-12)


def test_mass_accretion_diff_matches_production():
    from beorn.precomputation.massaccretion import mass_accretion
    p = _astro_params()
    z_bins = np.linspace(20.0, 6.0, 30)
    m_bins = np.array([1e10])
    alpha_bins = np.array([0.79])
    Mh_ref, dMh_dt_ref = mass_accretion(p, z_bins, m_bins, alpha_bins)

    Mh = mass_accretion_diff(z_bins, 1e10, 0.79)
    np.testing.assert_allclose(np.asarray(Mh), Mh_ref[0, 0], rtol=1e-12)

    dMh_dt = mass_accretion_derivative_diff(Mh, 0.79, p.cosmology.Om,
                                            p.cosmology.h0, z_bins)
    np.testing.assert_allclose(np.asarray(dMh_dt), dMh_dt_ref[0, 0], rtol=1e-10)


def test_ngam_dot_ion_diff_matches_production():
    from beorn.precomputation.helpers import Ngdot_ion
    from beorn.precomputation.massaccretion import mass_accretion
    p = _astro_params()
    z_bins = np.linspace(20.0, 6.0, 30)
    m_bins = np.array([1e10])
    alpha_bins = np.array([0.79])
    Mh_ref, dMh_dt_ref = mass_accretion(p, z_bins, m_bins, alpha_bins)
    ref = Ngdot_ion(p, z_bins, Mh_ref, dMh_dt_ref)[0, 0]

    d = _ASTRO_PARAM_DEFAULTS
    out = ngam_dot_ion_diff(z_bins, 1e10, 0.79, d['Om'], d['Ob'], d['h0'],
                            d['Nion'], d['f_st'], d['Mp'], d['g1'], d['g2'],
                            d['Mt'], d['g3'], d['g4'], d['halo_mass_min'],
                            d['f0_esc'], d['Mp_esc'], d['pl_esc'])
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-10)


@pytest.mark.parametrize('pname', ['Nion', 'f_st', 'f0_esc', 'g1'])
def test_ngam_dot_ion_diff_gradient_jax_and_torch(pname):
    """dNgam_dot(z=6)/dθ, jax vs finite differences, torch vs jax."""
    z_bins = np.linspace(20.0, 6.0, 30)
    d = dict(_ASTRO_PARAM_DEFAULTS)
    x0 = d.pop(pname)

    def S(val, backend):
        kw = dict(d)
        kw[pname] = val
        out = ngam_dot_ion_diff(z_bins, 1e10, 0.79, kw['Om'], kw['Ob'],
                                kw['h0'], kw['Nion'], kw['f_st'], kw['Mp'],
                                kw['g1'], kw['g2'], kw['Mt'], kw['g3'],
                                kw['g4'], kw['halo_mass_min'], kw['f0_esc'],
                                kw['Mp_esc'], kw['pl_esc'], backend=backend)
        return out[-1]

    g_jax = jax.grad(lambda v: S(v, 'jax'))(x0)
    h = 1e-4 * x0
    fd = (S(x0 + h, 'numpy') - S(x0 - h, 'numpy')) / (2 * h)
    assert float(g_jax) == pytest.approx(float(fd), rel=5e-3)

    if not _TORCH:
        pytest.skip('torch not installed')
    xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    S(xt, 'torch').backward()
    assert float(xt.grad) == pytest.approx(float(g_jax), rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# X-ray heating (issue #59, Phase A) — rho_xray_diff / rho_heat_diff
# ─────────────────────────────────────────────────────────────────────────────

from beorn.precomputation.differentiable import rho_xray_diff, rho_heat_diff  # noqa: E402

_XRAY_PARAM_DEFAULTS = dict(
    _ASTRO_PARAM_DEFAULTS,
    xray_normalisation=3.4e40, alS_xray=1.2,
    energy_min_sed_xray=500.0, energy_max_sed_xray=2000.0,
    energy_cutoff_min_xray=500.0, energy_cutoff_max_xray=2000.0,
    HI_frac=1 - 0.08, z_source_start=35.0, z_decoupling=135.0,
)


def _xray_params():
    p = _astro_params()
    d = _XRAY_PARAM_DEFAULTS
    p.source.xray_normalisation = d['xray_normalisation']
    p.source.alS_xray = d['alS_xray']
    p.source.energy_min_sed_xray = d['energy_min_sed_xray']
    p.source.energy_max_sed_xray = d['energy_max_sed_xray']
    p.source.energy_cutoff_min_xray = d['energy_cutoff_min_xray']
    p.source.energy_cutoff_max_xray = d['energy_cutoff_max_xray']
    p.solver.HI_frac = d['HI_frac']
    p.solver.z_source_start = d['z_source_start']
    p.solver.z_decoupling = d['z_decoupling']
    # A single (mass, alpha) bin, centered exactly on (Mh_center, alpha_center)
    # below, so RadiationProfileSolver.rho_xray/rho_heat compute the same
    # single-bin physics rho_xray_diff/rho_heat_diff do.
    p.solver.halo_mass_nbin = 2
    p.solver.halo_mass_bin_min = 1e9
    p.solver.halo_mass_bin_max = 1e11
    p.solver.halo_mass_accretion_alpha = np.array([0.5, 0.9])
    return p


def _reference_rho_xray_rho_heat(p, z_bins, Mh_center, alpha_center):
    """Build RadiationProfileSolver's rho_xray/rho_heat for one hand-picked bin."""
    from beorn.precomputation.solver import RadiationProfileSolver
    from beorn.precomputation.massaccretion import mass_accretion

    solver = RadiationProfileSolver(p, z_bins)
    Mh, dMh_dt = mass_accretion(p, z_bins, np.array([Mh_center]), np.array([alpha_center]))
    solver.halo_mass_evolution = Mh
    solver.halo_mass_derivative = dMh_dt
    xe = np.full(z_bins.size, 2e-4)
    rho_xray_ref = solver.rho_xray(solver.r_grid, xe)[:, 0, 0, :]
    rho_heat_ref = solver.rho_heat(rho_xray_ref[:, None, None, :])[:, 0, 0, :]
    return solver.r_grid, xe, rho_xray_ref, rho_heat_ref


def test_rho_xray_diff_matches_production():
    z_bins = np.linspace(20.0, 6.0, 12)
    Mh_center, alpha_center = 1e10, 0.79
    p = _xray_params()
    rr, xe, rho_xray_ref, _ = _reference_rho_xray_rho_heat(p, z_bins, Mh_center, alpha_center)

    d = _XRAY_PARAM_DEFAULTS
    out = rho_xray_diff(
        z_bins, rr, Mh_center, alpha_center, d['Om'], d['Ob'], d['h0'],
        d['f_st'], d['Mp'], d['g1'], d['g2'], d['Mt'], d['g3'], d['g4'],
        d['halo_mass_min'], d['xray_normalisation'], d['alS_xray'],
        d['energy_min_sed_xray'], d['energy_max_sed_xray'],
        d['energy_cutoff_min_xray'], d['energy_cutoff_max_xray'],
        d['HI_frac'], xe, d['z_source_start'],
    )
    assert out.shape == rho_xray_ref.shape
    nonzero = rho_xray_ref != 0
    assert nonzero.any()
    # atol floors comparisons deep in the profile's radial tail, where the
    # fixed-node lookback quadrature (n_zprime) vs production's adaptive
    # dz_prime=0.1 grid disagree most but the absolute values are physically
    # negligible (~1e-19 of the profile's peak).
    atol = 1e-9 * np.abs(rho_xray_ref[nonzero]).max()
    np.testing.assert_allclose(np.asarray(out)[nonzero], rho_xray_ref[nonzero],
                               rtol=0.15, atol=atol)


def test_rho_heat_diff_matches_production():
    z_bins = np.linspace(20.0, 6.0, 12)
    Mh_center, alpha_center = 1e10, 0.79
    p = _xray_params()
    rr, xe, rho_xray_ref, rho_heat_ref = _reference_rho_xray_rho_heat(
        p, z_bins, Mh_center, alpha_center)

    d = _XRAY_PARAM_DEFAULTS
    heat = rho_heat_diff(z_bins, rho_xray_ref, d['Om'], d['h0'], d['z_decoupling'])
    assert heat.shape == rho_heat_ref.shape
    nonzero = rho_heat_ref != 0
    assert nonzero.any()
    # atol floors comparisons near z_decoupling, where solve_ivp's reference
    # value is dominated by integration noise around zero (~1e-15) rather
    # than a real, physically meaningful heating rate.
    atol = 1e-3 * np.abs(rho_heat_ref[nonzero]).max()
    np.testing.assert_allclose(np.asarray(heat)[nonzero], rho_heat_ref[nonzero],
                               rtol=0.1, atol=atol)


@pytest.mark.parametrize('pname', ['xray_normalisation', 'alS_xray', 'f_st'])
def test_rho_xray_diff_gradient_jax_and_torch(pname):
    """d(sum of rho_xray at z=6)/dθ, jax vs finite differences, torch vs jax."""
    z_bins = np.linspace(20.0, 6.0, 12)
    rr = np.logspace(-2, np.log10(600), 40)
    xe = np.full(z_bins.size, 2e-4)
    d = dict(_XRAY_PARAM_DEFAULTS)
    x0 = d.pop(pname)

    def S(val, backend):
        kw = dict(d)
        kw[pname] = val
        out = rho_xray_diff(
            z_bins, rr, 1e10, 0.79, kw['Om'], kw['Ob'], kw['h0'],
            kw['f_st'], kw['Mp'], kw['g1'], kw['g2'], kw['Mt'], kw['g3'],
            kw['g4'], kw['halo_mass_min'], kw['xray_normalisation'],
            kw['alS_xray'], kw['energy_min_sed_xray'], kw['energy_max_sed_xray'],
            kw['energy_cutoff_min_xray'], kw['energy_cutoff_max_xray'],
            kw['HI_frac'], xe, kw['z_source_start'], backend=backend,
        )
        return out[..., -1].sum()

    g_jax = jax.grad(lambda v: S(v, 'jax'))(x0)
    h = 1e-4 * x0
    fd = (S(x0 + h, 'numpy') - S(x0 - h, 'numpy')) / (2 * h)
    assert np.isfinite(float(g_jax)) and float(g_jax) != 0.0
    assert float(g_jax) == pytest.approx(float(fd), rel=5e-2)

    if not _TORCH:
        pytest.skip('torch not installed')
    xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    S(xt, 'torch').backward()
    assert float(xt.grad) == pytest.approx(float(g_jax), rel=1e-6)


def test_rho_heat_diff_gradient_jax():
    """d(rho_heat at z=6)/d(xray_normalisation), jax vs finite differences."""
    z_bins = np.linspace(20.0, 6.0, 12)
    rr = np.logspace(-2, np.log10(600), 40)
    xe = np.full(z_bins.size, 2e-4)
    d = dict(_XRAY_PARAM_DEFAULTS)

    def S(xray_normalisation, backend):
        rho_x = rho_xray_diff(
            z_bins, rr, 1e10, 0.79, d['Om'], d['Ob'], d['h0'],
            d['f_st'], d['Mp'], d['g1'], d['g2'], d['Mt'], d['g3'], d['g4'],
            d['halo_mass_min'], xray_normalisation, d['alS_xray'],
            d['energy_min_sed_xray'], d['energy_max_sed_xray'],
            d['energy_cutoff_min_xray'], d['energy_cutoff_max_xray'],
            d['HI_frac'], xe, d['z_source_start'], backend=backend,
        )
        rho_h = rho_heat_diff(z_bins, rho_x, d['Om'], d['h0'], d['z_decoupling'],
                              backend=backend)
        return rho_h[..., -1].sum()

    x0 = d['xray_normalisation']
    g_jax = jax.grad(lambda v: S(v, 'jax'))(x0)
    h = 1e-4 * x0
    fd = (S(x0 + h, 'numpy') - S(x0 - h, 'numpy')) / (2 * h)
    assert np.isfinite(float(g_jax)) and float(g_jax) != 0.0
    assert float(g_jax) == pytest.approx(float(fd), rel=5e-2)


# ─────────────────────────────────────────────────────────────────────────────
# Lyman-alpha coupling (issue #59, Phase B) — rho_alpha_profile_diff
# ─────────────────────────────────────────────────────────────────────────────

from beorn.precomputation.differentiable import rho_alpha_profile_diff  # noqa: E402

_LYAL_PARAM_DEFAULTS = dict(
    _XRAY_PARAM_DEFAULTS,
    n_lyman_alpha_photons=9690.0, lyman_alpha_power_law=0.3,
)


def _lyal_params():
    p = _xray_params()
    p.source.n_lyman_alpha_photons = _LYAL_PARAM_DEFAULTS['n_lyman_alpha_photons']
    p.source.lyman_alpha_power_law = _LYAL_PARAM_DEFAULTS['lyman_alpha_power_law']
    return p


def _reference_rho_alpha(p, z_bins, r_grid, Mh_center, alpha_center):
    from beorn.precomputation.helpers import rho_alpha_profile
    from beorn.precomputation.massaccretion import mass_accretion

    Mh, dMh_dt = mass_accretion(p, z_bins, np.array([Mh_center]), np.array([alpha_center]))
    return rho_alpha_profile(p, z_bins, r_grid, Mh, dMh_dt)[:, 0, 0, :]


def test_rho_alpha_profile_diff_matches_production():
    z_bins = np.linspace(20.0, 6.0, 12)
    r_lyal = np.logspace(-5, 2, 60, base=10)
    Mh_center, alpha_center = 1e10, 0.79
    p = _lyal_params()
    rho_alpha_ref = _reference_rho_alpha(p, z_bins, r_lyal, Mh_center, alpha_center)

    d = _LYAL_PARAM_DEFAULTS
    out = rho_alpha_profile_diff(
        z_bins, r_lyal, Mh_center, alpha_center, d['Om'], d['Ob'], d['h0'],
        d['f_st'], d['Mp'], d['g1'], d['g2'], d['Mt'], d['g3'], d['g4'],
        d['halo_mass_min'], d['n_lyman_alpha_photons'], d['lyman_alpha_power_law'],
        d['z_source_start'],
    )
    assert out.shape == rho_alpha_ref.shape
    nonzero = rho_alpha_ref != 0
    assert nonzero.any()
    # atol floors the radial tail, where the fixed-node lookback quadrature
    # (n_zprime) vs production's adaptive dz grid disagree most but the
    # absolute values are physically negligible (see rho_xray_diff's test).
    atol = 1e-9 * np.abs(rho_alpha_ref[nonzero]).max()
    np.testing.assert_allclose(np.asarray(out)[nonzero], rho_alpha_ref[nonzero],
                               rtol=0.15, atol=atol)


@pytest.mark.parametrize('pname', ['n_lyman_alpha_photons', 'lyman_alpha_power_law', 'f_st'])
def test_rho_alpha_profile_diff_gradient_jax_and_torch(pname):
    """d(sum of rho_alpha at z=6)/dθ, jax vs finite differences, torch vs jax."""
    z_bins = np.linspace(20.0, 6.0, 12)
    r_lyal = np.logspace(-5, 2, 60, base=10)
    d = dict(_LYAL_PARAM_DEFAULTS)
    x0 = d.pop(pname)

    def S(val, backend):
        kw = dict(d)
        kw[pname] = val
        out = rho_alpha_profile_diff(
            z_bins, r_lyal, 1e10, 0.79, kw['Om'], kw['Ob'], kw['h0'],
            kw['f_st'], kw['Mp'], kw['g1'], kw['g2'], kw['Mt'], kw['g3'],
            kw['g4'], kw['halo_mass_min'], kw['n_lyman_alpha_photons'],
            kw['lyman_alpha_power_law'], kw['z_source_start'], backend=backend,
        )
        return out[..., -1].sum()

    g_jax = jax.grad(lambda v: S(v, 'jax'))(x0)
    h = 1e-4 * x0
    fd = (S(x0 + h, 'numpy') - S(x0 - h, 'numpy')) / (2 * h)
    assert np.isfinite(float(g_jax)) and float(g_jax) != 0.0
    assert float(g_jax) == pytest.approx(float(fd), rel=5e-2)

    if not _TORCH:
        pytest.skip('torch not installed')
    xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    S(xt, 'torch').backward()
    assert float(xt.grad) == pytest.approx(float(g_jax), rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Real ionizing budget (issue #59, Phase C) — ngam_dot_ion_population_diff
# ─────────────────────────────────────────────────────────────────────────────

def test_ngam_dot_ion_population_diff_is_bounded_by_per_halo_extremes():
    """A weighted mean over the mass grid must lie within the per-halo range
    at every z -- a reference-free structural check independent of exactly
    how the HMF weights are computed."""
    z_bins = np.linspace(20.0, 6.0, 12)
    mass_bins = np.logspace(8, 13, 60)
    d = _ASTRO_PARAM_DEFAULTS

    Ngam_pop = ngam_dot_ion_population_diff(
        z_bins, mass_bins, 0.79, d['Om'], d['Ob'], d['h0'], 0.97, 0.82,
        d['Nion'], d['f_st'], d['Mp'], d['g1'], d['g2'], d['Mt'], d['g3'],
        d['g4'], d['halo_mass_min'], d['f0_esc'], d['Mp_esc'], d['pl_esc'])
    Ngam_halo = ngam_dot_ion_diff(
        z_bins, mass_bins[:, None], 0.79, d['Om'], d['Ob'], d['h0'],
        d['Nion'], d['f_st'], d['Mp'], d['g1'], d['g2'], d['Mt'], d['g3'],
        d['g4'], d['halo_mass_min'], d['f0_esc'], d['Mp_esc'], d['pl_esc'])

    assert np.all(np.isfinite(Ngam_pop))
    assert np.all(Ngam_pop >= Ngam_halo.min(axis=0) - 1e-30)
    assert np.all(Ngam_pop <= Ngam_halo.max(axis=0) + 1e-30)


def test_ngam_dot_ion_population_diff_matches_manual_hmf_weighted_mean():
    """Cross-check the exact formula against an independently-written
    np.trapz weighted mean (bypassing trapz_static), not just structure."""
    from beorn.mass_function.differentiable import dndlnm

    z_bins = np.linspace(20.0, 6.0, 8)
    mass_bins = np.logspace(8, 13, 40)
    d = _ASTRO_PARAM_DEFAULTS

    out = ngam_dot_ion_population_diff(
        z_bins, mass_bins, 0.79, d['Om'], d['Ob'], d['h0'], 0.97, 0.82,
        d['Nion'], d['f_st'], d['Mp'], d['g1'], d['g2'], d['Mt'], d['g3'],
        d['g4'], d['halo_mass_min'], d['f0_esc'], d['Mp_esc'], d['pl_esc'])

    Ngam_halo = ngam_dot_ion_diff(
        z_bins, mass_bins[:, None], 0.79, d['Om'], d['Ob'], d['h0'],
        d['Nion'], d['f_st'], d['Mp'], d['g1'], d['g2'], d['Mt'], d['g3'],
        d['g4'], d['halo_mass_min'], d['f0_esc'], d['Mp_esc'], d['pl_esc'])
    lnM = np.log(mass_bins)
    ref = np.empty(z_bins.size)
    for i, z in enumerate(z_bins):
        weight = dndlnm(mass_bins, float(z), d['Om'], d['Ob'], d['h0'], 0.97, 0.82)
        ref[i] = np.trapezoid(weight * Ngam_halo[:, i], lnM) / np.trapezoid(weight, lnM)

    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-10)


@pytest.mark.parametrize('pname', ['Nion', 'f_st', 'sigma_8'])
def test_ngam_dot_ion_population_diff_gradient_jax_and_torch(pname):
    """d(Ngam_dot(z=6))/dθ, jax vs finite differences, torch vs jax --
    sigma_8 is a genuinely new gradient path (ngam_dot_ion_diff has no
    cosmology-power-spectrum dependence at all; this is HMF-only)."""
    z_bins = np.linspace(20.0, 6.0, 12)
    mass_bins = np.logspace(8, 13, 40)
    d = dict(_ASTRO_PARAM_DEFAULTS, ns=0.97, sigma_8=0.82)
    x0 = d.pop(pname)

    def S(val, backend):
        kw = dict(d)
        kw[pname] = val
        out = ngam_dot_ion_population_diff(
            z_bins, mass_bins, 0.79, kw['Om'], kw['Ob'], kw['h0'], kw['ns'],
            kw['sigma_8'], kw['Nion'], kw['f_st'], kw['Mp'], kw['g1'],
            kw['g2'], kw['Mt'], kw['g3'], kw['g4'], kw['halo_mass_min'],
            kw['f0_esc'], kw['Mp_esc'], kw['pl_esc'], backend=backend)
        return out[-1]

    g_jax = jax.grad(lambda v: S(v, 'jax'))(x0)
    h = 1e-4 * x0
    fd = (S(x0 + h, 'numpy') - S(x0 - h, 'numpy')) / (2 * h)
    assert np.isfinite(float(g_jax)) and float(g_jax) != 0.0
    assert float(g_jax) == pytest.approx(float(fd), rel=5e-3)

    if not _TORCH:
        pytest.skip('torch not installed')
    xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    S(xt, 'torch').backward()
    assert float(xt.grad) == pytest.approx(float(g_jax), rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Full per-bin differentiable twin (issue #59, Phase D)
#
# chmf_mass_bins/halo_field_diff(return_bins=True)/paint_fields_population_diff
# are the new building blocks; paint_snapshot_population_diff/
# dtb_global_signal_population_diff assemble them (plus the already-tested
# Phase A/B/C radiation-profile builders) into the true differentiable twin of
# paint_snapshot_diff/dtb_global_signal_diff — real per-bin X-ray heating and
# Lyman-alpha coupling instead of a toy profile/proxy.
# ─────────────────────────────────────────────────────────────────────────────

from beorn.lpt.chmf import chmf_mass_bins, halo_field_diff  # noqa: E402
from beorn.painting.differentiable import paint_fields_population_diff  # noqa: E402
from beorn.cosmo.differentiable import dTb_factor as dTb_factor_diff  # noqa: E402
from beorn.differentiable_pipeline import (  # noqa: E402
    paint_snapshot_population_diff, dtb_global_signal_population_diff,
)


def test_chmf_mass_bins_matches_halo_field_diff_grid():
    """halo_field_diff bins its expected halo counts onto exactly the grid
    chmf_mass_bins hands out standalone -- the extraction changed nothing."""
    M_env, M_min, n_mass_bins = 1e10, 1e9, 8
    M_centers, dln_M = chmf_mass_bins(M_env, M_min, n_mass_bins=n_mass_bins)
    assert M_centers.shape == (n_mass_bins,)
    assert dln_M.shape == (n_mass_bins,)

    rng = np.random.default_rng(0)
    delta_env = rng.standard_normal((6, 6, 6)) * 0.1
    _, M_centers_from_call = halo_field_diff(
        delta_env, M_env, 10.0, 0.315, 0.049, 0.673, 0.965, 0.811,
        cell_volume=1.0, M_min=M_min, n_mass_bins=n_mass_bins)
    np.testing.assert_allclose(M_centers_from_call, M_centers, rtol=1e-14)


def test_halo_field_diff_return_bins_reconstructs_combined_field():
    """The per-bin n_b_bins, weighted by w_bins and summed, must reconstruct
    the ordinary (return_bins=False) combined field exactly -- both paths
    run through the same loop in halo_field_diff, so this is a regression
    check on the return_bins plumbing, not the physics."""
    M_env, M_min, n_mass_bins = 1e10, 1e9, 6
    rng = np.random.default_rng(1)
    delta_env = rng.standard_normal((5, 5, 5)) * 0.1

    for weights in ('counts', 'mass'):
        field, M_centers = halo_field_diff(
            delta_env, M_env, 9.0, 0.315, 0.049, 0.673, 0.965, 0.811,
            cell_volume=2.0, M_min=M_min, n_mass_bins=n_mass_bins,
            weights=weights)
        field3, M_centers3, n_b_bins = halo_field_diff(
            delta_env, M_env, 9.0, 0.315, 0.049, 0.673, 0.965, 0.811,
            cell_volume=2.0, M_min=M_min, n_mass_bins=n_mass_bins,
            weights=weights, return_bins=True)

        np.testing.assert_allclose(field3, field, rtol=1e-14)
        np.testing.assert_allclose(M_centers3, M_centers, rtol=1e-14)
        assert n_b_bins.shape == (n_mass_bins,) + delta_env.shape

        w_bins = np.ones(n_mass_bins) if weights == 'counts' else M_centers
        reconstructed = np.tensordot(w_bins, np.asarray(n_b_bins), axes=1)
        np.testing.assert_allclose(reconstructed, np.asarray(field), rtol=1e-12)


def test_paint_fields_population_diff_n_bins_1_matches_paint_fields_diff():
    """A single bin must reduce exactly to paint_fields_diff's output -- same
    kernels, same accumulation, just one term in the Fourier sum."""
    N, L, z = 16, 50.0, 9.0
    rng = np.random.default_rng(2)
    halo_mesh = rng.random((N, N, N))
    R_bubble = 5.0
    r_alpha = np.logspace(-2, 1.5, 40)
    prof_alpha = np.exp(-r_alpha / 2.0)
    r_temp = np.linspace(1e-3, 20.0, 50)
    prof_temp = 50.0 * np.exp(-r_temp / 4.0)

    ref = paint_fields_diff(halo_mesh, z, L, R_bubble=R_bubble,
                            r_alpha=r_alpha, prof_alpha=prof_alpha,
                            r_temp=r_temp, prof_temp=prof_temp,
                            xHII_floor=1e-4, spread_iter=4)
    out = paint_fields_population_diff(
        halo_mesh[None, ...], z, L, R_bubble_bins=np.array([R_bubble]),
        r_alpha=r_alpha, prof_alpha_bins=prof_alpha[None, ...],
        r_temp=r_temp, prof_temp_bins=prof_temp[None, ...],
        xHII_floor=1e-4, spread_iter=4)

    for a, b in zip(out, ref):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12)


def test_dTb_factor_diff_matches_production():
    from beorn.cosmo import dTb_factor as dTb_factor_ref
    p = Parameters()
    p.cosmology.Om, p.cosmology.Ob, p.cosmology.h0 = 0.315, 0.049, 0.673
    ref = dTb_factor_ref(p)
    out = dTb_factor_diff(p.cosmology.Om, p.cosmology.Ob, p.cosmology.h0)
    assert float(out) == pytest.approx(ref, rel=1e-14)


_POP_PARAM_DEFAULTS = dict(_LYAL_PARAM_DEFAULTS, ns=0.97, sigma_8=0.82)

_N_POP, _L_POP = 8, 60.0


def _pop_history(z_grid, dk, backend, d=None, n_mass_bins=4):
    d = dict(_POP_PARAM_DEFAULTS) if d is None else d
    return dtb_global_signal_population_diff(
        z_grid, dk, _L_POP, _N_POP, d['Om'], d['Ob'], d['h0'], d['ns'],
        d['sigma_8'], d['Nion'], d['f_st'], d['Mp'], d['g1'], d['g2'],
        d['Mt'], d['g3'], d['g4'], d['halo_mass_min'], d['f0_esc'],
        d['Mp_esc'], d['pl_esc'], d['xray_normalisation'], d['alS_xray'],
        d['energy_min_sed_xray'], d['energy_max_sed_xray'],
        d['energy_cutoff_min_xray'], d['energy_cutoff_max_xray'],
        d['HI_frac'], d['n_lyman_alpha_photons'], d['lyman_alpha_power_law'],
        d['z_source_start'], d['z_decoupling'], n_mass_bins=n_mass_bins,
        backend=backend)


@pytest.fixture(scope='module')
def pop_noise():
    return np.random.default_rng(4).standard_normal((_N_POP,) * 3)


def test_dtb_global_signal_population_diff_matches_paint_snapshot_population_diff_per_snapshot(pop_noise):
    """The driver's own Ngam/R_bubble/rho_xray/rho_heat/rho_alpha solve-once
    -over-z-then-index plumbing must reproduce a direct
    paint_snapshot_population_diff call at every grid point."""
    dk = lpt_ics(pop_noise, _L_POP, 0.315, 0.049, 0.673, 0.965, 0.811,
                backend='numpy')
    d = _POP_PARAM_DEFAULTS
    n_mass_bins = 4
    z_grid = np.array([12.0, 9.0])

    dTb_hist, xHII_hist, Tk_hist = _pop_history(z_grid, dk, 'numpy',
                                                n_mass_bins=n_mass_bins)

    M_centers, _ = chmf_mass_bins(1e10, 1e9, n_mass_bins=n_mass_bins)
    xe = np.full(z_grid.size, 2e-4)
    Ngam_bins = ngam_dot_ion_diff(
        z_grid, M_centers[:, None], 0.79, d['Om'], d['Ob'], d['h0'],
        d['Nion'], d['f_st'], d['Mp'], d['g1'], d['g2'], d['Mt'], d['g3'],
        d['g4'], d['halo_mass_min'], d['f0_esc'], d['Mp_esc'], d['pl_esc'])
    R_bubble_bins = bubble_radius_diff(z_grid, Ngam_bins, d['Om'], d['Ob'], d['h0'])
    # must match dtb_global_signal_population_diff's own defaults exactly --
    # a different quadrature resolution gives a slightly different profile.
    r_grid_cell = np.logspace(-2, np.log10(600.0), 200)
    r_lyal = np.logspace(-5, 2, 1000, base=10)
    rho_xray_bins = rho_xray_diff(
        z_grid, r_grid_cell, M_centers[:, None], 0.79, d['Om'], d['Ob'], d['h0'],
        d['f_st'], d['Mp'], d['g1'], d['g2'], d['Mt'], d['g3'], d['g4'],
        d['halo_mass_min'], d['xray_normalisation'], d['alS_xray'],
        d['energy_min_sed_xray'], d['energy_max_sed_xray'],
        d['energy_cutoff_min_xray'], d['energy_cutoff_max_xray'], d['HI_frac'],
        xe, d['z_source_start'])
    rho_heat_bins = rho_heat_diff(z_grid, rho_xray_bins, d['Om'], d['h0'], d['z_decoupling'])
    rho_alpha_bins = rho_alpha_profile_diff(
        z_grid, r_lyal, M_centers[:, None], 0.79, d['Om'], d['Ob'], d['h0'],
        d['f_st'], d['Mp'], d['g1'], d['g2'], d['Mt'], d['g3'], d['g4'],
        d['halo_mass_min'], d['n_lyman_alpha_photons'], d['lyman_alpha_power_law'],
        d['z_source_start'])

    for i, z in enumerate(z_grid):
        _, dTb_direct, xhii_direct, Tk_direct = paint_snapshot_population_diff(
            float(z), dk, _L_POP, _N_POP, d['Om'], d['Ob'], d['h0'], d['ns'],
            d['sigma_8'], R_bubble_bins[:, i], rho_heat_bins[:, :, i],
            r_grid_cell, rho_alpha_bins[:, :, i], r_lyal, d['z_decoupling'],
            n_mass_bins=n_mass_bins)

        assert float(dTb_hist[i]) == pytest.approx(float(np.mean(dTb_direct)), rel=1e-10)
        assert float(xHII_hist[i]) == pytest.approx(float(np.mean(xhii_direct)), rel=1e-10)
        assert float(Tk_hist[i]) == pytest.approx(float(np.mean(Tk_direct)), rel=1e-10)


def test_dtb_global_signal_population_diff_finite_all_backends(pop_noise):
    """Regression guard for the two real bugs found while wiring this up:
    a float32 overflow (inf/inf -> NaN) in the collisional-coupling rho_b
    formula (fixed by assembling its constant chain from scalars before
    multiplying delta_b), and (separately) jax's float32 default overflowing
    xray_normalisation ~1e39-1e40 unless jax_enable_x64 is set (this test
    file sets it globally at import time)."""
    z_grid = np.array([12.0, 9.0])

    for backend in (['numpy', 'jax', 'torch'] if _TORCH else ['numpy', 'jax']):
        dk = lpt_ics(pop_noise, _L_POP, 0.315, 0.049, 0.673, 0.965, 0.811,
                    backend=backend)
        dTb_hist, xHII_hist, Tk_hist = _pop_history(z_grid, dk, backend)
        for hist in (dTb_hist, xHII_hist, Tk_hist):
            vals = np.asarray(hist.detach() if backend == 'torch' else hist)
            assert np.all(np.isfinite(vals)), backend
        xhii = np.asarray(xHII_hist.detach() if backend == 'torch' else xHII_hist)
        assert np.all(xhii >= 0.0) and np.all(xhii <= 1.0)


@pytest.mark.parametrize('pname', ['Nion', 'xray_normalisation', 'n_lyman_alpha_photons'])
def test_dtb_global_signal_population_diff_gradient_jax_and_torch(pop_noise, pname):
    """d(sum of the dTb history)/dtheta, jax vs finite differences, torch vs
    jax -- one representative parameter per channel (ionization, X-ray
    heating, Lyman-alpha coupling)."""
    z_grid = np.array([12.0, 9.0])
    d = dict(_POP_PARAM_DEFAULTS)
    x0 = d.pop(pname)

    def S(val, backend):
        kw = dict(d)
        kw[pname] = val
        dk = lpt_ics(pop_noise, _L_POP, kw['Om'], kw['Ob'], kw['h0'],
                    kw['ns'], kw['sigma_8'], backend=backend)
        dTb_hist, _, _ = _pop_history(z_grid, dk, backend, d=kw)
        return dTb_hist.sum()

    g_jax = jax.grad(lambda v: S(v, 'jax'))(x0)
    h = 1e-4 * x0
    fd = (S(x0 + h, 'numpy') - S(x0 - h, 'numpy')) / (2 * h)
    assert np.isfinite(float(g_jax))
    assert float(g_jax) == pytest.approx(float(fd), rel=5e-2)

    if not _TORCH:
        pytest.skip('torch not installed')
    xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    S(xt, 'torch').backward()
    assert float(xt.grad) == pytest.approx(float(g_jax), rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 exit test — dΔ²₂₁/dθ, one astro + one cosmology parameter
# ─────────────────────────────────────────────────────────────────────────────

N, L, Z = 16, 100.0, 9.0


def _delta2_21(s8, ngam_amp, backend, noise, eps_halo):
    """Differentiable toy 21-cm chain hitting every Phase 1+2 G-piece.

    The per-snapshot physics (density -> EPS halo field -> painting ->
    couplings -> dTb) is :func:`beorn.differentiable_pipeline.paint_snapshot_diff`
    itself (issue #42 G15 multi-z driver) -- not a separately maintained
    copy -- so this exit test and :func:`~beorn.differentiable_pipeline.dtb_global_signal_diff`
    are provably running identical single-z physics.
    """
    Om, Ob, h0, ns = 0.31, 0.045, 0.68, 0.97
    if backend == 'jax':
        xp = jnp
        noise_b = jnp.asarray(noise)
    else:
        xp = torch
        noise_b = torch.as_tensor(noise)

    # cosmology: ICs (painted density/EPS field are computed per snapshot below)
    dk = lpt_ics(noise_b, L, Om, Ob, h0, ns, s8, backend=backend)

    # astro: bubble radius from the photon-rate ODE (G12). Ngam_dot(z) is a
    # real function of Nion (=ngam_amp) through star-formation efficiency and
    # escape fraction (G14) now, not a bare amplitude — one representative
    # (mass, alpha) bin, param.yaml's defaults for the shape parameters not
    # under test.
    z_nodes = np.linspace(25.0, Z, 30)
    Ngam = ngam_dot_ion_diff(z_nodes, 1e10, 0.79, Om, Ob, h0, ngam_amp,
                             0.05, 2.8e11 * h0, 0.49, -0.61, 1e8, 4.0, -1.0,
                             1e8, 0.2, 1e10, 0.0, backend=backend)
    R_b = bubble_radius_diff(z_nodes, Ngam, Om, Ob, h0, backend=backend)[-1]

    # painting (G7 + G8 + G11) + couplings + dTb (G9) — heating stays a
    # fixed toy profile (the X-ray source term is out of scope for the
    # Ngam_dot(z) builder above, see G14); ngam_amp/Nion's gradient path
    # runs through the ionization channel only.
    _, dTb, _, _ = paint_snapshot_diff(Z, dk, R_b, L, N, Om, Ob, h0, ns, s8,
                                       eps_halo=eps_halo, backend=backend)

    ps, k = t2c.power_spectrum_1d(dTb, kbins=4, box_dims=L, backend=backend)
    if backend == 'jax':
        return (ps * jnp.asarray(k) ** 3).sum()
    return (ps * torch.as_tensor(np.asarray(k)) ** 3).sum()


@pytest.fixture(scope='module')
def chain_inputs():
    rng = np.random.default_rng(3)
    return (rng.standard_normal((N,) * 3),
            rng.standard_normal((8,) + (N,) * 3))


# ─────────────────────────────────────────────────────────────────────────────
# G15 — multi-redshift driver (dtb_global_signal_diff)
#
# paint_snapshot_diff/dtb_global_signal_diff loop the single-z chain over a
# redshift grid, solving Ngam_dot(z)/R_bubble(z) once over the whole grid
# (not re-solved per iteration) and reducing each snapshot to its spatial
# mean -- the differentiable counterpart of TemporalCube.global_mean.
# ─────────────────────────────────────────────────────────────────────────────

_ASTRO_DEFAULTS_G15 = dict(_ASTRO_PARAM_DEFAULTS)
_ASTRO_DEFAULTS_G15['ns'] = 0.97
_ASTRO_DEFAULTS_G15['sigma_8'] = 0.82


def _history(z_grid, dk, backend, d=None):
    d = dict(_ASTRO_DEFAULTS_G15) if d is None else d
    return dtb_global_signal_diff(
        z_grid, dk, L, N, d['Om'], d['Ob'], d['h0'], d['ns'], d['sigma_8'],
        d['Nion'], d['f_st'], d['Mp'], d['g1'], d['g2'], d['Mt'], d['g3'],
        d['g4'], d['halo_mass_min'], d['f0_esc'], d['Mp_esc'], d['pl_esc'],
        backend=backend)


def test_dtb_global_signal_diff_matches_paint_snapshot_diff_per_snapshot(chain_inputs):
    """dtb_global_signal_diff's plumbing (Ngam/R_bubble solve + loop + stack)
    must reproduce a direct paint_snapshot_diff call at every grid point --
    both run the identical per-snapshot physics, so this isolates the driver's
    own loop/indexing logic from the physics itself (already verified above
    and via the exit test below, which now calls paint_snapshot_diff too).

    A single-element z_grid can't be used here: bubble_radius_diff's
    integrating-factor ODE (cumtrapz_static -> np.diff) needs at least two
    nodes to define a step, so this uses the smallest grid that's actually
    valid input to that machinery.
    """
    noise, _ = chain_inputs
    dk = lpt_ics(jnp.asarray(noise), L, 0.31, 0.045, 0.68, 0.97, 0.82,
                backend='jax')
    d = _ASTRO_DEFAULTS_G15

    z_grid = np.array([12.0, 9.0])
    dTb_hist, xHII_hist, Tk_hist = _history(z_grid, dk, 'jax')

    mass_bins = np.logspace(np.log10(d['halo_mass_min']), 13.0, 50)
    Ngam = ngam_dot_ion_population_diff(
        z_grid, mass_bins, 0.79, d['Om'], d['Ob'], d['h0'], d['ns'],
        d['sigma_8'], d['Nion'], d['f_st'], d['Mp'], d['g1'], d['g2'],
        d['Mt'], d['g3'], d['g4'], d['halo_mass_min'], d['f0_esc'],
        d['Mp_esc'], d['pl_esc'], backend='jax')
    R_b = bubble_radius_diff(z_grid, Ngam, d['Om'], d['Ob'], d['h0'], backend='jax')

    for i, z in enumerate(z_grid):
        _, dTb_direct, xhii_direct, Tk_direct = paint_snapshot_diff(
            float(z), dk, R_b[i], L, N, d['Om'], d['Ob'], d['h0'], d['ns'],
            d['sigma_8'], eps_halo=None, backend='jax')

        assert float(dTb_hist[i]) == pytest.approx(float(jnp.mean(dTb_direct)), rel=1e-10)
        assert float(xHII_hist[i]) == pytest.approx(float(jnp.mean(xhii_direct)), rel=1e-10)
        assert float(Tk_hist[i]) == pytest.approx(float(jnp.mean(Tk_direct)), rel=1e-10)


def test_dtb_global_signal_diff_multi_z_finite_and_reionizes(chain_inputs):
    noise, _ = chain_inputs
    dk = lpt_ics(jnp.asarray(noise), L, 0.31, 0.045, 0.68, 0.97, 0.82,
                backend='jax')
    z_grid = np.linspace(15.0, 6.0, 5)   # decreasing z
    dTb_hist, xHII_hist, Tk_hist = _history(z_grid, dk, 'jax')

    for hist in (dTb_hist, xHII_hist, Tk_hist):
        assert np.all(np.isfinite(np.asarray(hist)))
    # reionization proceeds forward in time (decreasing z): allow small
    # wiggle from the toy chain/single realization, but not a net decrease.
    xhii = np.asarray(xHII_hist)
    assert xhii[-1] >= xhii[0]
    assert np.all(np.diff(xhii) > -0.05)


@pytest.mark.parametrize('pname', ['Nion', 'sigma_8'])
def test_dtb_global_signal_diff_gradient_jax_and_torch(chain_inputs, pname):
    """d(sum of the dTb history)/dθ, jax vs finite differences, torch vs jax."""
    noise, _ = chain_inputs
    z_grid = np.linspace(15.0, 6.0, 3)
    d = dict(_ASTRO_DEFAULTS_G15)
    x0 = d.pop(pname)

    def S(val, backend):
        kw = dict(d)
        kw[pname] = val
        if backend == 'jax':
            noise_b = jnp.asarray(noise)
        elif backend == 'torch':
            noise_b = torch.as_tensor(noise)
        else:
            noise_b = noise
        s8 = kw.pop('sigma_8')
        dk = lpt_ics(noise_b, L, kw['Om'], kw['Ob'], kw['h0'], kw['ns'], s8,
                    backend=backend)
        kw['sigma_8'] = s8
        dTb_hist, _, _ = _history(z_grid, dk, backend, d=kw)
        return dTb_hist.sum()

    g_jax = jax.grad(lambda v: S(v, 'jax'))(x0)
    h = 1e-4 * x0
    fd = (S(x0 + h, 'numpy') - S(x0 - h, 'numpy')) / (2 * h)
    assert float(g_jax) == pytest.approx(float(fd), rel=1e-2)

    if not _TORCH:
        pytest.skip('torch not installed')
    xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    S(xt, 'torch').backward()
    assert float(xt.grad) == pytest.approx(float(g_jax), rel=1e-4)


@pytest.mark.skipif(not _T2C_BACKEND,
                    reason='tools21cm without differentiable backend')
@pytest.mark.parametrize('pname,x0', [('sigma_8', 0.82), ('ngam_amp', 5000.0)])
def test_exit_grad_jax_21cm(chain_inputs, pname, x0):
    noise, eps_halo = chain_inputs

    def S(val):
        s8 = val if pname == 'sigma_8' else 0.82
        amp = val if pname == 'ngam_amp' else 5000.0
        return _delta2_21(s8, amp, 'jax', noise, eps_halo)

    g = jax.grad(S)(x0)
    # ngam_amp's chain is more nonlinear now (real f_star/f_esc physics, not a
    # bare amplitude — G14), so central differences need a smaller relative
    # step than sigma_8's to stay within the finite-difference truncation
    # tolerance below.
    h = (3e-4 if pname == 'sigma_8' else 1e-5) * x0
    fd = (S(x0 + h) - S(x0 - h)) / (2 * h)
    assert float(g) == pytest.approx(float(fd), rel=5e-3)


@pytest.mark.skipif(not (_T2C_BACKEND and _TORCH),
                    reason='needs torch + tools21cm differentiable backend')
def test_exit_grad_torch_matches_jax_21cm(chain_inputs):
    noise, eps_halo = chain_inputs
    g_jax = float(jax.grad(
        lambda a: _delta2_21(0.82, a, 'jax', noise, eps_halo))(5000.0))

    amp = torch.tensor(5000.0, dtype=torch.float64, requires_grad=True)
    out = _delta2_21(0.82, amp, 'torch', noise, eps_halo)
    out.backward()
    assert float(amp.grad) == pytest.approx(g_jax, rel=1e-4)
