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
)
from beorn.astro_differentiable import f_star_halo_diff, f_esc_diff

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
# Phase 2 exit test — dΔ²₂₁/dθ, one astro + one cosmology parameter
# ─────────────────────────────────────────────────────────────────────────────

N, L, Z = 16, 100.0, 9.0


def _delta2_21(s8, ngam_amp, backend, noise, eps_halo):
    """Differentiable toy 21-cm chain hitting every Phase 1+2 G-piece."""
    from beorn.lpt import lpt_ics, lpt_density, lpt_linear_density
    from beorn.lpt.chmf import halo_field_diff

    Om, Ob, h0, ns = 0.31, 0.045, 0.68, 0.97
    if backend == 'jax':
        xp = jnp
        noise_b = jnp.asarray(noise)
    else:
        xp = torch
        noise_b = torch.as_tensor(noise)

    # cosmology: ICs -> density (painted) + linear conditioning field
    dk = lpt_ics(noise_b, L, Om, Ob, h0, ns, s8, backend=backend)
    delta_b = lpt_density(dk, L, Z, Om, backend=backend)
    cell = L / N
    R_cell = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * cell
    dlin = lpt_linear_density(dk, L, Z, Om, backend=backend, R_tophat=R_cell)

    # EPS halo intensity field (G6) with reparameterised shot noise
    M_env = 4.0 / 3.0 * np.pi * R_cell ** 3 * 2.775e11 * 0.31  # ~cell mass
    hmesh, _ = halo_field_diff(dlin, M_env, Z, Om, Ob, h0, ns, s8,
                               cell_volume=cell ** 3, M_min=1e9,
                               n_mass_bins=8, weights='counts',
                               eps=eps_halo, backend=backend)

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

    # painting (G7 + G8 + G11) — heating stays a fixed toy profile (the X-ray
    # source term is out of scope for the Ngam_dot(z) builder above, see G14);
    # ngam_amp/Nion's gradient path runs through the ionization channel only.
    r_nodes = np.linspace(1e-3, 20.0, 60)
    prof_T = 100.0 * np.exp(-r_nodes / 3.0)          # toy heating profile
    prof_T_b = (xp.asarray(prof_T) if backend == 'jax'
                else xp.as_tensor(prof_T).to(dlin.dtype))
    xhii, _, dT = paint_fields_diff(hmesh, Z, L, R_bubble=R_b,
                                    r_temp=r_nodes, prof_temp=prof_T_b,
                                    backend=backend, xHII_floor=1e-4,
                                    spread_iter=4)

    # couplings + dTb (G9) — clamp kernel ringing so log(Tk) stays finite
    dT_pos = xp.where(dT > 0, dT, xp.zeros_like(dT))
    Tk = dT_pos + 2.0                                # adiabatic floor
    xal = 1.0e-2 * hmesh                             # toy Ly-a coupling
    xtot = xal * s_alpha_diff(Z, Tk, 1 - xhii, backend=backend) \
        + x_coll_diff(Z, Tk, 1 - xhii, 1e-3, backend=backend)
    dTb = dtb_diff(Z, Tk, xtot, delta_b, xhii, factor=27.0, backend=backend)

    ps, k = t2c.power_spectrum_1d(dTb, kbins=4, box_dims=L, backend=backend)
    if backend == 'jax':
        return (ps * jnp.asarray(k) ** 3).sum()
    return (ps * torch.as_tensor(np.asarray(k)) ** 3).sum()


@pytest.fixture(scope='module')
def chain_inputs():
    rng = np.random.default_rng(3)
    return (rng.standard_normal((N,) * 3),
            rng.standard_normal((8,) + (N,) * 3))


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
