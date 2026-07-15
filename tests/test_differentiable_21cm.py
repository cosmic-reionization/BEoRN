"""Phase 2 (issue #42) differentiable 21-cm tests: G7/G8/G9/G12/G13 + exit test.

The exit test differentiates Δ²₂₁(k) — the power spectrum of the dTb field
from a full differentiable chain (ICs → LPT density → EPS halo field →
bubble/profile painting → couplings → dTb) — w.r.t. one astro parameter (the
ionizing-photon amplitude, standing in for Nion) and one cosmology parameter
(σ₈), in jax and torch, against central finite differences.
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
)

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

    # astro: bubble radius from the photon-rate ODE (G12), amp ~ Nion
    z_nodes = np.linspace(25.0, Z, 30)
    Ngam = ngam_amp * (1.0e51 * (1 + xp.zeros(z_nodes.size,
                                              dtype=dlin.dtype)))
    R_b = bubble_radius_diff(z_nodes, Ngam, Om, Ob, h0, backend=backend)[-1]

    # painting (G7 + G8 + G11)
    r_nodes = np.linspace(1e-3, 20.0, 60)
    prof_T = 100.0 * np.exp(-r_nodes / 3.0)          # toy heating profile
    prof_T_b = ngam_amp * (xp.asarray(prof_T) if backend == 'jax'
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
@pytest.mark.parametrize('pname,x0', [('sigma_8', 0.82), ('ngam_amp', 1.0)])
def test_exit_grad_jax_21cm(chain_inputs, pname, x0):
    noise, eps_halo = chain_inputs

    def S(val):
        s8 = val if pname == 'sigma_8' else 0.82
        amp = val if pname == 'ngam_amp' else 1.0
        return _delta2_21(s8, amp, 'jax', noise, eps_halo)

    g = jax.grad(S)(x0)
    h = 3e-4 * x0
    fd = (S(x0 + h) - S(x0 - h)) / (2 * h)
    assert float(g) == pytest.approx(float(fd), rel=5e-3)


@pytest.mark.skipif(not (_T2C_BACKEND and _TORCH),
                    reason='needs torch + tools21cm differentiable backend')
def test_exit_grad_torch_matches_jax_21cm(chain_inputs):
    noise, eps_halo = chain_inputs
    g_jax = float(jax.grad(
        lambda a: _delta2_21(0.82, a, 'jax', noise, eps_halo))(1.0))

    amp = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    out = _delta2_21(0.82, amp, 'torch', noise, eps_halo)
    out.backward()
    assert float(amp.grad) == pytest.approx(g_jax, rel=1e-4)
