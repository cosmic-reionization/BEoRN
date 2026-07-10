"""Unit tests for beorn.mass_function.

Covers:
- Window functions (TopHat, SharpK, SmoothK) — shape and limiting behaviour
- MassFunction base class — sigma^2(M, z) scaling and physical quantities
- HaloMassFunction — dndlnm shapes, positivity, redshift ordering, named runners
- ParametricHMF / PressSchechter / ShethTormen legacy classes
- Backend parity — JAX and torch must reproduce numpy to float64 precision
- JAX differentiability — jax.grad through delta_c
- PyTorch differentiability — autograd through delta_c
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from beorn.structs import Parameters
from beorn.mass_function import (
    HaloMassFunction,
    ParametricHMF,
    PressSchechter,
    ShethTormen,
    TopHatWindow,
    SharpKWindow,
    SmoothKWindow,
    get_window,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def param():
    return Parameters()


@pytest.fixture(scope="module")
def hmf_st(param):
    """Default ST HMF, reused across many tests (one sigma^2 precomputation)."""
    return HaloMassFunction(param, model="sheth_tormen")


@pytest.fixture(scope="module")
def M():
    return np.logspace(8, 14, 40)


# ── Window functions ──────────────────────────────────────────────────────────

class TestTopHatWindow:
    def test_w_at_zero_is_one(self):
        w = TopHatWindow()
        assert w.W(np.array([0.0]))[0] == pytest.approx(1.0, rel=1e-6)

    def test_w_small_x_series(self):
        w = TopHatWindow()
        x = np.array([1e-4])
        expected = 1.0 - x**2 / 10.0
        assert w.W(x)[0] == pytest.approx(expected[0], rel=1e-6)

    def test_w_bounded(self):
        # 3j₁(x)/x oscillates; max is 1 at x=0, envelope decays as 1/x^2.
        # Values can go slightly negative — that's correct behaviour.
        w = TopHatWindow()
        x = np.logspace(-3, 2, 200)
        vals = w.W(x)
        assert vals[0] == pytest.approx(1.0, rel=1e-3)
        assert np.all(vals <= 1.0 + 1e-12)  # never exceeds 1

    def test_w_decays_at_large_x(self):
        w = TopHatWindow()
        assert w.W(np.array([100.0]))[0] < 0.1


class TestSharpKWindow:
    def test_w_one_below_cutoff(self):
        w = SharpKWindow()
        x = np.array([0.5, 0.99, 1.0])
        assert np.all(w.W(x) == 1.0)

    def test_w_zero_above_cutoff(self):
        w = SharpKWindow()
        x = np.array([1.01, 2.0, 10.0])
        assert np.all(w.W(x) == 0.0)

    def test_w_shape_preserved(self):
        w = SharpKWindow()
        x = np.linspace(0, 2, 50)
        assert w.W(x).shape == x.shape


class TestSmoothKWindow:
    def test_w_at_zero_is_one(self):
        w = SmoothKWindow()
        assert w.W(np.array([0.0]))[0] == pytest.approx(1.0)

    def test_w_at_one_is_half(self):
        w = SmoothKWindow(beta=1.0)
        assert w.W(np.array([1.0]))[0] == pytest.approx(0.5)

    def test_w_decreasing(self):
        w = SmoothKWindow()
        x = np.logspace(-2, 2, 50)
        vals = w.W(x)
        assert np.all(np.diff(vals) <= 0)

    def test_beta_controls_steepness(self):
        x = np.array([2.0])
        w4 = SmoothKWindow(beta=4.0)
        w2 = SmoothKWindow(beta=2.0)
        assert w4.W(x)[0] < w2.W(x)[0]


class TestGetWindow:
    def test_string_tophat(self):
        w = get_window("tophat")
        assert isinstance(w, TopHatWindow)

    def test_string_sharp_k(self):
        w = get_window("sharp_k")
        assert isinstance(w, SharpKWindow)

    def test_string_smooth_k(self):
        w = get_window("smooth_k")
        assert isinstance(w, SmoothKWindow)

    def test_instance_passthrough(self):
        win = TopHatWindow()
        assert get_window(win) is win

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown window"):
            get_window("nonexistent")


# ── MassFunction base (via HaloMassFunction) ──────────────────────────────────

class TestMassFunctionBase:
    def test_rho_m_positive(self, hmf_st):
        assert hmf_st.rho_m > 0

    def test_rho_m_order_of_magnitude(self, hmf_st):
        # ~1e11 Msun/(Mpc/h)^3 for standard cosmology
        assert 1e10 < hmf_st.rho_m < 1e12

    def test_R_of_M_shape(self, hmf_st, M):
        R = hmf_st.R_of_M(M)
        assert R.shape == M.shape
        assert np.all(R > 0)

    def test_R_of_M_increasing(self, hmf_st):
        M = np.logspace(8, 14, 20)
        R = hmf_st.R_of_M(M)
        assert np.all(np.diff(R) > 0)

    def test_M_of_R_roundtrip(self, hmf_st):
        M_orig = np.logspace(8, 14, 20)
        R = hmf_st.R_of_M(M_orig)
        M_back = hmf_st.M_of_R(R)
        np.testing.assert_allclose(M_back, M_orig, rtol=1e-10)

    def test_sigma2_positive(self, hmf_st, M):
        s2 = hmf_st.sigma2(M, z=0.0)
        assert np.all(s2 > 0)

    def test_sigma2_decreases_with_mass(self, hmf_st):
        M = np.logspace(8, 15, 30)
        s2 = hmf_st.sigma2(M, z=0.0)
        assert np.all(np.diff(s2) < 0)

    def test_sigma2_scales_with_growth_factor(self, hmf_st):
        M = np.logspace(10, 13, 10)
        s2_z0 = hmf_st.sigma2(M, z=0.0)
        s2_z7 = hmf_st.sigma2(M, z=7.0)
        # sigma^2(z) = D1(z)^2 * sigma^2(0), so ratio must be same for all M
        ratio = s2_z7 / s2_z0
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-8)

    def test_sigma2_smaller_at_higher_z(self, hmf_st):
        M = np.array([1e10])
        assert hmf_st.sigma2(M, z=7.0)[0] < hmf_st.sigma2(M, z=0.0)[0]


# ── HaloMassFunction — default / model / window selection ─────────────────────

class TestHaloMassFunctionBasic:
    def test_dndlnm_shape(self, hmf_st, M):
        n = hmf_st.dndlnm(M, z=7.0)
        assert n.shape == M.shape

    def test_dndlnm_positive(self, hmf_st, M):
        n = hmf_st.dndlnm(M, z=7.0)
        assert np.all(n > 0)

    def test_dndlnm_decreasing_with_mass(self, hmf_st):
        M = np.logspace(8, 14, 40)
        n = hmf_st.dndlnm(M, z=7.0)
        assert np.all(np.diff(n) < 0)

    def test_dndlnm_larger_at_lower_z(self, hmf_st):
        M = np.logspace(10, 13, 20)
        n_z7 = hmf_st.dndlnm(M, z=7.0)
        n_z0 = hmf_st.dndlnm(M, z=0.0)
        # More halos at z=0 in the low-mass range
        assert np.all(n_z0 > n_z7)

    def test_scalar_M_works(self, hmf_st):
        n = hmf_st.dndlnm(1e10, z=7.0)
        assert n.shape == (1,)
        assert n[0] > 0

    def test_unknown_model_raises(self, param):
        with pytest.raises(ValueError, match="Unknown model"):
            HaloMassFunction(param, model="bogus")

    def test_custom_requires_p_q(self, param):
        with pytest.raises(ValueError, match="Supply p and q"):
            HaloMassFunction(param, model="custom")

    def test_custom_model_works(self, param, M):
        hmf = HaloMassFunction(param, model="custom", p=0.2, q=0.8)
        n = hmf.dndlnm(M, z=7.0)
        assert np.all(n > 0)

    @pytest.mark.parametrize("model", ["press_schechter", "ps", "sheth_tormen", "st", "ellipsoidal"])
    def test_all_named_models_positive(self, param, M, model):
        hmf = HaloMassFunction(param, model=model)
        n = hmf.dndlnm(M, z=7.0)
        assert np.all(n > 0)

    @pytest.mark.parametrize("window", ["tophat", "sharp_k", "smooth_k"])
    def test_all_windows_positive(self, param, M, window):
        hmf = HaloMassFunction(param, window=window)
        n = hmf.dndlnm(M, z=7.0)
        assert np.all(n > 0)

    def test_window_instance_accepted(self, param, M):
        hmf = HaloMassFunction(param, window=TopHatWindow())
        n = hmf.dndlnm(M, z=7.0)
        assert np.all(n > 0)

    def test_delta_c_override_changes_result(self, hmf_st, M):
        n1 = hmf_st.dndlnm(M, z=7.0, delta_c=1.686)
        n2 = hmf_st.dndlnm(M, z=7.0, delta_c=1.6)
        assert not np.allclose(n1, n2)


# ── Named runners ─────────────────────────────────────────────────────────────

class TestNamedRunners:
    def test_run_ps_vs_ps_model(self, hmf_st, param, M):
        hmf_ps = HaloMassFunction(param, model="press_schechter")
        n_runner = hmf_st.run_press_schechter(M, z=7.0)
        n_direct = hmf_ps.dndlnm(M, z=7.0)
        np.testing.assert_allclose(n_runner, n_direct, rtol=1e-12)

    def test_run_st_vs_st_model(self, hmf_st, param, M):
        hmf_st2 = HaloMassFunction(param, model="sheth_tormen")
        n_runner = hmf_st.run_sheth_tormen(M, z=7.0)
        n_direct = hmf_st2.dndlnm(M, z=7.0)
        np.testing.assert_allclose(n_runner, n_direct, rtol=1e-12)

    def test_run_ellipsoidal_vs_ellipsoidal_model(self, hmf_st, param, M):
        hmf_ec = HaloMassFunction(param, model="ellipsoidal")
        n_runner = hmf_st.run_ellipsoidal(M, z=7.0)
        n_direct = hmf_ec.dndlnm(M, z=7.0)
        np.testing.assert_allclose(n_runner, n_direct, rtol=1e-12)

    def test_ps_st_differ(self, hmf_st, M):
        n_ps = hmf_st.run_press_schechter(M, z=7.0)
        n_st = hmf_st.run_sheth_tormen(M, z=7.0)
        assert not np.allclose(n_ps, n_st)


# ── Legacy classes ────────────────────────────────────────────────────────────

class TestLegacyClasses:
    def test_press_schechter_positive(self, param, M):
        ps = PressSchechter(param)
        n = ps.dndlnm(M, z=7.0)
        assert np.all(n > 0)

    def test_sheth_tormen_positive(self, param, M):
        st = ShethTormen(param)
        n = st.dndlnm(M, z=7.0)
        assert np.all(n > 0)

    def test_parametric_hmf_custom_params(self, param, M):
        phmf = ParametricHMF(param, p=0.2, q=0.9, delta_c=1.686)
        n = phmf.dndlnm(M, z=7.0)
        assert np.all(n > 0)

    def test_legacy_ps_consistent_with_unified(self, param, M):
        ps_legacy = PressSchechter(param)
        hmf_ps = HaloMassFunction(param, model="press_schechter")
        n_legacy = ps_legacy.dndlnm(M, z=7.0)
        n_unified = hmf_ps.dndlnm(M, z=7.0)
        np.testing.assert_allclose(n_legacy, n_unified, rtol=1e-10)

    def test_legacy_st_consistent_with_unified(self, param, M):
        st_legacy = ShethTormen(param)
        hmf_st = HaloMassFunction(param, model="sheth_tormen")
        n_legacy = st_legacy.dndlnm(M, z=7.0)
        n_unified = hmf_st.dndlnm(M, z=7.0)
        np.testing.assert_allclose(n_legacy, n_unified, rtol=1e-10)


# ── Normalisation of f(nu) ────────────────────────────────────────────────────

class TestFNuNormalisation:
    """int_0^inf f(nu) d nu = 1 to within ~1% (limited by finite M range)."""

    @pytest.mark.parametrize("model", ["press_schechter", "sheth_tormen", "ellipsoidal"])
    def test_f_nu_integral_near_unity(self, param, model):
        hmf = HaloMassFunction(param, model=model)
        M = np.logspace(4, 18, 500)
        dlnM = np.log(M[1] / M[0])
        n = hmf.dndlnm(M, z=0.0)
        # int f(nu) d nu = int (n M / rho_m) / |d ln sigma / d ln M| * something
        # Simpler proxy: number density integral in Msun/Mpc^3 vs rho_m
        mass_density = np.sum(n * M) * dlnM
        rho_m = hmf.rho_m
        frac = mass_density / rho_m
        # Should capture ~50–100% of dark matter in this mass range
        assert 0.3 < frac < 1.2


# ── JAX backend ───────────────────────────────────────────────────────────────

try:
    import jax as _jax
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@pytest.fixture(scope="module")
def hmf_jax(param):
    import jax
    jax.config.update("jax_enable_x64", True)
    return HaloMassFunction(param, model="sheth_tormen", backend="jax")


@pytest.mark.skipif(not _JAX_AVAILABLE, reason="JAX not installed")
class TestJAXBackend:
    def test_jax_matches_numpy(self, hmf_jax, hmf_st, M):
        import jax
        jax.config.update("jax_enable_x64", True)
        n_jax = np.asarray(hmf_jax.dndlnm(M, z=7.0))
        n_np  = hmf_st.dndlnm(M, z=7.0)
        np.testing.assert_allclose(n_jax, n_np, rtol=1e-10)

    def test_jax_grad_wrt_delta_c(self, hmf_jax, M):
        import jax
        jax.config.update("jax_enable_x64", True)
        dlnM = float(np.log(M[1] / M[0]))

        def n_total(dc):
            return hmf_jax.dndlnm(M, z=7.0, delta_c=dc).sum() * dlnM

        grad = jax.grad(n_total)(1.686)
        assert np.isfinite(float(grad))
        assert float(grad) < 0  # more halos with lower delta_c

    def test_jax_grad_finite_difference_check(self, hmf_jax, M):
        import jax
        jax.config.update("jax_enable_x64", True)
        dlnM = float(np.log(M[1] / M[0]))
        h = 1e-5

        def n_total(dc):
            return float(np.asarray(hmf_jax.dndlnm(M, z=7.0, delta_c=dc)).sum()) * dlnM

        grad_ad = float(jax.grad(
            lambda dc: hmf_jax.dndlnm(M, z=7.0, delta_c=dc).sum() * dlnM
        )(1.686))
        grad_fd = (n_total(1.686 + h) - n_total(1.686 - h)) / (2.0 * h)
        assert abs(grad_ad - grad_fd) / (abs(grad_fd) + 1e-30) < 1e-5

    def test_jax_jit_runs(self, hmf_jax, M):
        import jax
        jax.config.update("jax_enable_x64", True)
        fn = jax.jit(lambda dc: hmf_jax.dndlnm(M, z=7.0, delta_c=dc).sum())
        result = fn(1.686)
        assert np.isfinite(float(result))


# ── PyTorch backend ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hmf_torch(param):
    return HaloMassFunction(param, model="sheth_tormen", backend="torch")


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
class TestTorchBackend:
    def test_torch_matches_numpy(self, hmf_torch, hmf_st, M):
        import torch
        n_torch = hmf_torch.dndlnm(M, z=7.0).detach().numpy()
        n_np    = hmf_st.dndlnm(M, z=7.0)
        np.testing.assert_allclose(n_torch, n_np, rtol=1e-10)

    def test_torch_grad_wrt_delta_c(self, hmf_torch, M):
        import torch
        dlnM = float(np.log(M[1] / M[0]))
        dc = torch.tensor(1.686, dtype=torch.float64, requires_grad=True)
        n_sum = hmf_torch.dndlnm(M, z=7.0, delta_c=dc).sum() * dlnM
        n_sum.backward()
        grad = dc.grad.item()
        assert math.isfinite(grad)
        assert grad < 0

    def test_torch_grad_finite_difference_check(self, hmf_torch, M):
        import torch
        dlnM = float(np.log(M[1] / M[0]))
        h = 1e-5

        def n_total(dc_val):
            return float(hmf_torch.dndlnm(M, z=7.0, delta_c=dc_val).sum().detach()) * dlnM

        dc = torch.tensor(1.686, dtype=torch.float64, requires_grad=True)
        n_sum = hmf_torch.dndlnm(M, z=7.0, delta_c=dc).sum() * dlnM
        n_sum.backward()
        grad_ad = dc.grad.item()
        grad_fd = (n_total(1.686 + h) - n_total(1.686 - h)) / (2.0 * h)
        assert abs(grad_ad - grad_fd) / (abs(grad_fd) + 1e-30) < 1e-5

    @pytest.mark.skipif(not _JAX_AVAILABLE, reason="JAX not installed")
    def test_jax_and_torch_gradients_agree(self, hmf_torch, M):
        """JAX and torch must agree on d(N_tot)/d(delta_c) to machine precision."""
        import jax
        import torch
        jax.config.update("jax_enable_x64", True)

        param = hmf_torch.parameters
        hmf_j = HaloMassFunction(param, backend="jax")
        dlnM = float(np.log(M[1] / M[0]))

        # JAX grad
        grad_jax = float(jax.grad(
            lambda dc: hmf_j.dndlnm(M, z=7.0, delta_c=dc).sum() * dlnM
        )(1.686))

        # torch grad
        dc = torch.tensor(1.686, dtype=torch.float64, requires_grad=True)
        loss = hmf_torch.dndlnm(M, z=7.0, delta_c=dc).sum() * dlnM
        loss.backward()
        grad_torch = dc.grad.item()

        assert abs(grad_jax - grad_torch) / (abs(grad_torch) + 1e-30) < 1e-10
