"""Unit tests for beorn.astro, beorn.cross_sections, beorn.couplings."""
import numpy as np
import pytest

from beorn.structs.parameters import Parameters
from beorn.astro import S_fct, f_star_Halo, f_esc, f_Xh, eps_xray
from beorn.cross_sections import sigma_HI, sigma_HeI, sigma_HeII, alpha_HII
from beorn.couplings import x_coll, S_alpha, eps_lyal


@pytest.fixture
def params():
    return Parameters()


# ── S_fct ─────────────────────────────────────────────────────────────────────

def test_s_fct_approaches_one_when_mh_much_larger_than_mt():
    result = S_fct(Mh=1e15, Mt=1e8, g3=4, g4=-1)
    assert abs(result - 1.0) < 1e-3


def test_s_fct_scalar_and_array_consistent():
    Mh_arr = np.array([1e10, 1e11, 1e12])
    scalar_results = np.array([S_fct(m, 1e8, 4, -1) for m in Mh_arr])
    array_result = S_fct(Mh_arr, 1e8, 4, -1)
    np.testing.assert_allclose(array_result, scalar_results, rtol=1e-10)


def test_s_fct_monotone_with_mass():
    masses = np.logspace(7, 14, 20)
    values = S_fct(masses, Mt=1e8, g3=4, g4=-1)
    assert np.all(np.diff(values) >= 0)


# ── f_star_Halo ───────────────────────────────────────────────────────────────

def test_f_star_halo_below_mass_min_is_zero(params):
    Mh = np.array([1e5, 1e6, 1e7, params.source.halo_mass_min * 0.5])
    result = f_star_Halo(params, Mh)
    np.testing.assert_array_equal(result, 0.0)


def test_f_star_halo_bounded_zero_to_one(params):
    Mh = np.logspace(8, 15, 50)
    result = f_star_Halo(params, Mh)
    assert np.all(result >= 0)
    assert np.all(result <= 1)


def test_f_star_halo_nonzero_above_min(params):
    Mh = np.array([1e12])
    result = f_star_Halo(params, Mh)
    assert result[0] > 0


# ── f_esc ─────────────────────────────────────────────────────────────────────

def test_f_esc_bounded_zero_to_one(params):
    Mh = np.logspace(8, 15, 50)
    result = f_esc(params, Mh)
    assert np.all(result >= 0)
    assert np.all(result <= 1)


def test_f_esc_constant_when_pl_zero(params):
    # Default pl_esc=0 => fesc = min(f0_esc, 1) everywhere
    Mh = np.logspace(8, 14, 10)
    result = f_esc(params, Mh)
    expected = min(params.source.f0_esc, 1.0)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


# ── f_Xh ──────────────────────────────────────────────────────────────────────

def test_f_xh_zero_gives_zero():
    assert f_Xh(0.0) == pytest.approx(0.0)


def test_f_xh_one_gives_one():
    assert f_Xh(1.0) == pytest.approx(1.0)


def test_f_xh_monotone():
    xe = np.linspace(0.01, 1.0, 50)
    vals = f_Xh(xe)
    assert np.all(np.diff(vals) > 0)


def test_f_xh_power_law():
    xe = 0.5
    assert f_Xh(xe) == pytest.approx(0.5 ** 0.225)


# ── eps_xray ──────────────────────────────────────────────────────────────────

def test_eps_xray_positive(params):
    nu = np.logspace(17, 19, 10)
    result = eps_xray(nu, params)
    assert np.all(result > 0)


def test_eps_xray_powerlaw_shape(params):
    # eps ~ nu^(-(sed_xray + 1))  =>  eps(nu2)/eps(nu1) = (nu2/nu1)^(-(sed+1))
    nu1, nu2 = 1e17, 2e17
    r1 = eps_xray(nu1, params)
    r2 = eps_xray(nu2, params)
    sed = params.source.alS_xray
    expected_ratio = (nu2 / nu1) ** (-(sed + 1))
    assert r2 / r1 == pytest.approx(expected_ratio, rel=1e-5)


# ── sigma_HI ──────────────────────────────────────────────────────────────────

def test_sigma_hi_positive_above_threshold():
    sigma = sigma_HI(13.6)
    assert sigma > 0


def test_sigma_hi_decreases_with_energy():
    E = np.logspace(np.log10(13.6), 3, 30)
    sigma = sigma_HI(E)
    assert np.all(np.diff(sigma) < 0)


def test_sigma_hi_order_of_magnitude():
    # At 13.6 eV, HI cross section ~6e-18 cm^2
    sigma = sigma_HI(13.6)
    assert 1e-19 < sigma < 1e-16


def test_sigma_hi_array_input():
    E = np.array([15.0, 50.0, 200.0])
    result = sigma_HI(E)
    assert result.shape == (3,)


# ── sigma_HeI ─────────────────────────────────────────────────────────────────

def test_sigma_hei_non_negative():
    E = np.logspace(np.log10(24.6), 3, 20)
    sigma = sigma_HeI(E)
    assert np.all(sigma >= 0)


def test_sigma_hei_decreases_with_energy():
    E = np.logspace(2, 3, 20)
    sigma = sigma_HeI(E)
    assert np.all(np.diff(sigma) <= 0)


# ── sigma_HeII ────────────────────────────────────────────────────────────────

def test_sigma_heii_non_negative():
    E = np.logspace(2, 3, 20)
    sigma = sigma_HeII(E)
    assert np.all(sigma >= 0)


def test_sigma_heii_larger_than_hi_at_same_energy():
    E = 100.0
    assert sigma_HeII(E) > sigma_HI(E)


# ── alpha_HII ─────────────────────────────────────────────────────────────────

def test_alpha_hii_at_1e4K():
    alpha = alpha_HII(1e4)
    assert alpha == pytest.approx(2.6e-13, rel=1e-6)


def test_alpha_hii_decreases_with_temperature():
    T = np.logspace(3, 6, 20)
    alpha = alpha_HII(T)
    assert np.all(np.diff(alpha) < 0)


def test_alpha_hii_array_input():
    T = np.array([1e4, 2e4, 5e4])
    result = alpha_HII(T)
    assert result.shape == (3,)


# ── S_alpha ───────────────────────────────────────────────────────────────────

def test_s_alpha_between_zero_and_one():
    result = S_alpha(z=10.0, Tk=1000.0, xHI=0.5)
    assert 0 < result <= 1.0


def test_s_alpha_approaches_one_at_high_temperature():
    result = S_alpha(z=10.0, Tk=1e6, xHI=0.5)
    assert result > 0.99


def test_s_alpha_decreases_with_lower_temperature():
    s_hot = S_alpha(z=10.0, Tk=1e5, xHI=0.5)
    s_cold = S_alpha(z=10.0, Tk=100.0, xHI=0.5)
    assert s_hot > s_cold


# ── x_coll ────────────────────────────────────────────────────────────────────

def test_x_coll_positive():
    result = x_coll(z=10.0, Tk=1000.0, xHI=0.9, rho_b=1e-4)
    assert result > 0


def test_x_coll_scales_with_density():
    low = x_coll(z=10.0, Tk=1000.0, xHI=0.9, rho_b=1e-5)
    high = x_coll(z=10.0, Tk=1000.0, xHI=0.9, rho_b=1e-3)
    assert high > low


# ── eps_lyal ──────────────────────────────────────────────────────────────────

def test_eps_lyal_positive(params):
    from beorn.constants import nu_al, nu_LL
    nu = np.linspace(nu_al * 1.01, nu_LL * 0.99, 10)
    result = eps_lyal(nu, params)
    assert np.all(result > 0)
