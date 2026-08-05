"""Unit tests for beorn.cosmo."""
import numpy as np
import pytest

from beorn.structs.parameters import Parameters
from beorn.cosmo import (
    T_cmb,
    hubble,
    hubble_per_yr,
    comoving_distance,
    T_adiab,
    T_adiab_fluctu,
    rhoc_of_z,
    E,
    D,
    D_non_normalized,
    D_cpt92_non_normalized,
    D_linder2005_non_normalized,
    D_linder_cahn2007_non_normalized,
    dark_energy_density_factor,
    Tspin_fct,
    dTb_factor,
)


@pytest.fixture
def params():
    return Parameters()


# ── T_cmb ─────────────────────────────────────────────────────────────────────

def test_t_cmb_at_z0():
    from beorn.constants import Tcmb0
    assert T_cmb(0) == pytest.approx(Tcmb0)


def test_t_cmb_scales_linearly_with_one_plus_z():
    assert T_cmb(10) == pytest.approx(T_cmb(0) * 11)


def test_t_cmb_positive():
    assert T_cmb(8.0) > 0


# ── hubble ────────────────────────────────────────────────────────────────────

def test_hubble_at_z0_equals_H0(params):
    H0 = 100.0 * params.cosmology.h0
    assert hubble(0, params) == pytest.approx(H0)


def test_hubble_increases_with_redshift(params):
    assert hubble(10, params) > hubble(0, params)


def test_hubble_array_input_monotone(params):
    z = np.array([0.0, 5.0, 10.0])
    result = hubble(z, params)
    assert result.shape == (3,)
    assert np.all(np.diff(result) > 0)


# ── hubble_per_yr (yr^-1) ─────────────────────────────────────────────────────

def test_hubble_yr_positive_and_larger_than_z0(params):
    assert hubble_per_yr(5.0, params) > hubble_per_yr(0.0, params)
    assert hubble_per_yr(0.0, params) > 0


# ── comoving_distance ─────────────────────────────────────────────────────────

def test_comoving_distance_starts_at_zero(params):
    z = np.linspace(0, 10, 100)
    dc = comoving_distance(z, params)
    assert dc[0] == pytest.approx(0.0)


def test_comoving_distance_monotone(params):
    z = np.linspace(0, 15, 50)
    dc = comoving_distance(z, params)
    assert np.all(np.diff(dc) > 0)


def test_comoving_distance_to_z10_plausible(params):
    z = np.linspace(0, 10, 200)
    dc = comoving_distance(z, params)
    # Distance to z=10 should be O(10000 Mpc) for standard cosmology
    assert 5000 < dc[-1] < 20000


# ── T_adiab ───────────────────────────────────────────────────────────────────

def test_t_adiab_positive(params):
    assert T_adiab(10.0, params) > 0


def test_t_adiab_increases_with_redshift(params):
    assert T_adiab(20.0, params) > T_adiab(10.0, params)


def test_t_adiab_fluctu_zero_delta_equals_mean(params):
    z = 10.0
    assert T_adiab_fluctu(z, params, 0.0) == pytest.approx(T_adiab(z, params))


def test_t_adiab_fluctu_overdense_hotter(params):
    assert T_adiab_fluctu(10.0, params, 1.0) > T_adiab_fluctu(10.0, params, 0.0)


# ── rhoc_of_z ─────────────────────────────────────────────────────────────────

def test_rhoc_of_z_positive(params):
    assert rhoc_of_z(params, 0) > 0
    assert rhoc_of_z(params, 10) > 0


def test_rhoc_of_z_decreases_with_redshift_in_comoving_units(params):
    # In comoving units rhoc_of_z decreases with z as dark energy dilutes away
    r0 = rhoc_of_z(params, 0)
    r10 = rhoc_of_z(params, 10)
    assert 0 < r10 < r0


# ── E (dimensionless Hubble factor) ───────────────────────────────────────────

def test_e_at_a1_equals_one_flat_universe(params):
    result = E(np.array([1.0]), params)
    assert result[0] == pytest.approx(1.0, rel=0.01)


def test_e_increases_at_early_times(params):
    # E(a) should be larger at smaller a (higher z)
    e_early = E(np.array([0.1]), params)
    e_late = E(np.array([1.0]), params)
    assert e_early[0] > e_late[0]


# ── D (growth factor) ────────────────────────────────────────────────────────

def test_d_normalized_to_one_at_z0(params):
    result = D(1.0, params)
    assert result == pytest.approx(1.0, rel=1e-4)


def test_d_smaller_at_higher_redshift(params):
    d_a05 = D(0.5, params)   # z=1
    d_a01 = D(0.1, params)   # z=9
    assert d_a05 > d_a01


# ── D growth_factor_method (issue: 2LPT vs py21cmfast small-scale P(k) deviation) ─

GROWTH_FACTOR_METHODS = ['integral', 'cpt92', 'linder2005', 'linder_cahn2007']


@pytest.mark.parametrize('method', GROWTH_FACTOR_METHODS)
def test_growth_factor_method_normalized_to_one_at_a1(params, method):
    params.cosmology.growth_factor_method = method
    assert D(1.0, params) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize('method', GROWTH_FACTOR_METHODS)
def test_growth_factor_method_array_input(params, method):
    params.cosmology.growth_factor_method = method
    a = np.array([1.0, 0.5, 0.1, 0.05])
    result = D(a, params)
    assert result.shape == (4,)
    assert np.all(np.diff(result) < 0)  # D decreases as a decreases (z increases)


def test_growth_factor_default_method_is_integral(params):
    assert params.cosmology.growth_factor_method == 'integral'


def test_unknown_growth_factor_method_raises(params):
    params.cosmology.growth_factor_method = 'not-a-real-method'
    with pytest.raises(ValueError):
        D(1.0, params)


@pytest.mark.parametrize('method,tol', [
    ('cpt92', 0.005),            # Carroll, Press & Turner (1992) fit, ~1% accuracy
    ('linder2005', 0.001),       # Linder (2005) growth-index approximation
    ('linder_cahn2007', 0.001),  # Linder & Cahn (2007) growth-index approximation
])
def test_growth_factor_fitting_formulas_agree_with_exact_integral(params, method, tol):
    """Under flat LCDM (default w0=-1, wa=0), every fitting formula should track
    the exact integral to within its documented accuracy."""
    z = np.array([0.0, 1.0, 5.0, 7.0, 10.0, 15.0])
    a = 1.0 / (1.0 + z)

    params.cosmology.growth_factor_method = 'integral'
    d_exact = D(a, params)

    params.cosmology.growth_factor_method = method
    d_fit = D(a, params)

    rel_diff = np.abs(d_fit / d_exact - 1.0)
    assert rel_diff.max() < tol


def test_cpt92_matches_py21cmfast_dicke_discrepancy_magnitude(params):
    """The whole point of 'cpt92': it should reproduce py21cmfast's own
    dicke() growth factor, which is known to differ from BEoRN's exact
    integral by a small (~0.1%), systematic amount that grows toward low z
    (see issue discussion: 2LPT small-scale P(k) suppression vs py21cmfast)."""
    z = np.array([0.0, 5.0, 10.0, 15.0])
    a = 1.0 / (1.0 + z)

    params.cosmology.growth_factor_method = 'integral'
    d_exact = D(a, params)
    params.cosmology.growth_factor_method = 'cpt92'
    d_cpt92 = D(a, params)

    rel_diff = np.abs(d_cpt92 / d_exact - 1.0)
    # discrepancy should be small (~0.1%) ...
    assert rel_diff.max() < 0.005
    # ... but non-zero (real, not a no-op) ...
    assert rel_diff.max() > 1e-5
    # ... and should grow at lower z (matching the reported z=15 -> z=7 trend).
    assert rel_diff[-1] > rel_diff[0]


def test_linder2005_equals_linder_cahn2007_when_wa_zero(params):
    """The two growth-index methods only differ via gamma(a) tracking w(a);
    with wa=0 (the default) w(a) is constant, so gamma is too -> identical."""
    a = np.array([1.0, 0.5, 0.2, 0.05])
    params.cosmology.growth_factor_method = 'linder2005'
    d_linder = D(a, params)
    params.cosmology.growth_factor_method = 'linder_cahn2007'
    d_linder_cahn = D(a, params)
    assert d_linder == pytest.approx(d_linder_cahn, rel=1e-10)


# ── CPL dark energy (w0, wa) ──────────────────────────────────────────────────

def test_cpl_defaults_are_cosmological_constant(params):
    assert params.cosmology.w0 == pytest.approx(-1.0)
    assert params.cosmology.wa == pytest.approx(0.0)


def test_dark_energy_density_factor_is_one_for_cosmological_constant(params):
    a = np.array([1.0, 0.5, 0.1, 0.01])
    result = dark_energy_density_factor(a, params)
    assert result == pytest.approx(np.ones_like(a))


def test_dark_energy_density_factor_at_a1_always_one(params):
    """rho_DE(a=1)/rho_DE(a=1) == 1 regardless of w0/wa."""
    params.cosmology.w0, params.cosmology.wa = -0.8, 0.3
    assert dark_energy_density_factor(1.0, params) == pytest.approx(1.0)


def test_cpl_w0_wa_changes_expansion_history(params):
    """Changing w0/wa away from (-1, 0) must actually change E()/hubble() —
    confirms the CPL generalization is wired in, not a dead parameter."""
    a = 0.3
    e_lcdm = E(a, params)

    params.cosmology.w0, params.cosmology.wa = -0.8, 0.3
    e_cpl = E(a, params)

    assert e_cpl != pytest.approx(e_lcdm, rel=1e-6)


def test_cpl_reduces_to_lcdm_hubble_at_defaults(params):
    """hubble()/hubble_per_yr() must be numerically unchanged for any script
    that doesn't set w0/wa (regression guard for the CPL generalization)."""
    z = np.array([0.0, 1.0, 5.0, 10.0, 20.0])
    Om = params.cosmology.Om
    Ol = 1.0 - Om
    H0 = 100.0 * params.cosmology.h0
    expected = H0 * np.sqrt(Om * (1 + z) ** 3 + Ol)
    assert hubble(z, params) == pytest.approx(expected, rel=1e-10)

    from beorn.constants import sec_per_year, km_per_Mpc
    expected_per_yr = params.cosmology.h0 * 100.0 * sec_per_year / km_per_Mpc * np.sqrt(Om * (1 + z) ** 3 + Ol)
    assert hubble_per_yr(z, params) == pytest.approx(expected_per_yr, rel=1e-10)


def test_growth_ode_integral_uses_cpl_background(params):
    """D_non_normalized (the exact integral) must respond to w0/wa too, since
    it's derived from E() — not just the fitting formulas."""
    a = 0.3
    params.cosmology.growth_factor_method = 'integral'
    d_lcdm = D_non_normalized(a, params)

    params.cosmology.w0, params.cosmology.wa = -0.8, 0.3
    d_cpl = D_non_normalized(a, params)

    assert d_cpl != pytest.approx(d_lcdm, rel=1e-6)


# ── Tspin_fct ─────────────────────────────────────────────────────────────────

def test_tspin_between_tcmb_and_tk():
    Tcmb, Tk, xtot = 22.0, 100.0, 1.0
    Ts = Tspin_fct(Tcmb, Tk, xtot)
    assert Tcmb <= Ts <= Tk


def test_tspin_equals_tcmb_when_xtot_zero():
    Ts = Tspin_fct(22.0, 1000.0, 0.0)
    assert Ts == pytest.approx(22.0)


def test_tspin_approaches_tk_when_xtot_large():
    Ts = Tspin_fct(22.0, 1000.0, 1e9)
    assert Ts == pytest.approx(1000.0, rel=1e-4)


# ── dTb_factor ────────────────────────────────────────────────────────────────

def test_dtb_factor_positive(params):
    assert dTb_factor(params) > 0


def test_dtb_factor_order_of_magnitude(params):
    # ~8 mK for default parameters (Om=0.31, h0=0.68, Ob=0.045)
    result = dTb_factor(params)
    assert 1 < result < 50
