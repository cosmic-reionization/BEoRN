"""Unit tests for beorn.cosmo."""
import numpy as np
import pytest

from beorn.structs.parameters import Parameters
from beorn.cosmo import (
    T_cmb,
    hubble,
    Hubble,
    comoving_distance,
    T_adiab,
    T_adiab_fluctu,
    rhoc_of_z,
    E,
    D,
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


# ── Hubble (yr^-1) ────────────────────────────────────────────────────────────

def test_hubble_yr_positive_and_larger_than_z0(params):
    # Hubble and hubble use slightly different Omega_Lambda conventions;
    # test each function independently rather than cross-comparing.
    assert Hubble(5.0, params) > Hubble(0.0, params)
    assert Hubble(0.0, params) > 0


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
