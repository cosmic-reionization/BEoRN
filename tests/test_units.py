"""Unit tests for beorn.units (issue #49, Phase A scaffolding + Phase B boundary accessors)."""
import numpy as np
import pytest

from beorn.structs.parameters import Parameters
from beorn.structs.halo_catalog import HaloCatalog
from beorn.units import length_factor, mass_factor, rhoc0_physical
from beorn.constants import rhoc0


@pytest.fixture
def params():
    return Parameters()


def test_use_hunits_defaults_true(params):
    assert params.simulation.use_hunits is True


def test_length_factor_is_noop_by_default(params):
    assert length_factor(params) == 1.0


def test_mass_factor_is_noop_by_default(params):
    assert mass_factor(params) == 1.0


def test_length_factor_converts_when_use_hunits_false(params):
    params.simulation.use_hunits = False
    assert length_factor(params) == pytest.approx(1.0 / params.cosmology.h0)


def test_mass_factor_converts_when_use_hunits_false(params):
    params.simulation.use_hunits = False
    assert mass_factor(params) == pytest.approx(1.0 / params.cosmology.h0)


def test_rhoc0_physical_independent_of_use_hunits(params):
    expected = rhoc0 * params.cosmology.h0 ** 2
    assert rhoc0_physical(params) == pytest.approx(expected)
    params.simulation.use_hunits = False
    assert rhoc0_physical(params) == pytest.approx(expected)


# ── Phase B: boundary accessors (Parameters.Lbox_physical, HaloCatalog.positions_physical) ──

def test_lbox_physical_is_noop_by_default(params):
    assert params.Lbox_physical == pytest.approx(params.simulation.Lbox)


def test_lbox_physical_converts_when_use_hunits_false(params):
    params.simulation.use_hunits = False
    assert params.Lbox_physical == pytest.approx(params.simulation.Lbox / params.cosmology.h0)


def _make_catalog(params):
    n = 5
    rng = np.random.default_rng(0)
    positions = rng.random((n, 3)) * params.simulation.Lbox
    masses = np.full(n, 10 * params.source.halo_mass_min)
    return HaloCatalog(positions=positions, masses=masses, parameters=params)


def test_positions_physical_is_noop_by_default(params):
    cat = _make_catalog(params)
    assert np.array_equal(cat.positions_physical, cat.positions)


def test_positions_physical_converts_when_use_hunits_false(params):
    cat = _make_catalog(params)
    params.simulation.use_hunits = False
    expected = cat.positions / params.cosmology.h0
    assert np.allclose(cat.positions_physical, expected)


def test_masses_unaffected_by_use_hunits(params):
    cat = _make_catalog(params)
    masses_before = cat.masses.copy()
    params.simulation.use_hunits = False
    assert np.array_equal(cat.masses, masses_before)


# ── Phase C: derived/output quantities (Parameters.kbins_physical) ──────────

def test_kbins_physical_is_noop_by_default(params):
    assert np.array_equal(params.kbins_physical, params.simulation.kbins)


def test_kbins_physical_converts_when_use_hunits_false(params):
    kbins_hunits = params.simulation.kbins
    params.simulation.use_hunits = False
    expected = kbins_hunits * params.cosmology.h0
    assert np.allclose(params.kbins_physical, expected)
