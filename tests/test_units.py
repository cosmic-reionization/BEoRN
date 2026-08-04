"""Unit tests for beorn.units and the Lbox_hunits design (issue #49)."""
import numpy as np
import pytest

from beorn.structs.parameters import Parameters
from beorn.structs.halo_catalog import HaloCatalog
from beorn.units import length_factor, mass_factor, rhoc0_physical
from beorn.constants import rhoc0


@pytest.fixture
def params():
    return Parameters()


def test_use_hunits_defaults_false(params):
    assert params.simulation.use_hunits is False


def test_length_factor_is_noop_when_use_hunits_true(params):
    params.simulation.use_hunits = True
    assert length_factor(params) == 1.0


def test_mass_factor_is_noop_when_use_hunits_true(params):
    params.simulation.use_hunits = True
    assert mass_factor(params) == 1.0


def test_length_factor_converts_by_default(params):
    assert length_factor(params) == pytest.approx(1.0 / params.cosmology.h0)


def test_mass_factor_converts_by_default(params):
    assert mass_factor(params) == pytest.approx(1.0 / params.cosmology.h0)


def test_rhoc0_physical_independent_of_use_hunits(params):
    expected = rhoc0 * params.cosmology.h0 ** 2
    assert rhoc0_physical(params) == pytest.approx(expected)
    params.simulation.use_hunits = True
    assert rhoc0_physical(params) == pytest.approx(expected)


# ── Lbox_hunits: the internal, always-Mpc/h resolution of the raw Lbox input ──

def test_lbox_hunits_is_noop_when_use_hunits_true(params):
    params.simulation.use_hunits = True
    assert params.Lbox_hunits == pytest.approx(params.simulation.Lbox)


def test_lbox_hunits_converts_by_default(params):
    # use_hunits=False by default: raw Lbox means physical Mpc, Lbox_hunits
    # converts it to the internal Mpc/h representation.
    assert params.Lbox_hunits == pytest.approx(params.simulation.Lbox * params.cosmology.h0)


def test_lbox_hunits_same_physical_box_either_way(params):
    # Setting Lbox=X with use_hunits=False should describe the SAME physical
    # box as setting Lbox=X*h0 with use_hunits=True -- the whole point of the
    # toggle is that Lbox_hunits (what internal code actually consumes) agrees.
    h0 = params.cosmology.h0
    params.simulation.Lbox = 100.0
    params.simulation.use_hunits = False
    lbox_hunits_physical_input = params.Lbox_hunits

    params.simulation.Lbox = 100.0 * h0
    params.simulation.use_hunits = True
    lbox_hunits_hunits_input = params.Lbox_hunits

    assert lbox_hunits_physical_input == pytest.approx(lbox_hunits_hunits_input)


# ── HaloCatalog.positions_physical (unaffected by the Lbox_hunits redesign --
# positions are always an internal-computation output in Mpc/h, never a raw
# user input, so this boundary accessor's own logic is unchanged) ───────────

def _make_catalog(params):
    n = 5
    rng = np.random.default_rng(0)
    positions = rng.random((n, 3)) * params.Lbox_hunits
    masses = np.full(n, 10 * params.source.halo_mass_min)
    return HaloCatalog(positions=positions, masses=masses, parameters=params)


def test_positions_physical_is_noop_when_use_hunits_true(params):
    params.simulation.use_hunits = True
    cat = _make_catalog(params)
    assert np.array_equal(cat.positions_physical, cat.positions)


def test_positions_physical_converts_by_default(params):
    cat = _make_catalog(params)
    expected = cat.positions / params.cosmology.h0
    assert np.allclose(cat.positions_physical, expected)


def test_masses_unaffected_by_use_hunits(params):
    cat = _make_catalog(params)
    masses_before = cat.masses.copy()
    params.simulation.use_hunits = True
    assert np.array_equal(cat.masses, masses_before)
