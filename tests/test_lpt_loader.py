"""Unit tests for LPTHaloLoader's parameters-restructuring changes: default
LPT order (now 2LPT, matching cosmo_sim.density_source), the seed/halo_seed
split, and halo_sim.* resolution for R_env/n_mass_bins/delta_c/hmf_model."""
import numpy as np
import pytest

from beorn.structs import Parameters
from beorn.lpt import SecondOrderLPT, ZeldovichApproximation
from beorn.load_input_data import LPTHaloLoader

N, L = 16, 100.0


@pytest.fixture
def params():
    p = Parameters()
    p.simulation.Ncell = N
    p.simulation.Lbox = L
    p.solver.redshifts = np.array([10.0, 8.0])
    return p


def test_default_lpt_solver_is_second_order(params):
    """Deliberate default change: matches cosmo_sim.density_source == '2LPT'."""
    loader = LPTHaloLoader(params)
    assert isinstance(loader.lpt_solver, SecondOrderLPT)


def test_explicit_lpt_solver_overrides_default(params):
    custom = ZeldovichApproximation(params, seed=1, verbose=False)
    loader = LPTHaloLoader(params, lpt_solver=custom)
    assert loader.lpt_solver is custom


def test_halo_seed_defaults_from_halo_sim_random_seed(params):
    params.halo_sim.random_seed = 999
    loader = LPTHaloLoader(params)
    assert loader._base_seed == 999


def test_halo_seed_explicit_kwarg_wins(params):
    loader = LPTHaloLoader(params, halo_seed=7)
    assert loader._base_seed == 7


def test_seed_and_halo_seed_are_independent(params):
    """The LPT IC seed (seed=) and the halo-catalog seed (halo_seed=) must
    not conflate -- historically a single `seed` kwarg drove both."""
    loader = LPTHaloLoader(params, seed=1, halo_seed=2)
    assert loader.lpt_solver.seed == 1
    assert loader._base_seed == 2


def test_r_env_and_n_mass_bins_resolve_from_halo_sim(params):
    params.halo_sim.R_env = 2.0
    params.halo_sim.n_mass_bins = 12
    loader = LPTHaloLoader(params)
    assert loader.R_env == pytest.approx(2.0)
    assert loader.n_mass_bins == 12


def test_load_halo_catalog_reproducible(params):
    loader = LPTHaloLoader(params, n_mass_bins=8)
    cat1 = loader.load_halo_catalog(0)
    cat2 = loader.load_halo_catalog(0)
    np.testing.assert_array_equal(cat1.masses, cat2.masses)
    np.testing.assert_array_equal(cat1.positions, cat2.positions)
