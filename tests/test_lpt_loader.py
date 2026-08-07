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
    """Also the canonical test for the issue #56 mismatch warning: seed=1
    diverges from the default cosmo_sim.IC_seed (12345), so LPTHaloLoader
    must warn that the sampled halos won't be spatially correlated with a
    density field built elsewhere from cosmo_sim.IC_seed."""
    custom = ZeldovichApproximation(params, seed=1, verbose=False)
    with pytest.warns(UserWarning, match="not.*correlated"):
        loader = LPTHaloLoader(params, lpt_solver=custom)
    assert loader.lpt_solver is custom


def test_halo_seed_defaults_from_halo_sim_halo_sampler_seed(params):
    params.halo_sim.halo_sampler_seed = 999
    loader = LPTHaloLoader(params)
    assert loader._base_seed == 999


def test_halo_seed_explicit_kwarg_wins(params):
    loader = LPTHaloLoader(params, halo_seed=7)
    assert loader._base_seed == 7


def test_seed_and_halo_seed_are_independent(params):
    """The LPT IC seed (seed=) and the halo-catalog seed (halo_seed=) must
    not conflate -- historically a single `seed` kwarg drove both.
    cosmo_sim.IC_seed is set to match seed= here so this test stays focused
    on independence rather than also triggering the issue #56 mismatch
    warning (covered separately below)."""
    params.cosmo_sim.IC_seed = 1
    loader = LPTHaloLoader(params, seed=1, halo_seed=2)
    assert loader.lpt_solver.seed == 1
    assert loader._base_seed == 2


def test_ic_seed_mismatch_warns(params):
    params.cosmo_sim.IC_seed = 5
    with pytest.warns(UserWarning, match="not.*correlated"):
        LPTHaloLoader(params, seed=6)


def test_ic_seed_no_mismatch_no_warning(params, recwarn):
    """Default construction: halo_sim.IC_seed (None) inherits
    cosmo_sim.IC_seed, so the loader's solver seed matches it and no
    mismatch warning fires."""
    LPTHaloLoader(params)
    assert len(recwarn) == 0


def test_halo_sim_ic_seed_used_when_no_explicit_seed_kwarg(params):
    params.halo_sim.IC_seed = 8
    params.cosmo_sim.IC_seed = 8
    loader = LPTHaloLoader(params)
    assert loader.lpt_solver.seed == 8


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
