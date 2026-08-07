"""Unit tests for the Parameters schema reorganization: CosmoSimParameters
(density_source, mass_assignment, oversample) and the new HaloSimParameters
group."""
import numpy as np
import pytest

from beorn.structs.parameters import Parameters, CosmoSimParameters, HaloSimParameters


@pytest.fixture
def params():
    return Parameters()


# ── CosmoSimParameters ────────────────────────────────────────────────────────

def test_cosmo_sim_density_source_defaults_to_2lpt(params):
    assert params.cosmo_sim.density_source == '2LPT'


def test_cosmo_sim_mass_assignment_defaults_to_cic(params):
    assert params.cosmo_sim.mass_assignment == 'CIC'


def test_cosmo_sim_oversample_defaults_to_one(params):
    assert params.cosmo_sim.oversample == 1


def test_cosmo_sim_no_longer_has_halo_catalogs_thesan_mass_assignment():
    assert not hasattr(CosmoSimParameters(), 'halo_catalogs_thesan_mass_assignment')


def test_simulation_no_longer_has_mass_assignment_or_oversample(params):
    assert not hasattr(params.simulation, 'mass_assignment')
    assert not hasattr(params.simulation, 'oversample')


# ── HaloSimParameters ─────────────────────────────────────────────────────────

def test_halo_sim_defaults_match_prior_hardcoded_constructor_defaults(params):
    """Regression guard: these were previously hardcoded kwargs on
    CHMFSampler/LPTHaloLoader/CHMF -- moving them into Parameters must not
    silently change any of them except the two deliberate ones flagged below
    (hmf_model, and LPTHaloLoader's implicit LPT order, tested elsewhere)."""
    hs = params.halo_sim
    assert hs.delta_c == pytest.approx(1.686)
    assert hs.R_env is None
    assert hs.n_mass_bins == 40
    assert hs.random_seed == 42
    assert hs.mass_assignment == 'NGP'
    assert hs.halo_source == 'CHMF'


def test_halo_sim_hmf_model_default_is_deliberately_st(params):
    """Deliberate default change (not a preserved-behavior regression guard):
    hmf_model now defaults to 'ST', not the old hardcoded 'PS'."""
    assert params.halo_sim.hmf_model == 'ST'


def test_halo_sim_mass_range_independent_of_source(params):
    """halo_sim.halo_mass_min/max are their own fields, decoupled from
    source.halo_mass_min/max (the star-forming/painting cutoff)."""
    assert params.halo_sim.halo_mass_min == pytest.approx(params.source.halo_mass_min)
    assert params.halo_sim.halo_mass_max == pytest.approx(params.source.halo_mass_max)
    params.halo_sim.halo_mass_min = 1e7
    assert params.source.halo_mass_min == pytest.approx(1e8)


def test_halo_sim_unknown_hmf_model_rejected_by_chmfsampler(params):
    from beorn.lpt.chmf import CHMF, CHMFSampler
    with pytest.raises(ValueError, match="Unknown hmf_model"):
        CHMFSampler(params, chmf=CHMF(params), hmf_model='bogus')


# ── Parameters.halo_sim wiring ────────────────────────────────────────────────

def test_parameters_has_halo_sim_field(params):
    assert isinstance(params.halo_sim, HaloSimParameters)


def test_parameters_from_dict_round_trips_halo_sim(tmp_path):
    p = Parameters()
    p.halo_sim.hmf_model = 'PS'
    p.halo_sim.n_mass_bins = 20
    p.cosmo_sim.density_source = '1LPT'
    p.cosmo_sim.mass_assignment = 'TSC'

    yaml_path = tmp_path / "params.yaml"
    p.to_yaml(yaml_path)
    loaded = Parameters.from_yaml(yaml_path)

    assert loaded.halo_sim.hmf_model == 'PS'
    assert loaded.halo_sim.n_mass_bins == 20
    assert loaded.cosmo_sim.density_source == '1LPT'
    assert loaded.cosmo_sim.mass_assignment == 'TSC'


# ── beorn_hash sensitivity (issue: relocated fields must stay hash-visible) ───

def test_beorn_hash_changes_with_cosmo_sim_mass_assignment(params):
    h0 = params.beorn_hash()
    params.cosmo_sim.mass_assignment = 'TSC'
    h1 = params.beorn_hash()
    assert h0 != h1


def test_beorn_hash_changes_with_cosmo_sim_oversample(params):
    h0 = params.beorn_hash()
    params.cosmo_sim.oversample = 4
    h1 = params.beorn_hash()
    assert h0 != h1


def test_beorn_hash_unaffected_by_cosmo_sim_random_seed(params):
    """Rest of cosmo_sim (random_seed, density_source, snapshot_redshifts,
    file_root) stays excluded from beorn_hash, as before -- it's "which data
    source", not the physics model."""
    h0 = params.beorn_hash()
    params.cosmo_sim.random_seed = 999
    h1 = params.beorn_hash()
    assert h0 == h1


def test_beorn_hash_unaffected_by_halo_sim(params):
    """halo_sim is not part of beorn_hash (halo-generation choices are a
    separate concern from the astrophysical/grid model this hash covers)."""
    h0 = params.beorn_hash()
    params.halo_sim.hmf_model = 'PS'
    params.halo_sim.random_seed = 999
    h1 = params.beorn_hash()
    assert h0 == h1
