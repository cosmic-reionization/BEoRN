"""Unit tests for HaloSimParameters.field_oversample end-to-end in
LPTHaloLoader (issue #56): the CHMF's own conditioning field is generated at
a finer, phase-synchronized resolution and decimated back down to Ncell,
reducing the R_env top-hat window's residual variance bias.
"""
import warnings

import numpy as np
import pytest

from beorn.structs import Parameters
from beorn.load_input_data.lpt_loader import LPTHaloLoader
from beorn.cosmo import D as growth_D
from beorn.lpt.chmf import CHMF

N, L, Z = 16, 100.0, 10.0
SEED = 12345


def _params(field_oversample=1):
    p = Parameters()
    p.simulation.Ncell = N
    p.simulation.Lbox = L
    p.solver.redshifts = np.array([10.0, 8.0])
    p.cosmo_sim.IC_seed = SEED
    p.halo_sim.IC_seed = None
    p.halo_sim.field_oversample = field_oversample
    return p


def test_field_oversample_one_does_not_use_fine_path():
    loader = LPTHaloLoader(_params(field_oversample=1))
    assert loader._fine_delta_k is None
    assert loader._N_fine is None


def test_field_oversample_one_matches_pre_existing_behaviour(monkeypatch):
    """Backward-compat regression: field_oversample=1 everywhere must still
    call CHMFSampler.sample with a raw delta_field, not precomputed_delta_env."""
    loader = LPTHaloLoader(_params(field_oversample=1))
    captured = {}
    original_sample = loader.sampler.sample

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(loader.sampler, 'sample', spy)
    loader.load_halo_catalog(0)
    assert captured.get('precomputed_delta_env') is None


def test_field_oversample_four_runs_end_to_end():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loader = LPTHaloLoader(_params(field_oversample=4))
        cat = loader.load_halo_catalog(0)

    assert loader._fine_delta_k is not None
    assert loader._N_fine == N * 4
    assert len(cat.masses) > 0
    assert np.all(cat.positions >= 0) and np.all(cat.positions <= loader.parameters.Lbox_hunits)


def test_field_oversample_reduces_conditioning_field_variance_bias():
    """The whole point of issue #56's field_oversample: the presmoothed,
    decimated conditioning field's variance should track the analytic
    sigma^2(M_env, z) much more closely than smoothing at Ncell directly."""
    p = _params(field_oversample=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loader = LPTHaloLoader(p)

    z = Z
    delta_coarse = loader.lpt_solver.get_linear_density(z)
    delta_env_coarse, M_env = loader.sampler._environment(delta_coarse, None)

    D1 = growth_D(1.0 / (1.0 + z), p) / growth_D(1.0, p)
    delta_fine = np.fft.irfftn(
        D1 * loader._fine_delta_k, s=(loader._N_fine,) * 3,
    ).astype(np.float32)
    delta_env_fine, M_env_fine = loader.sampler._environment(delta_fine, None)
    factor = loader._N_fine // N
    delta_env_decimated = delta_env_fine[::factor, ::factor, ::factor]

    assert M_env == M_env_fine
    chmf = CHMF(p, power_spectrum=loader.lpt_solver.power_spectrum)
    sigma2_analytic = chmf.sigma2(M_env, z)
    bias_coarse = abs(delta_env_coarse.var() - sigma2_analytic) / sigma2_analytic
    bias_fine = abs(delta_env_decimated.var() - sigma2_analytic) / sigma2_analytic

    assert bias_fine < bias_coarse


def test_field_oversample_conditioning_field_correlates_with_coarse_density():
    """The fine-derived conditioning field must stay spatially correlated
    with the same loader's own coarse density field -- otherwise the sampled
    halos would decorrelate from the density field used elsewhere."""
    p = _params(field_oversample=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loader = LPTHaloLoader(p)

    z = Z
    delta_coarse = loader.lpt_solver.get_linear_density(z)
    delta_env_coarse, _ = loader.sampler._environment(delta_coarse, None)

    D1 = growth_D(1.0 / (1.0 + z), p) / growth_D(1.0, p)
    delta_fine = np.fft.irfftn(
        D1 * loader._fine_delta_k, s=(loader._N_fine,) * 3,
    ).astype(np.float32)
    delta_env_fine, _ = loader.sampler._environment(delta_fine, None)
    factor = loader._N_fine // N
    delta_env_decimated = delta_env_fine[::factor, ::factor, ::factor]

    corr = np.corrcoef(delta_env_coarse.ravel(), delta_env_decimated.ravel())[0, 1]
    assert corr > 0.9


def test_field_oversample_warns_and_is_ignored_with_explicit_lpt_solver():
    from beorn.lpt import ZeldovichApproximation
    p = _params(field_oversample=4)
    custom = ZeldovichApproximation(p, seed=p.cosmo_sim.IC_seed, verbose=False)
    with pytest.warns(UserWarning, match="ignored when an explicit lpt_solver"):
        loader = LPTHaloLoader(p, lpt_solver=custom)
    assert loader._fine_delta_k is None


def test_field_oversample_inherits_from_cosmo_sim_when_halo_sim_is_none():
    p = _params(field_oversample=1)
    p.halo_sim.field_oversample = None
    p.cosmo_sim.field_oversample = 4
    loader = LPTHaloLoader(p)
    assert loader._fine_delta_k is not None
    assert loader._N_fine == N * 4
