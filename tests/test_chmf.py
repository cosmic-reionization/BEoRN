"""Unit tests for CHMF/CHMFSampler reading their defaults from
Parameters.halo_sim (parameters restructuring) instead of hardcoded
constructor kwargs."""
import numpy as np
import pytest

from beorn.structs import Parameters
from beorn.lpt import ZeldovichApproximation
from beorn.lpt.chmf import CHMF, CHMFSampler

N, L, Z = 16, 100.0, 10.0


@pytest.fixture
def params():
    p = Parameters()
    p.simulation.Ncell = N
    p.simulation.Lbox = L
    return p


@pytest.fixture
def delta_field(params):
    # Raw (unsmoothed) field -- CHMFSampler smooths it to the conditioning
    # scale internally (issue #54); no R_tophat needed here anymore.
    # get_linear_density is identical regardless of LPT order (it bypasses
    # the displacement/painting machinery entirely -- ZeldovichApproximation
    # here is just the cheapest class to construct, not a 1LPT assumption).
    za = ZeldovichApproximation(params, verbose=False, seed=11)
    return za.get_linear_density(Z).astype(np.float64)


# ── CHMF.delta_c resolution ────────────────────────────────────────────────────

def test_chmf_delta_c_defaults_from_halo_sim(params):
    assert CHMF(params).delta_c == pytest.approx(params.halo_sim.delta_c)


def test_chmf_delta_c_explicit_kwarg_wins(params):
    assert CHMF(params, delta_c=1.5).delta_c == pytest.approx(1.5)


def test_chmf_delta_c_reflects_halo_sim_override(params):
    params.halo_sim.delta_c = 1.7
    assert CHMF(params).delta_c == pytest.approx(1.7)


# ── CHMFSampler.hmf_model resolution ──────────────────────────────────────────

def test_chmfsampler_hmf_model_defaults_from_halo_sim(params):
    sampler = CHMFSampler(params, chmf=CHMF(params))
    assert sampler.hmf_model == params.halo_sim.hmf_model == 'ST'


def test_chmfsampler_hmf_model_explicit_kwarg_wins(params):
    sampler = CHMFSampler(params, chmf=CHMF(params), hmf_model='PS')
    assert sampler.hmf_model == 'PS'


def test_chmfsampler_hmf_model_reflects_halo_sim_override(params):
    params.halo_sim.hmf_model = 'PS'
    sampler = CHMFSampler(params, chmf=CHMF(params))
    assert sampler.hmf_model == 'PS'


# ── CHMFSampler.sample/expected_counts R_env/n_mass_bins/seed resolution ──────

def test_expected_counts_n_mass_bins_defaults_from_halo_sim(params, delta_field):
    params.halo_sim.n_mass_bins = 15
    sampler = CHMFSampler(params, chmf=CHMF(params))
    M_centers, lam = sampler.expected_counts(delta_field, Z)
    assert M_centers.shape == (15,)


def test_sample_seed_defaults_from_halo_sim_and_is_reproducible(params, delta_field):
    params.halo_sim.random_seed = 123
    sampler = CHMFSampler(params, chmf=CHMF(params))
    cat1 = sampler.sample(delta_field, Z, n_mass_bins=10)
    cat2 = sampler.sample(delta_field, Z, n_mass_bins=10)
    np.testing.assert_array_equal(cat1.masses, cat2.masses)
    np.testing.assert_array_equal(cat1.positions, cat2.positions)


# ── halo_sim.halo_mass_max caps the sampled mass range ────────────────────────

def _expected_top_bin_center(M_min, M_max, n_mass_bins):
    """Mirrors CHMFSampler._mass_bins' log-spacing formula exactly."""
    M_edges = np.logspace(np.log10(M_min), np.log10(M_max), n_mass_bins + 1)
    return np.sqrt(M_edges[-2] * M_edges[-1])


def test_halo_mass_max_caps_below_m_env(params):
    """halo_sim.halo_mass_max should cap the upper mass-bin edge below M_env
    when set below it -- previously only M_env*0.999 capped the range."""
    sampler = CHMFSampler(params, chmf=CHMF(params))
    M_min = params.halo_sim.halo_mass_min
    cell_size = params.Lbox_hunits / params.simulation.Ncell
    M_env = sampler.chmf.rho_m * cell_size ** 3

    # Baseline: no halo_mass_max override -> capped only by M_env.
    M_centers_default, _ = sampler._mass_bins(M_env, n_mass_bins=10)
    assert M_centers_default[-1] == pytest.approx(
        _expected_top_bin_center(M_min, M_env * 0.999, 10), rel=1e-10)

    # Cap well below M_env.
    halo_mass_max = M_env / 100.0
    params.halo_sim.halo_mass_max = halo_mass_max
    M_centers_capped, _ = sampler._mass_bins(M_env, n_mass_bins=10)
    assert M_centers_capped[-1] < M_centers_default[-1]
    assert M_centers_capped[-1] == pytest.approx(
        _expected_top_bin_center(M_min, halo_mass_max, 10), rel=1e-10)


def test_halo_mass_min_resolves_from_halo_sim(params, delta_field):
    sampler = CHMFSampler(params, chmf=CHMF(params))
    params.halo_sim.halo_mass_min = 1e9
    M_centers, _ = sampler.expected_counts(delta_field, Z, n_mass_bins=10)
    assert M_centers[0] > 1e8  # would be ~1e8 (old default) without the override


# ── issue #54: CHMFSampler must smooth its own conditioning field ─────────────

def test_environment_default_path_smooths_raw_field_to_sigma2_menv(params, delta_field):
    """R_env=None path: delta_field is raw, but _environment must return a
    field whose variance matches the analytic sigma^2(M_env, z) -- this
    would fail before the fix (returned delta_field completely unmodified)."""
    sampler = CHMFSampler(params, chmf=CHMF(params))
    delta_env, M_env = sampler._environment(delta_field, None)
    sigma2_env = sampler.chmf.sigma2(M_env, Z)
    assert float(np.var(delta_env)) == pytest.approx(sigma2_env, rel=0.3)


def test_environment_explicit_r_env_smooths_once_to_correct_variance(params, delta_field):
    """R_env set explicitly and > cell_size: must smooth exactly once, with
    the analytically-consistent top-hat window -- previously (issue #54)
    this path double-smoothed an already-presmoothed field with an
    inconsistent Gaussian window whenever LPTHaloLoader was used with an
    explicit R_env."""
    sampler = CHMFSampler(params, chmf=CHMF(params))
    cell_size = params.Lbox_hunits / params.simulation.Ncell
    R_env = 3.0 * cell_size
    delta_env, M_env = sampler._environment(delta_field, R_env)
    sigma2_env = sampler.chmf.sigma2(M_env, Z)
    assert float(np.var(delta_env)) == pytest.approx(sigma2_env, rel=0.3)


def test_expected_counts_volume_average_matches_ps_with_raw_field():
    """EPS self-consistency check -- issue #54's actual bug-closure test.

    Volume-averaging the conditional HMF over a RAW (unsmoothed) field must
    reproduce the analytic unconditional Press-Schechter dn/dlnM. Before the
    fix, delta_field was silently used as-is (never smoothed to sigma^2(M_env)),
    biasing this comparison by 5-20% depending on redshift (issue #54's own
    measurement); after the fix, CHMFSampler smooths it internally and this
    comparison holds within ordinary finite-box/grid discreteness.

    Uses a larger grid (N=96) than this file's other tests: at N=48 the
    residual from finite-grid discreteness alone (not the smoothing bug --
    confirmed by averaging over several independent seeds, which barely
    moved the residual) grows toward ~17-19% for the highest tested mass
    bin, too close to a tolerance that must also stay tight enough to catch
    a reintroduced bug (issue #54's own worst-case bias was ~20%). Finer
    grids sample sigma(M)'s underlying power spectrum with less discreteness
    error, matching the trend issue #54 itself reports (~6% residual at
    N=256 for a correctly-smoothed field, vs the much coarser N=48 here).
    """
    p = Parameters()
    p.simulation.Ncell = 96
    p.simulation.Lbox = 100.0
    za = ZeldovichApproximation(p, verbose=False, seed=3)
    delta_raw = za.get_linear_density(Z).astype(np.float64)

    # hmf_model='PS' -- pure EPS conditional sampling is the model whose
    # volume average is exactly the unconditional Press-Schechter curve
    # (the 'ST' rescaling is a separate calibration layered on top).
    sampler = CHMFSampler(p, chmf=CHMF(p), hmf_model='PS')
    cell_size = p.Lbox_hunits / p.simulation.Ncell
    M_env = sampler.chmf.rho_m * cell_size ** 3
    # Restrict the tested mass range to well below M_env: near the cutoff,
    # sigma_eff^2 = sigma2(M) - sigma2_env -> 0, so nu_eff and the
    # conditional HMF become intrinsically singular/noisy -- an expected
    # feature of EPS near M_env, unrelated to the smoothing fix under test.
    p.halo_sim.halo_mass_max = M_env / 30.0
    n_mass_bins = 8
    M_centers, dln_M = sampler._mass_bins(M_env, n_mass_bins)
    V_cell = cell_size ** 3

    _, lam = sampler.expected_counts(delta_raw, Z, n_mass_bins=n_mass_bins)
    dndlnm_measured = lam.mean(axis=(1, 2, 3)) / (dln_M * V_cell)
    dndlnm_analytic = sampler.chmf.hmf_ps(M_centers, Z)

    rel_err = np.abs(dndlnm_measured / dndlnm_analytic - 1.0)
    assert np.all(rel_err < 0.15), rel_err
