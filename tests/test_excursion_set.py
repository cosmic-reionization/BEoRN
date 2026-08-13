"""Unit tests for the deterministic excursion-set halo finder (issue:
massive-halo robustness -- see CHMF.barrier/tophat_smooth_static prereqs,
ExcursionSetFinder, and CHMFSampler's deterministic_mass_fraction coupling).
"""
import numpy as np
import pytest

from beorn.structs import Parameters
from beorn.lpt import ZeldovichApproximation, CHMF, CHMFSampler, ExcursionSetFinder
from beorn.load_input_data import LPTHaloLoader

N, L = 64, 200.0
Z_LOW = 0.5  # redshift with plenty of massive structure, for completeness tests


@pytest.fixture
def params():
    p = Parameters()
    p.simulation.Ncell = N
    p.simulation.Lbox = L
    p.simulation.use_hunits = True
    return p


@pytest.fixture
def chmf(params):
    return CHMF(params)


# ── ExcursionSetFinder basics ─────────────────────────────────────────────────

def test_finder_rejects_m_split_below_field_resolution(params, chmf):
    delta = np.zeros((N, N, N), dtype=np.float64)
    finder = ExcursionSetFinder(chmf)
    M_env = chmf.rho_m * (params.Lbox_hunits / N) ** 3
    with pytest.raises(ValueError, match="own cell size"):
        finder.find(delta, Z_LOW, M_split=M_env / 100.0)


def test_finder_rejects_m_max_below_m_split(params, chmf):
    delta = np.zeros((N, N, N), dtype=np.float64)
    finder = ExcursionSetFinder(chmf)
    M_env = chmf.rho_m * (params.Lbox_hunits / N) ** 3
    with pytest.raises(ValueError, match="must exceed"):
        finder.find(delta, Z_LOW, M_split=M_env, M_max=M_env / 2.0)


def test_finder_returns_empty_catalog_for_empty_field(params, chmf):
    """A field with no overdensity anywhere should find zero halos and
    claim zero cells."""
    delta = np.zeros((N, N, N), dtype=np.float64)
    finder = ExcursionSetFinder(chmf)
    M_env = chmf.rho_m * (params.Lbox_hunits / N) ** 3
    cat, claimed = finder.find(delta, Z_LOW, M_split=M_env, n_scales=8)
    assert cat.masses.size == 0
    assert cat.positions.shape == (0, 3)
    assert not np.any(claimed)


def test_finder_finds_massive_halos_at_low_z(params, chmf):
    """Sanity check: a real LPT field at z=0.5 (plenty of structure) must
    produce a non-trivial number of halos above M_split, with masses
    strictly exceeding M_split (the entire point of this tier)."""
    za = ZeldovichApproximation(params, verbose=False, seed=11)
    delta = za.get_linear_density(Z_LOW).astype(np.float64)
    M_env = chmf.rho_m * (params.Lbox_hunits / N) ** 3

    finder = ExcursionSetFinder(chmf)
    cat, claimed = finder.find(delta, Z_LOW, M_split=M_env, n_scales=24)

    assert cat.masses.size > 0
    assert np.all(cat.masses >= M_env)
    assert claimed.mean() > 0.0

    # Mass convention (post-fix): every accepted patch gets the nominal
    # filter mass M(R) at its crossing scale, not a pixel-count-derived
    # value -- so every catalog mass must be exactly one of the walk's
    # log-spaced M(R) nodes (mirroring find()'s own internal M_values).
    M_max = 0.1 * chmf.rho_m * params.Lbox_hunits ** 3
    M_values = np.logspace(np.log10(M_max), np.log10(M_env), 24)
    assert np.all(np.isin(cat.masses, M_values))

    # The fixed convention assigns >= the old (undercounting) pixel-count
    # mass would have for the same claimed volume -- a directional sanity
    # check standing in for the old exact accounting identity, which relied
    # on the since-removed pixel-count mass convention.
    assert cat.masses.sum() >= chmf.rho_m * np.count_nonzero(claimed) * (params.Lbox_hunits / N) ** 3


def test_finder_small_patch_rejected_below_min_patch_mass(params, chmf):
    """A single-cell density spike produces a tiny (~1-cell) patch at small
    R; a min_patch_mass above that patch's own mass must reject it, leaving
    its cell unclaimed for a smaller scale that never comes (n_scales small
    here) -- i.e. absent from both the catalog and the claimed mask."""
    delta = np.zeros((N, N, N), dtype=np.float64)
    delta[N // 2, N // 2, N // 2] = 10.0  # far above any barrier, localized

    M_env = chmf.rho_m * (params.Lbox_hunits / N) ** 3
    cell_volume = (params.Lbox_hunits / N) ** 3
    one_cell_mass = chmf.rho_m * cell_volume

    finder = ExcursionSetFinder(chmf)
    cat, claimed = finder.find(
        delta, Z_LOW, M_split=M_env, n_scales=16,
        min_patch_mass=one_cell_mass * 5,  # stricter than the spike's own patch
    )
    assert cat.masses.size == 0
    assert not np.any(claimed)


def test_finder_small_patch_accepted_when_min_patch_mass_low(params, chmf):
    """Same spike, but with min_patch_mass at its default (M_split) --
    should still reject it since M_split > one_cell_mass; lowering
    min_patch_mass explicitly below the patch's own mass must accept it."""
    delta = np.zeros((N, N, N), dtype=np.float64)
    delta[N // 2, N // 2, N // 2] = 10.0

    M_env = chmf.rho_m * (params.Lbox_hunits / N) ** 3
    cell_volume = (params.Lbox_hunits / N) ** 3
    one_cell_mass = chmf.rho_m * cell_volume

    finder = ExcursionSetFinder(chmf)
    cat, claimed = finder.find(
        delta, Z_LOW, M_split=M_env, n_scales=16,
        min_patch_mass=one_cell_mass * 0.5,
    )
    assert cat.masses.size >= 1
    assert claimed[N // 2, N // 2, N // 2]


# ── Completeness vs. analytic PS (loose tolerance, single realization) ───────

def test_completeness_order_of_magnitude_matches_analytic_ps(params, chmf):
    """Total mass claimed above M_split, as a fraction of the box mass,
    should be within an order of magnitude of the analytic Press-Schechter
    collapsed-mass-fraction above the same threshold -- a coarse sanity
    check (not a percent-level statistical test: a single N=64 realization
    has large sample variance for the rare, massive tail)."""
    za = ZeldovichApproximation(params, verbose=False, seed=7)
    delta = za.get_linear_density(Z_LOW).astype(np.float64)
    M_env = chmf.rho_m * (params.Lbox_hunits / N) ** 3

    finder = ExcursionSetFinder(chmf)
    cat, _ = finder.find(delta, Z_LOW, M_split=M_env, n_scales=24)
    measured_frac = cat.masses.sum() / (chmf.rho_m * params.Lbox_hunits ** 3)

    M_test = np.logspace(np.log10(M_env), np.log10(M_env * 50), 500)
    dndlnm = chmf.hmf_ps(M_test, Z_LOW)
    mass_density = np.trapezoid(dndlnm * M_test, np.log(M_test))
    analytic_frac = mass_density / chmf.rho_m

    assert measured_frac > 0
    assert 0.1 < measured_frac / analytic_frac < 10.0


# ── CHMFSampler.deterministic_mass_fraction coupling (direct math check) ────

def test_deterministic_mass_fraction_rescales_expected_counts_exactly(params):
    za = ZeldovichApproximation(params, verbose=False, seed=5)
    delta = za.get_linear_density(Z_LOW).astype(np.float64)
    sampler = CHMFSampler(params, chmf=CHMF(params))

    M_centers_ref, lam_ref = sampler.expected_counts(delta, Z_LOW, n_mass_bins=10)

    rng = np.random.default_rng(0)
    fraction = rng.uniform(0.0, 0.9, size=(N, N, N))
    M_centers, lam = sampler.expected_counts(
        delta, Z_LOW, n_mass_bins=10, deterministic_mass_fraction=fraction,
    )

    np.testing.assert_array_equal(M_centers, M_centers_ref)
    np.testing.assert_allclose(lam, lam_ref * (1.0 - fraction)[None, ...], rtol=1e-10)


def test_m_split_caps_mass_bins_below_halo_mass_max(params):
    sampler = CHMFSampler(params, chmf=CHMF(params))
    M_env = sampler.chmf.rho_m * (params.Lbox_hunits / N) ** 3
    M_split = M_env / 10.0

    M_centers_default, _ = sampler._mass_bins(M_env, n_mass_bins=10)
    M_centers_split, _ = sampler._mass_bins(M_env, n_mass_bins=10, M_split=M_split)

    assert M_centers_split[-1] < M_centers_default[-1]
    assert M_centers_split[-1] <= M_split


def test_m_split_bin_edge_stays_strictly_below_m_split(params):
    """The stochastic tier's mass range must stay strictly below M_split,
    not touch it -- otherwise its top bin edge and the excursion-set
    tier's own smallest halo mass (exactly M_split) would coincide."""
    sampler = CHMFSampler(params, chmf=CHMF(params))
    M_env = sampler.chmf.rho_m * (params.Lbox_hunits / N) ** 3
    M_split = M_env / 10.0

    # Delegates to _mass_range rather than re-deriving the cap logic inline,
    # so this test can't drift out of sync with it again (issue: this test
    # broke when halo_mass_max's default changed from 1e16 to None, because
    # its own inline copy of the capping logic didn't handle None).
    M_min, M_max = sampler._mass_range(M_env, M_split)
    M_edges = np.logspace(np.log10(M_min), np.log10(M_max), 11)

    assert M_edges[-1] < M_split


def test_m_split_none_reproduces_default_mass_bins(params):
    """Regression guard: M_split=None (the excursion-set-off default) must
    not change _mass_bins' behavior at all."""
    sampler = CHMFSampler(params, chmf=CHMF(params))
    M_env = sampler.chmf.rho_m * (params.Lbox_hunits / N) ** 3
    a, dln_a = sampler._mass_bins(M_env, n_mass_bins=10)
    b, dln_b = sampler._mass_bins(M_env, n_mass_bins=10, M_split=None)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(dln_a, dln_b)


# ── LPTHaloLoader end-to-end wiring ───────────────────────────────────────────

def test_loader_off_is_bitwise_identical_to_legacy_call(params):
    """excursion_set_method='off' (default) must reproduce sampler.sample()
    called without any of the new kwargs, exactly. Isolated from the
    (separate) apply_eulerian_displacement axis, which defaults to True and
    would otherwise diverge from this undisplaced direct call."""
    params.solver.redshifts = np.array([Z_LOW])
    params.halo_sim.apply_eulerian_displacement = False
    loader = LPTHaloLoader(params, n_mass_bins=10)
    assert loader._excursion_finder is None

    cat_via_loader = loader.load_halo_catalog(0)

    delta = loader.lpt_solver.get_linear_density(Z_LOW)
    cat_direct = loader.sampler.sample(
        delta_field=delta, z=Z_LOW, n_mass_bins=10, seed=loader._base_seed ^ 0,
    )
    np.testing.assert_array_equal(cat_via_loader.masses, cat_direct.masses)
    np.testing.assert_array_equal(cat_via_loader.positions, cat_direct.positions)


def test_loader_exact_extends_mass_range_beyond_m_env(params):
    params.solver.redshifts = np.array([Z_LOW])
    params.halo_sim.excursion_set_method = 'exact'
    loader = LPTHaloLoader(params, n_mass_bins=10)
    assert loader._excursion_finder is not None

    cat = loader.load_halo_catalog(0)
    assert cat.masses.max() > loader.M_split


def test_loader_exact_stochastic_and_excursion_masses_are_disjoint_at_m_split(params):
    """The two tiers' mass ranges must not overlap: every stochastically
    sampled halo's mass must be strictly below M_split, and the
    excursion-set tier's smallest halo mass must be exactly M_split (its
    own walk's own floor) -- a boundary-overlap regression guard."""
    params.solver.redshifts = np.array([Z_LOW])
    params.halo_sim.excursion_set_method = 'exact'
    loader = LPTHaloLoader(params, n_mass_bins=10)

    delta = loader.lpt_solver.get_linear_density(Z_LOW)
    det_catalog, det_mass_fraction = loader._run_excursion_set(delta, Z_LOW)
    stoch_catalog = loader.sampler.sample(
        delta_field=delta, z=Z_LOW, n_mass_bins=10, seed=loader._base_seed ^ 0,
        M_split=loader.M_split, deterministic_mass_fraction=det_mass_fraction,
    )

    if stoch_catalog.masses.size:
        assert stoch_catalog.masses.max() < loader.M_split
    if det_catalog is not None and det_catalog.masses.size:
        assert det_catalog.masses.min() >= loader.M_split


def test_loader_exact_reproducible(params):
    params.solver.redshifts = np.array([Z_LOW])
    params.halo_sim.excursion_set_method = 'exact'
    loader = LPTHaloLoader(params, n_mass_bins=10)

    cat1 = loader.load_halo_catalog(0)
    cat2 = loader.load_halo_catalog(0)
    np.testing.assert_array_equal(cat1.masses, cat2.masses)
    np.testing.assert_array_equal(cat1.positions, cat2.positions)


def test_loader_soft_raises_not_usable(params):
    params.halo_sim.excursion_set_method = 'soft'
    with pytest.raises(ValueError, match="not usable via LPTHaloLoader"):
        LPTHaloLoader(params)


def test_loader_unknown_excursion_set_method_raises(params):
    params.halo_sim.excursion_set_method = 'bogus'
    with pytest.raises(ValueError, match="Unknown excursion_set_method"):
        LPTHaloLoader(params)


# N, L for the two M_split-below-M_env tests below: with field_oversample=3
# and M_split=M_env/5, ExcursionSetFinder walks the (oversample-refined) fine
# field all the way down to a small M_split, and the number of accepted
# patches its per-patch Python loop must process grows steeply with grid
# resolution (measured: N=16/fineN=48 -> 663 patches/~1s; N=32/fineN=96 ->
# 5130 patches/~48s) -- the module's own file-level N=64 (fineN=192) blows
# this up past 30 minutes. N_SMALL/L_SMALL keep the same cell size (and thus
# the same resolving power for finding real structure) as N, L above while
# shrinking the total box volume, so the test still meaningfully exercises
# the M_split < M_env-with-oversampling code path without the runtime blowup.
N_SMALL, L_SMALL = 16, 50.0


def _small_oversample_params():
    p = Parameters()
    p.simulation.Ncell = N_SMALL
    p.simulation.Lbox = L_SMALL
    p.simulation.use_hunits = True
    return p


def test_loader_m_split_below_m_env_requires_field_oversample():
    """M_split < M_env with field_oversample == 1 (the default) must raise
    at the point of use, since the coarse grid alone cannot resolve
    sub-cell mass scales."""
    params = _small_oversample_params()
    params.solver.redshifts = np.array([Z_LOW])
    params.halo_sim.excursion_set_method = 'exact'
    M_env = CHMF(params).rho_m * (params.Lbox_hunits / N_SMALL) ** 3
    params.halo_sim.M_split = M_env / 5.0
    loader = LPTHaloLoader(params, n_mass_bins=10)
    with pytest.raises(ValueError, match="own cell size"):
        loader.load_halo_catalog(0)


def test_loader_m_split_below_m_env_works_with_field_oversample():
    params = _small_oversample_params()
    params.solver.redshifts = np.array([Z_LOW])
    params.halo_sim.excursion_set_method = 'exact'
    params.halo_sim.field_oversample = 3
    M_env = CHMF(params).rho_m * (params.Lbox_hunits / N_SMALL) ** 3
    params.halo_sim.M_split = M_env / 5.0
    loader = LPTHaloLoader(params, n_mass_bins=10)
    cat = loader.load_halo_catalog(0)
    assert cat.masses.max() > params.halo_sim.M_split
