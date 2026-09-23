"""Review tests for the mass-dependent f_st scatter (sigma0 / sigma1 / mpiv).

Covers the three review questions against the shipped code, plus strict-xfail tests that
pin down latent failure modes found in the review (they turn into XPASS -> failure once the
behaviour is fixed, so the marker must then be removed). Companion script:
fst_stochastic/review/check_mass_dep_scatter.py.
"""
import sys
import types
from pathlib import Path
from types import SimpleNamespace

if "MAS_library" not in sys.modules:
    sys.modules["MAS_library"] = types.SimpleNamespace(MASL=None)

import numpy as np
import pytest

from beorn.painting.coordinator import PaintingCoordinator, sigma_dex_for_mass
from beorn.precomputation.solver import RadiationProfileFstSolver
from beorn.structs.parameters import Parameters
from beorn.structs.radiation_profiles import RadiationProfilesFStarGrid

FST_DIR = Path(__file__).resolve().parents[1] / "fst_stochastic"
YAML_DEP = FST_DIR / "parameters_mass_dep_test.yaml"
YAML_INDEP = FST_DIR / "parameters_mass_indep_test.yaml"
LN10 = np.log(10.0)

# fst_stochastic/ is a separate (nested, gitignored) repository; skip what reads its yamls.
needs_yaml = pytest.mark.skipif(
    not (YAML_DEP.exists() and YAML_INDEP.exists()),
    reason="fst_stochastic/ test yamls not present",
)


class DummyHaloCatalog:
    def __init__(self, masses, alpha_vals, selected=None):
        self._all_masses = np.asarray(masses, dtype=float)
        self._all_alpha_vals = np.asarray(alpha_vals, dtype=float)
        self.selected = np.arange(self._all_masses.size) if selected is None else np.asarray(selected)

    @property
    def masses(self):
        return self._all_masses[self.selected]

    @property
    def alphas(self):
        return self._all_alpha_vals[self.selected]

    @property
    def size(self):
        return self.selected.size

    def get_halo_indices(self, alpha_range, mass_range):
        m, a = self.masses, self.alphas
        mask = (a >= alpha_range[0]) & (a < alpha_range[1]) & (m >= mass_range[0]) & (m < mass_range[1])
        return np.where(mask)[0]

    def at_indices(self, idx):
        return DummyHaloCatalog(self._all_masses, self._all_alpha_vals, self.selected[np.asarray(idx)])


class DummyLoader:
    def __init__(self, halo_catalog=None):
        self._halo_catalog = halo_catalog
        self.redshifts = np.array([12.0, 10.0])

    def load_halo_catalog(self, z_index):
        return self._halo_catalog

    def load_density_field(self, z_index):
        return np.zeros((4, 4, 4), dtype=float)


def coordinator_for(params, halo_catalog=None):
    return PaintingCoordinator(params, DummyLoader(halo_catalog), output_handler=None, cache_handler=None)


def log_uniform_masses(n, lo=9.0, hi=12.0, seed=0):
    return 10 ** np.random.default_rng(seed).uniform(lo, hi, n)


def f_st_grid_for(params):
    return RadiationProfileFstSolver._build_f_st_grid(SimpleNamespace(parameters=params))


# ---------------------------------------------------------------------------
# Q3: f_st_paint_sigma1 = 0 must reproduce the mass-independent sampler
# ---------------------------------------------------------------------------

def _mass_dep_and_constant_params(sigma0=0.222658):
    dep = Parameters.from_yaml(YAML_DEP)
    dep.source.f_st_paint_sigma1 = 0.0
    const = Parameters.from_yaml(YAML_DEP)
    const.source.f_st_paint_sigma0 = None
    const.source.f_st_paint_sigma1 = 0.0  # sigma1 without sigma0 is rejected
    const.source.f_st_paint_sigma = float(sigma0) * LN10  # dex -> natural-log width
    dep.source.f_st_paint_sigma0 = sigma0
    return dep, const


@needs_yaml
def test_sigma1_zero_sampler_is_bitwise_identical_to_constant_sigma():
    """Array-valued sigma (mass-dependent branch) and scalar sigma (constant branch) must
    consume the RNG stream identically, so sigma1=0 reproduces the constant path exactly."""
    dep, const = _mass_dep_and_constant_params()
    masses = log_uniform_masses(200_000)

    s_dep = coordinator_for(dep)._sample_f_st_for_halos(masses, np.random.default_rng(12345 + 68))
    s_const = coordinator_for(const)._sample_f_st_for_halos(masses, np.random.default_rng(12345 + 68))

    np.testing.assert_array_equal(s_dep, s_const)

    # Control: the same comparison must detect a non-zero slope.
    dep.source.f_st_paint_sigma1 = -0.040749
    s_slope = coordinator_for(dep)._sample_f_st_for_halos(masses, np.random.default_rng(12345 + 68))
    assert not np.array_equal(s_slope, s_const)


def test_sigma1_zero_paint_single_fstar_uses_same_profile_slices_as_constant_sigma(monkeypatch):
    """End to end through paint_single_fstar: every halo gets the same f_st grid slice,
    in the same (alpha, mass, f_st) painting group, under sigma1=0 and under constant sigma."""
    rng = np.random.default_rng(3)
    n = 400
    masses = 10 ** rng.uniform(8.1, 9.95, n)
    alphas = rng.uniform(0.5, 0.99, n)
    halo_catalog = DummyHaloCatalog(masses, alphas)

    def make_params(sigma0, sigma1, sigma):
        params = SimpleNamespace(
            simulation=SimpleNamespace(
                Ncell=4, Lbox=10.0, cores=1, fft_backend="numpy",
                store_grids=["Grid_xHII", "Grid_Temp", "Grid_xal"],
                compute_s_alpha_fluctuations=False,
                halo_mass_bins=np.array([1.0e8, 1.0e9, 1.0e10]),
                halo_mass_accretion_alpha=np.array([0.5, 1.0]),
            ),
            source=SimpleNamespace(
                min_xHII_value=0.0, f_st=0.018814, f_st_paint_distribution="lognormal",
                f_st_paint_sigma=sigma, f_st_paint_sigma0=sigma0, f_st_paint_sigma1=sigma1,
                f_st_paint_sigma_mpiv=1e11, f_st_paint_min=5e-4, f_st_paint_max=0.55,
                f_st_paint_seed=12345,
            ),
        )
        return params

    f_st_grid = np.logspace(np.log10(5e-4), np.log10(0.55), 13)
    f_n = f_st_grid.size
    profiles = RadiationProfilesFStarGrid(
        parameters=SimpleNamespace(),
        z_history=np.array([12.0, 10.0]),
        halo_mass_bins=np.array([[[1.0e8, 1.0e8]], [[1.0e9, 1.0e9]], [[1.0e10, 1.0e10]]]),
        f_st_grid=f_st_grid,
        rho_xray=np.ones((3, 2, 1, f_n, 2)),
        rho_heat=np.ones((3, 2, 1, f_n, 2)),
        rho_alpha=np.ones((5, 2, 1, f_n, 2)),
        R_bubble=np.ones((2, 1, f_n, 2)),
        r_lyal=np.logspace(-5, 2, 5),
        r_grid_cell=np.logspace(-2, 1, 3),
    )

    records = []

    def fake_profiles_of_halo_bin(self, z_index, alpha_index, mass_index, f_st_index):
        tag = float(1000 * mass_index + 100 * alpha_index + f_st_index)
        return np.array(tag), np.array([tag]), np.array([tag])

    def fake_paint_single_mass_bin(self, halo_catalog, z, radial_grid, r_lyal, profiles_of_bin, **kwargs):
        records[-1].append((int(np.asarray(profiles_of_bin[0]).item()), tuple(halo_catalog.selected.tolist())))
        return (None, None, None)

    monkeypatch.setattr(RadiationProfilesFStarGrid, "profiles_of_halo_bin", fake_profiles_of_halo_bin)
    monkeypatch.setattr(PaintingCoordinator, "paint_single_mass_bin", fake_paint_single_mass_bin)
    monkeypatch.setattr("beorn.painting.coordinator.spreading_excess_fast", lambda parameters, grid: grid)
    monkeypatch.setattr("beorn.painting.coordinator.T_adiab_fluctu", lambda z, parameters, delta_b: np.zeros_like(delta_b))
    monkeypatch.setattr("beorn.painting.coordinator.S_alpha", lambda z, temp, neutral_frac: 1.0)

    sigma0 = 0.222658
    for params in (make_params(sigma0, 0.0, 0.5), make_params(None, 0.0, sigma0 * LN10),
                   make_params(sigma0, -0.040749, 0.5)):
        records.append([])
        coordinator_for(params, halo_catalog).paint_single_fstar(0, profiles)

    dep_zero, const, dep_slope = records
    assert sum(len(sel) for _, sel in const) == halo_catalog.size
    assert len({tag % 100 for tag, _ in const}) >= 3  # several f_st slices actually used
    assert dep_zero == const
    assert dep_slope != const  # control: a real slope changes the assignment


@needs_yaml
def test_yaml_pair_rounding_does_not_change_painted_slices():
    """parameters_mass_indep_test.yaml writes sigma0*ln10 rounded to 0.512689; that rounding
    must not move any halo to another f_st grid point versus sigma1=0 on the mass-dep yaml."""
    dep = Parameters.from_yaml(YAML_DEP)
    dep.source.f_st_paint_sigma1 = 0.0
    indep = Parameters.from_yaml(YAML_INDEP)
    assert indep.source.f_st_paint_sigma0 is None

    masses = log_uniform_masses(300_000)
    grid = f_st_grid_for(dep)
    np.testing.assert_array_equal(grid, f_st_grid_for(indep))

    c_dep, c_ind = coordinator_for(dep), coordinator_for(indep)
    s_dep = c_dep._sample_f_st_for_halos(masses, np.random.default_rng(7))
    s_ind = c_ind._sample_f_st_for_halos(masses, np.random.default_rng(7))

    assert np.max(np.abs(s_ind / s_dep - 1.0)) < 1e-6
    np.testing.assert_array_equal(c_dep._nearest_f_st_indices(s_dep, grid),
                                  c_ind._nearest_f_st_indices(s_ind, grid))


# ---------------------------------------------------------------------------
# Q1: f_st grid coverage for parameters_mass_dep_test.yaml
# ---------------------------------------------------------------------------

@needs_yaml
def test_mass_dep_yaml_f_st_grid_covers_the_sampled_range():
    params = Parameters.from_yaml(YAML_DEP)
    s = params.source
    grid = f_st_grid_for(params)

    assert grid[0] <= s.f_st_paint_min and s.f_st_paint_max <= grid[-1]
    assert grid[0] < s.f_st < grid[-1]

    # sigma1 < 0: the widest distribution is at the lowest painted mass.
    sig_widest = sigma_dex_for_mass(np.array([s.halo_mass_min]), s.f_st_paint_sigma0,
                                    s.f_st_paint_sigma1, s.f_st_paint_sigma_mpiv)[0]
    assert s.f_st_paint_sigma1 < 0
    n_sig_lo = (np.log10(s.f_st) - np.log10(s.f_st_paint_min)) / sig_widest
    n_sig_hi = (np.log10(s.f_st_paint_max) - np.log10(s.f_st)) / sig_widest
    assert n_sig_lo > 4.5 and n_sig_hi > 4.5


@needs_yaml
def test_log_grid_snapping_keeps_log_mean_but_inflates_linear_mean():
    """Nearest-in-log snapping is unbiased in log10 f_st, but the painted LINEAR mean f_st
    (photon budget) is inflated by sinh(u0)/u0, u0 = step*ln10/2 (Sheppard/Widrow)."""
    params = Parameters.from_yaml(YAML_DEP)
    coord = coordinator_for(params)
    grid = f_st_grid_for(params)
    step = np.diff(np.log10(grid)).mean()

    masses = np.full(400_000, 1e10)
    sampled = coord._sample_f_st_for_halos(masses, np.random.default_rng(11))
    snapped = grid[coord._nearest_f_st_indices(sampled, grid)]

    assert abs(np.log10(snapped).mean() - np.log10(sampled).mean()) < 2e-3
    u0 = step * LN10 / 2
    np.testing.assert_allclose(snapped.mean() / sampled.mean(), np.sinh(u0) / u0, rtol=3e-3)
    assert snapped.mean() / sampled.mean() > 1.01  # ~+1.4% for the 13-point [5e-4, 0.55] grid


# ---------------------------------------------------------------------------
# Latent failure modes (strict xfail: documents current behaviour)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "Parameters defines f_st_paint_min/max = 0.01/0.2, so the getattr fallback to "
    "f_st_grid_min/max in _sample_f_st_for_halos is dead code; omitting the paint bounds "
    "silently clips at 0.01/0.2 instead of the grid bounds the docstring promises."))
def test_omitted_paint_bounds_fall_back_to_grid_bounds():
    params = Parameters()
    params.source.f_st = 0.018814
    params.source.f_st_grid_min = 5e-4
    params.source.f_st_grid_max = 0.55
    params.source.f_st_paint_sigma0 = 0.222658
    params.source.f_st_paint_sigma1 = -0.040749

    sampled = coordinator_for(params)._sample_f_st_for_halos(np.full(50_000, 1e9), np.random.default_rng(1))
    assert sampled.min() < 0.01


def test_paint_single_fstar_rejects_paint_range_outside_f_st_grid(monkeypatch):
    halo_catalog = DummyHaloCatalog(np.array([2e8, 3e8, 2e9, 3e9]), np.full(4, 0.7))
    params = SimpleNamespace(
        simulation=SimpleNamespace(
            Ncell=4, Lbox=10.0, cores=1, fft_backend="numpy",
            store_grids=["Grid_xHII", "Grid_Temp", "Grid_xal"], compute_s_alpha_fluctuations=False,
            halo_mass_bins=np.array([1.0e8, 1.0e9, 1.0e10]), halo_mass_accretion_alpha=np.array([0.5, 1.0]),
        ),
        source=SimpleNamespace(
            min_xHII_value=0.0, f_st=0.1, f_st_paint_distribution="lognormal", f_st_paint_sigma=0.4,
            f_st_paint_min=1e-3, f_st_paint_max=0.9, f_st_paint_seed=1,  # grid below is [0.01, 0.2]
        ),
    )
    profiles = RadiationProfilesFStarGrid(
        parameters=SimpleNamespace(), z_history=np.array([12.0, 10.0]),
        halo_mass_bins=np.array([[[1.0e8, 1.0e8]], [[1.0e9, 1.0e9]], [[1.0e10, 1.0e10]]]),
        f_st_grid=np.array([0.01, 0.05, 0.1, 0.2]),
        rho_xray=np.ones((3, 2, 1, 4, 2)), rho_heat=np.ones((3, 2, 1, 4, 2)),
        rho_alpha=np.ones((5, 2, 1, 4, 2)), R_bubble=np.ones((2, 1, 4, 2)),
        r_lyal=np.logspace(-5, 2, 5), r_grid_cell=np.logspace(-2, 1, 3),
    )
    monkeypatch.setattr(PaintingCoordinator, "paint_single_mass_bin", lambda self, **kw: (None, None, None))
    monkeypatch.setattr("beorn.painting.coordinator.spreading_excess_fast", lambda parameters, grid: grid)
    monkeypatch.setattr("beorn.painting.coordinator.T_adiab_fluctu", lambda z, parameters, delta_b: np.zeros_like(delta_b))
    monkeypatch.setattr("beorn.painting.coordinator.S_alpha", lambda z, temp, neutral_frac: 1.0)

    with pytest.raises(RuntimeError, match="outside the precomputed f_st grid"):
        coordinator_for(params, halo_catalog).paint_single_fstar(0, profiles)


@pytest.mark.parametrize("distribution", ["normal", "uniform"])
def test_mass_dependent_scatter_with_non_lognormal_distribution_raises(distribution):
    params = Parameters()
    params.source.f_st_paint_distribution = distribution
    params.source.f_st_paint_sigma0 = 0.2
    with pytest.raises(ValueError, match="only defined for the 'lognormal' distribution"):
        coordinator_for(params)._sample_f_st_for_halos(np.full(10, 1e10), np.random.default_rng(1))


def test_sigma1_without_sigma0_raises():
    params = Parameters()
    params.source.f_st_paint_sigma1 = -0.04
    with pytest.raises(ValueError, match="without f_st_paint_sigma0"):
        coordinator_for(params)._sample_f_st_for_halos(np.full(10, 1e10), np.random.default_rng(1))


@pytest.mark.parametrize("bad", [0.0, -1e10, np.nan, np.inf])
def test_zero_negative_or_nonfinite_masses_raise_before_sampling(bad):
    params = Parameters()
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="zero, negative or non-finite"):
        coordinator_for(params)._sample_f_st_for_halos(np.array([1e10, bad, 1e11]), rng)
    # Nothing was drawn: the stream is untouched for the next (valid) call.
    np.testing.assert_array_equal(rng.random(3), np.random.default_rng(1).random(3))


# ---------------------------------------------------------------------------
# run_config case 'tree_mdep_scatter' (fst_stochastic/run_config.py)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (FST_DIR / "run_config.py").exists(), reason="fst_stochastic/ not present")
def test_tree_mdep_scatter_case_shares_the_profile_cube_and_paints():
    sys.path.insert(0, str(FST_DIR))
    try:
        import run_config
    finally:
        sys.path.remove(str(FST_DIR))

    def case_params(case):
        params = run_config.load_parameters(FST_DIR / run_config.CASE_PARAMETER_FILES[case], Path("."),
                                            alpha_grid=run_config.KEEP_YAML_ALPHA)
        run_config.apply_case(params, case)
        return params

    mdep, const = case_params("tree_mdep_scatter"), case_params("tree_scatter")
    assert run_config.uses_fst_grid("tree_mdep_scatter")
    assert mdep.source.f_st == const.source.f_st == run_config.THESAN1_FST_MEDIAN
    assert mdep.source.f_st_paint_sigma0 == run_config.THESAN1_FST_SIGMA0_DEX
    assert mdep.source.f_st_paint_sigma1 == run_config.THESAN1_FST_SIGMA1_DEX
    assert const.source.f_st_paint_sigma0 is None and const.source.f_st_paint_sigma1 == 0.0
    np.testing.assert_array_equal(mdep.solver.halo_mass_accretion_alpha, const.solver.halo_mass_accretion_alpha)
    # Paint-only difference: same (mass, alpha, f_st, z) cube, different painted output.
    assert mdep.profiles_fstar_hash() == const.profiles_fstar_hash()
    assert mdep.beorn_hash() != const.beorn_hash()

    coord = coordinator_for(mdep)
    coord._raise_if_f_st_outside_profile_coverage(f_st_grid_for(mdep))
    masses = log_uniform_masses(100_000, lo=9.0, hi=12.0)
    sampled = coord._sample_f_st_for_halos(masses, np.random.default_rng(5))
    low, high = masses < 10 ** 9.5, masses > 10 ** 11.5
    assert np.log10(sampled[low]).std() > np.log10(sampled[high]).std()
