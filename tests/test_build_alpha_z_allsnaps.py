"""Tests for ``fst_stochastic/build_alpha_z_allsnaps.py`` (production plan 2026-09-17, Phase 1).

The builder measures alpha(z) at EVERY THESAN snapshot by selecting central subhalos
straight from the merger-tree cache, bypassing the group catalogs that exist for only 48
of the 81 snapshots.  Everything that decides a number is exercised here against a
synthetic tree whose mass histories are exactly exponential, so the production fit must
recover each alpha to machine precision and any deviation is a selection/indexing bug:

* the satellite subhalo never enters (it would otherwise win its FoF row's alpha --
  ``fst_stochastic/docs/central_satellite.md``);
* a full branch yields its hand-computed alpha;
* a short branch with no surviving descendant falls through to ``alpha_fallback``, which
  is computed over ALL centrals *before* the mass cut, exactly as
  ``MergerTreeLoader.load_halo_catalog`` step 2 does it;
* a short branch WITH a descendant inherits instead, and is counted as inherited;
* the D3 (bin-centre) column differs from the raw column by the expected amount;
* the DM-mass cut drops a central below ``M_cut``;
* the overlap check fails on an alpha offset above 0.005 at N_cat >= 1000 and passes
  below it, in either the binned or the raw column.

The fixture follows ``tests/test_merger_tree_alpha_indexing.py``; ``make_parameters``
follows ``tests/test_four_case_support.py``.
"""
import importlib
import sys
import types

sys.modules.setdefault("MAS_library", types.SimpleNamespace(MASL=None))

from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent
           / "fst_stochastic" / "build_alpha_z_allsnaps.py")
if not _SCRIPT.exists():   # pragma: no cover - fst_stochastic is the local HPC checkout
    pytest.skip("fst_stochastic scripts not present (local HPC scripts)",
                allow_module_level=True)
sys.path.insert(0, str(_SCRIPT.parent))
builder = importlib.import_module("build_alpha_z_allsnaps")

from beorn.load_input_data.merger_tree_base import MergerTreeLoader
from beorn.structs.parameters import Parameters


# ---------------------------------------------------------------------------
# Synthetic tree
# ---------------------------------------------------------------------------

# Descending and deliberately non-uniform, as in THESAN (a uniform grid hides shift bugs).
ALL_REDSHIFTS = np.array([14.0, 13.1, 12.3, 11.6, 11.0, 10.5, 10.1, 9.8])
N_SNAPSHOTS = ALL_REDSHIFTS.size
LOOKBACK = 4
SNAP = 5                      # the snapshot every test builds a row for

# halo:          0       1       2 (sat) 3 (light) 4 (fb)  5 (inherits)
TRUE_ALPHAS = np.array([0.33,   0.47,   1.20,   1.20,     0.33,   0.47])
# Mass at SNAP, in tree-cache units (1e10 Msun/h).  Halo 3 is the only central below the
# cut; the satellite is heavy so that including it would be obvious.
M_AT_SNAP = np.array([1.0,    2.0,    5.0,    0.01,     3.0,    4.0])
IS_CENTRAL = np.array([True,  True,   False,  True,     True,   True])
N_HALOS = TRUE_ALPHAS.size

# Production values, so the binning under test is the one the run uses.
ALPHA_EDGES = np.linspace(0.0, 1.5, 16)          # run_config.TREE_ALPHA_GRID
M_CUT_CACHE = 1e9 * (0.3089 - 0.0486) / 0.3089 / 1e10    # 0.0842668, see the script


def entry_index(snapshot: int, halo: int) -> int:
    """Flat cache index of halo ``halo`` at snapshot ``snapshot``."""
    return snapshot * N_HALOS + halo


class SyntheticTreeLoader(MergerTreeLoader):
    """Merger-tree loader over the synthetic tree, indexed by RAW snapshot number.

    ``redshifts`` and ``snapshot_numbers`` cover every snapshot, which is exactly what
    ``build_alpha_z_allsnaps.AllSnapshotThesanLoader`` arranges for the real loader, so
    ``redshift_index`` is the raw snapshot number here too.

    ``M(z) = M_at_snap * exp(-alpha * (z - z_SNAP))`` is exactly exponential, and
    ``vectorized_alpha_fit`` normalises by the branch root's mass, so every full branch
    recovers its alpha exactly regardless of the prefactor.

    Args:
        parameters (Parameters): BEoRN parameter container.
        terminate: list of ``(halo, snapshot)`` pairs; each removes that entry's
            progenitor link, ending the branch going backward there.
    """

    def __init__(self, parameters, terminate=()):
        super().__init__(parameters)
        self._terminations = list(terminate)
        # Mirrors AllSnapshotThesanLoader: record the pre-inheritance fit mask so the
        # builder can split fitted / inherited / fallback.
        self.last_seed_finite = None

    @property
    def redshifts(self):
        return ALL_REDSHIFTS

    @property
    def snapshot_numbers(self):
        return np.arange(N_SNAPSHOTS)

    @property
    def all_snapshot_redshifts(self):
        return ALL_REDSHIFTS

    def _inherit_alphas_from_descendants(self, snap_now, current_halo_ids, seed_alphas,
                                         *args, **kwargs):
        self.last_seed_finite = np.isfinite(np.asarray(seed_alphas, dtype=np.float64))
        return super()._inherit_alphas_from_descendants(
            snap_now, current_halo_ids, seed_alphas, *args, **kwargs)

    def load_tree_cache(self):
        n = N_SNAPSHOTS * N_HALOS
        halo_ids = np.tile(np.arange(N_HALOS), N_SNAPSHOTS)
        snap_num = np.repeat(np.arange(N_SNAPSHOTS), N_HALOS)
        dz = ALL_REDSHIFTS[snap_num] - ALL_REDSHIFTS[SNAP]
        mass = M_AT_SNAP[halo_ids] * np.exp(-TRUE_ALPHAS[halo_ids] * dz)

        main_progenitor = np.full(n, -1, dtype=np.int64)
        for snapshot in range(1, N_SNAPSHOTS):
            for halo in range(N_HALOS):
                main_progenitor[entry_index(snapshot, halo)] = entry_index(snapshot - 1, halo)
        for halo, snapshot in self._terminations:
            main_progenitor[entry_index(snapshot, halo)] = -1

        is_central = np.tile(IS_CENTRAL, N_SNAPSHOTS)
        return halo_ids, snap_num, mass, main_progenitor, is_central

    def get_halo_information_from_catalog(self, redshift_index):
        raise AssertionError("the all-snapshot builder must not read a group catalog")

    def load_density_field(self, redshift_index):
        raise AssertionError("the all-snapshot builder must not read the density field")


def make_parameters():
    parameters = Parameters()
    parameters.source.alpha_constant = None
    parameters.source.alpha_constant_z = None
    parameters.source.mass_accretion_lookback = LOOKBACK
    parameters.source.alpha_fallback = "mean"
    parameters.source.halo_mass_min = 1e9
    parameters.solver.halo_mass_accretion_alpha = ALPHA_EDGES.copy()
    parameters.cosmology.Om = 0.3089
    parameters.cosmology.Ob = 0.0486
    parameters.cosmology.Ol = 0.6911
    parameters.cosmology.h0 = 0.6774
    return parameters


# Halo 4 is short-branch at SNAP (progenitor cut inside the window) AND has no surviving
# descendant that reaches SNAP (its later links are cut too) -> alpha_fallback.
# Halo 5 is short-branch as well but keeps its descendants -> inherits.
TERMINATE_FALLBACK_AND_INHERIT = [(4, SNAP - 1), (4, SNAP + 1), (4, SNAP + 2),
                                  (5, SNAP - 1)]

# alpha_fallback = mean over the CENTRALS with a finite alpha, before the mass cut:
# halos 0, 1, 3 (light but central) and 5 (inherited 0.47).
EXPECTED_ALPHA_FB = (0.33 + 0.47 + 1.20 + 0.47) / 4.0     # 0.6175


def build_row(terminate=TERMINATE_FALLBACK_AND_INHERIT, m_cut=M_CUT_CACHE):
    parameters = make_parameters()
    loader = SyntheticTreeLoader(parameters, terminate=terminate)
    cache = loader.load_tree_cache()
    return builder.build_snapshot_row(loader, SNAP, cache, ALPHA_EDGES, m_cut, parameters)


# ---------------------------------------------------------------------------
# Selection: satellites, the mass cut, and the alignment with the fitted alphas
# ---------------------------------------------------------------------------

def test_satellite_is_never_selected():
    loader = SyntheticTreeLoader(make_parameters())
    _, snap_num, mass, _, is_central = loader.load_tree_cache()

    rows, keep = builder.select_central_rows(snap_num, mass, is_central, SNAP, M_CUT_CACHE)

    halo_of_row = rows % N_HALOS
    assert 2 not in halo_of_row, "the satellite must not reach the alpha population"
    assert set(halo_of_row.tolist()) == {0, 1, 3, 4, 5}


def test_mass_cut_drops_the_light_central():
    loader = SyntheticTreeLoader(make_parameters())
    _, snap_num, mass, _, is_central = loader.load_tree_cache()

    rows, keep = builder.select_central_rows(snap_num, mass, is_central, SNAP, M_CUT_CACHE)

    assert sorted((rows[keep] % N_HALOS).tolist()) == [0, 1, 4, 5]
    dropped = rows[~keep] % N_HALOS
    assert dropped.tolist() == [3], "only the sub-M_cut central may be dropped"


def test_selection_lines_up_with_the_fitted_alphas():
    """The mass column and the alpha column must be the same halos, in the same order."""
    loader = SyntheticTreeLoader(make_parameters())
    halo_ids, snap_num, mass, _, is_central = loader.load_tree_cache()
    ids, _ = loader.get_halo_accretion_rate_from_tree(SNAP)

    rows, _ = builder.select_central_rows(snap_num, mass, is_central, SNAP, M_CUT_CACHE)

    np.testing.assert_array_equal(halo_ids[rows], ids)


def test_default_m_cut_is_the_dm_fraction_of_the_painted_cut():
    parameters = make_parameters()

    m_cut = builder.default_m_cut_msun_h(parameters)

    assert m_cut == pytest.approx(1e9 * (0.3089 - 0.0486) / 0.3089)
    assert m_cut / builder.MSUN_PER_CACHE_UNIT == pytest.approx(0.0842668, abs=1e-6)


# ---------------------------------------------------------------------------
# One snapshot end to end: fit, inheritance, fallback, binning
# ---------------------------------------------------------------------------

def test_full_branches_recover_the_hand_alpha():
    row, extra = build_row()

    np.testing.assert_allclose(sorted(extra["alphas_selected"])[:2], [0.33, 0.47],
                               rtol=1e-10)


def test_fallback_fills_the_descendantless_short_branch():
    row, extra = build_row()

    # alpha_fb is the mean over all CENTRALS with a finite alpha, before the mass cut:
    # the light central (1.20) pulls it up, which is production behaviour.
    assert extra["alpha_fb"] == pytest.approx(EXPECTED_ALPHA_FB, rel=1e-10)
    assert sorted(extra["alphas_selected"])[-1] == pytest.approx(EXPECTED_ALPHA_FB,
                                                                rel=1e-10)
    n_fitted, n_inherited, n_fallback = int(row[4]), int(row[5]), int(row[6])
    assert (n_fitted, n_inherited, n_fallback) == (2, 1, 1)
    assert int(row[3]) == 4


def test_short_branch_with_a_descendant_inherits_instead_of_falling_back():
    _, extra = build_row()

    # Halo 5's descendant branch fits 0.47 exactly, so two selected halos carry 0.47.
    assert np.sum(np.isclose(extra["alphas_selected"], 0.47)) == 2


def test_bin_centre_column_differs_from_the_raw_column_as_expected():
    row, extra = build_row()

    aN_bin, aN_raw = row[7], row[8]
    expected_raw = (0.33 + 0.47 + 0.47 + EXPECTED_ALPHA_FB) / 4.0
    # 0.33 -> [0.3, 0.4) -> 0.35; 0.47 -> [0.4, 0.5) -> 0.45; 0.6175 -> [0.6, 0.7) -> 0.65.
    expected_bin = (0.35 + 0.45 + 0.45 + 0.65) / 4.0
    assert aN_raw == pytest.approx(expected_raw, rel=1e-10)
    assert aN_bin == pytest.approx(expected_bin, rel=1e-10)
    assert aN_bin != pytest.approx(aN_raw, abs=1e-6)
    np.testing.assert_allclose(sorted(extra["alphas_binned"]), [0.35, 0.45, 0.45, 0.65],
                               rtol=1e-10)


def test_mass_weighted_moment_uses_the_selected_masses_only():
    row, _ = build_row()

    # aM over halos 0 (1.0, 0.35), 1 (2.0, 0.45), 4 (3.0, 0.65), 5 (4.0, 0.45).
    masses = np.array([1.0, 2.0, 3.0, 4.0])
    binned = np.array([0.35, 0.45, 0.65, 0.45])
    assert row[9] == pytest.approx(float(np.sum(binned * masses) / masses.sum()), rel=1e-10)


def test_a_light_central_cannot_change_the_mean_but_does_change_the_fallback():
    """Guards the ordering: the cut is applied AFTER alpha_fallback is derived."""
    row_cut, extra_cut = build_row()
    row_nocut, extra_nocut = build_row(m_cut=0.0)

    assert int(row_cut[3]) == 4 and int(row_nocut[3]) == 5
    assert extra_cut["alpha_fb"] == pytest.approx(extra_nocut["alpha_fb"], rel=1e-12)
    assert row_cut[7] != pytest.approx(row_nocut[7], abs=1e-6)


# ---------------------------------------------------------------------------
# The pure numeric helpers
# ---------------------------------------------------------------------------

def test_fill_and_clamp_matches_load_halo_catalog_step_5():
    alphas = np.array([np.nan, -0.5, 0.42, 1.49, np.inf])

    filled, n_fallback = builder.fill_and_clamp(alphas, 0.6, ALPHA_EDGES)

    # Step 4 replaces every NON-FINITE alpha (NaN and Inf alike) by the fallback; step 5
    # then clamps the finite ones into [edges[0], edges[-2]].
    assert n_fallback == 2
    np.testing.assert_allclose(filled, [0.6, ALPHA_EDGES[0], 0.42, ALPHA_EDGES[-2], 0.6])


def test_bin_to_centres_uses_the_same_assignment_as_a_histogram():
    rng = np.random.default_rng(0)
    alphas = rng.uniform(0.0, 1.5, 500)

    centres = builder.bin_to_centres(alphas, ALPHA_EDGES)

    counts, _ = np.histogram(alphas, bins=ALPHA_EDGES)
    from beorn.structs.helpers import bin_centers
    expected_mean = np.sum(counts * bin_centers(ALPHA_EDGES)) / counts.sum()
    assert centres.mean() == pytest.approx(expected_mean, rel=1e-12)


def test_cosmic_time_decreases_with_redshift():
    parameters = make_parameters()

    t_high = builder.cosmic_time_Myr(14.0, parameters)
    t_low = builder.cosmic_time_Myr(5.5, parameters)

    assert 200.0 < t_high < 400.0, "t(z=14) should be a few hundred Myr"
    assert t_low > t_high
    np.testing.assert_allclose(builder.cosmic_time_Myr([14.0, 5.5], parameters),
                               [t_high, t_low], rtol=1e-12)


def test_alpha_det_candidates_over_the_d2_range():
    snaps = np.arange(0, 81)
    alphas = np.where(snaps == 0, 10.0, np.where(snaps == 80, -10.0, 0.5))
    t = np.linspace(0.0, 800.0, 81)

    cand = builder.alpha_det_candidates(snaps, alphas, t)

    assert cand["n_range"] == 79
    assert cand["plain_range"] == pytest.approx(0.5)
    assert cand["trapezoid_range"] == pytest.approx(0.5)
    # plain_all sees the two outlier rows as well, so it is NOT the D2 mean.
    assert cand["plain_all"] == pytest.approx((79 * 0.5 + 10.0 - 10.0) / 81)


def test_parse_snapshot_spec():
    np.testing.assert_array_equal(builder.parse_snapshot_spec(None, 5), np.arange(5))
    np.testing.assert_array_equal(builder.parse_snapshot_spec("3,0-1", 5), [0, 1, 3])
    with pytest.raises(ValueError):
        builder.parse_snapshot_spec("0-5", 5)


def test_output_paths_are_the_plan_names_and_never_the_protected_files():
    paths = builder.output_paths(Path("/tmp/x"), "allsnaps")

    assert paths["table"].name == "alpha_z_number_allsnaps.txt"
    assert paths["moments"].name == "alpha_z_allsnaps_moments.txt"
    assert paths["check"].name == "alpha_z_allsnaps_overlap_check.txt"
    for name in builder.PROTECTED_FILES:
        assert name not in {p.name for p in paths.values()}
    with pytest.raises(ValueError):
        builder.output_paths(Path("/tmp/x"), "")


# ---------------------------------------------------------------------------
# The Phase-1 gate
# ---------------------------------------------------------------------------

CAT_Z = np.array([6.0, 7.0, 8.0])
CAT_N = np.array([5000.0, 5000.0, 50.0])
CAT_BIN = np.array([0.500, 0.480, 0.400])
CAT_RAW = np.array([0.495, 0.475, 0.395])


def _check(tree_rows):
    return builder.overlap_check_rows(tree_rows, CAT_Z, CAT_N, CAT_BIN, CAT_RAW)


def test_overlap_check_passes_within_tolerance():
    records, passed = _check([(6.0, 4900, 0.504, 0.499), (7.0, 4900, 0.476, 0.471)])

    assert passed
    assert [r["gated"] for r in records] == [True, True]
    assert records[0]["ratio"] == pytest.approx(0.98)


def test_overlap_check_fails_on_a_binned_offset_above_the_tolerance():
    records, passed = _check([(6.0, 4900, 0.506, 0.495)])

    assert not passed
    assert records[0]["d_bin"] == pytest.approx(0.006)
    assert records[0]["ok"] is False


def test_overlap_check_fails_on_a_raw_offset_above_the_tolerance():
    """Both columns are gated (plan section 3 states the criterion for both)."""
    _, passed = _check([(6.0, 4900, 0.501, 0.501)])

    assert not passed


def test_sparse_rows_are_reported_but_not_gated():
    records, passed = _check([(8.0, 40, 0.500, 0.500)])

    assert passed
    assert records[0]["gated"] is False
    assert records[0]["d_bin"] == pytest.approx(0.100)


def test_rows_without_a_catalog_counterpart_are_skipped():
    records, passed = _check([(9.5, 100, 0.9, 0.9), (6.0, 4900, 0.501, 0.496)])

    assert passed
    assert [r["z"] for r in records] == [6.0]
