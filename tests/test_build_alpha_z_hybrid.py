"""Tests for ``fst_stochastic/build_alpha_z_hybrid.py`` (production plan 2026-09-17, Phase 1).

The hybrid table takes the catalog value verbatim where a catalog row exists and, where none
does, the tree-cache value minus the overlap offset interpolated in z.  What can go wrong is
orientation, not arithmetic: the all-snapshot table is in snapshot order (**descending** z)
while the catalog table is ascending, and ``np.interp`` returns silent nonsense for a
decreasing ``xp``.  The fixture below therefore keeps the production orientation -- tree rows
descending in z, catalog rows ascending -- and spaces the redshifts non-uniformly so that an
interior interpolation weight is never 1/2 (which would hide a swapped-neighbour bug).

Five raw snapshots, three overlap rows.  Two different overlap sets over the same tree arrays
cover an interior gap plus both end clamps:

* overlaps {0, 1, 3}: gap 2 interpolates, gap 4 clamps at the low-z end;
* overlaps {1, 3, 4}: gap 0 clamps at the high-z end, gap 2 interpolates.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent
           / "fst_stochastic" / "build_alpha_z_hybrid.py")
if not _SCRIPT.exists():   # pragma: no cover - fst_stochastic is the local HPC checkout
    pytest.skip("fst_stochastic scripts not present (local HPC scripts)",
                allow_module_level=True)
sys.path.insert(0, str(_SCRIPT.parent))
builder = importlib.import_module("build_alpha_z_hybrid")


# ---------------------------------------------------------------------------
# Synthetic tables
# ---------------------------------------------------------------------------

# Snapshot order, i.e. descending z, deliberately non-uniform (as in THESAN).
SNAP = np.array([0, 1, 2, 3, 4])
TREE_Z = np.array([12.0, 10.0, 9.0, 6.0, 5.5])
TREE_ALPHA = np.array([0.30, 0.40, 0.44, 0.50, 0.52])
TREE_N = np.array([10.0, 100.0, 1000.0, 5000.0, 9000.0])


def catalog(which):
    """The catalog table for the overlap set ``which`` (tree row indices), ascending in z.

    Catalog alphas are offset from the tree ones by a hand-chosen, z-dependent amount so
    the interpolated correction at a gap is a known number.
    """
    offsets = {0: -0.04, 1: +0.006, 2: +0.010, 3: +0.002, 4: +0.000}
    idx = sorted(which, key=lambda i: TREE_Z[i])         # ascending z
    z = np.array([TREE_Z[i] for i in idx])
    alpha = np.array([TREE_ALPHA[i] - offsets[i] for i in idx])
    n = np.array([TREE_N[i] * 2.0 for i in idx])         # distinct from the tree counts
    return z, alpha, n


def build(which):
    cat_z, cat_alpha, cat_n = catalog(which)
    out = builder.hybrid_table(TREE_Z, TREE_ALPHA, TREE_N, cat_z, cat_alpha, cat_n)
    return out, (cat_z, cat_alpha, cat_n)


# ---------------------------------------------------------------------------
# match_overlap_rows
# ---------------------------------------------------------------------------

def test_match_overlap_rows_picks_the_nearest_row():
    cat_z, _, _ = catalog({0, 1, 3})
    idx_tree, idx_cat = builder.match_overlap_rows(TREE_Z, cat_z)
    # cat_z is ascending: z = 6.0 (tree row 3), 10.0 (row 1), 12.0 (row 0).
    assert idx_tree.tolist() == [3, 1, 0]
    assert idx_cat.tolist() == [0, 1, 2]


def test_match_overlap_rows_tolerates_the_1e_5_disagreement_of_the_real_files():
    idx_tree, _ = builder.match_overlap_rows(TREE_Z, TREE_Z[[3, 1, 0]] + 1.0e-5)
    assert idx_tree.tolist() == [3, 1, 0]


def test_match_overlap_rows_rejects_an_unmatched_catalog_row():
    with pytest.raises(ValueError, match="no all-snapshot row within"):
        builder.match_overlap_rows(TREE_Z, np.array([7.5]))


def test_match_overlap_rows_rejects_duplicate_matches():
    with pytest.raises(ValueError, match="same all-snapshot row"):
        builder.match_overlap_rows(TREE_Z, np.array([6.0, 6.0]))


# ---------------------------------------------------------------------------
# hybrid_table: overlap rows are the catalog, verbatim
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("which", [{0, 1, 3}, {1, 3, 4}])
def test_overlap_rows_reproduce_the_catalog_exactly(which):
    out, (cat_z, cat_alpha, cat_n) = build(which)
    idx_tree, idx_cat = builder.match_overlap_rows(TREE_Z, cat_z)
    assert out["is_overlap"].sum() == 3
    assert sorted(np.flatnonzero(out["is_overlap"]).tolist()) == sorted(which)
    # Bit-identical, not merely close: the production table must equal alpha_z_number.txt.
    assert np.array_equal(out["mean_alpha"][idx_tree], cat_alpha[idx_cat])
    assert np.array_equal(out["n_halos"][idx_tree], cat_n[idx_cat])
    assert np.array_equal(out["correction"][idx_tree], np.zeros(3))


# ---------------------------------------------------------------------------
# hybrid_table: gap rows are tree minus the interpolated offset
# ---------------------------------------------------------------------------

def test_interior_gap_row_is_tree_minus_the_interpolated_offset():
    out, _ = build({0, 1, 3})
    # Gap row 2 (z = 9.0) sits between overlaps at z = 6.0 (offset +0.002) and z = 10.0
    # (offset +0.006); the weight is 3/4, not 1/2.
    expected_corr = 0.002 + (9.0 - 6.0) / (10.0 - 6.0) * (0.006 - 0.002)
    assert out["correction"][2] == pytest.approx(expected_corr)
    assert out["mean_alpha"][2] == pytest.approx(TREE_ALPHA[2] - expected_corr)
    assert out["n_halos"][2] == TREE_N[2]        # gap rows keep the tree count


def test_interior_gap_is_unaffected_by_the_other_overlap_set():
    out, _ = build({1, 3, 4})
    # Now the neighbours of z = 9.0 are z = 6.0 (+0.002) and z = 10.0 (+0.006) again, but
    # reached through a differently ordered catalog -- the answer must not change.
    expected_corr = 0.002 + (9.0 - 6.0) / (10.0 - 6.0) * (0.006 - 0.002)
    assert out["correction"][2] == pytest.approx(expected_corr)


def test_gap_below_the_overlap_range_clamps_to_the_low_z_end_offset():
    out, _ = build({0, 1, 3})
    # Gap row 4 (z = 5.5) is below every overlap z; the lowest-z overlap is row 3 (+0.002).
    assert out["correction"][4] == pytest.approx(0.002)
    assert out["mean_alpha"][4] == pytest.approx(TREE_ALPHA[4] - 0.002)


def test_gap_above_the_overlap_range_clamps_to_the_high_z_end_offset():
    out, _ = build({1, 3, 4})
    # Gap row 0 (z = 12.0) is above every overlap z; the highest-z overlap is row 1 (+0.006).
    assert out["correction"][0] == pytest.approx(0.006)
    assert out["mean_alpha"][0] == pytest.approx(TREE_ALPHA[0] - 0.006)


def test_descending_tree_order_is_handled():
    """The same problem fed ascending must give the same per-row answer.

    This is the regression guard for the ``np.interp`` orientation: interpolating against a
    decreasing ``xp`` would change the gap values here but not in an all-ascending fixture.
    """
    out_desc, (cat_z, cat_alpha, cat_n) = build({0, 1, 3})
    rev = np.argsort(TREE_Z)
    out_asc = builder.hybrid_table(TREE_Z[rev], TREE_ALPHA[rev], TREE_N[rev],
                                   cat_z, cat_alpha, cat_n)
    assert out_asc["mean_alpha"] == pytest.approx(out_desc["mean_alpha"][rev])
    assert out_asc["correction"] == pytest.approx(out_desc["correction"][rev])


# ---------------------------------------------------------------------------
# alpha_det candidates
# ---------------------------------------------------------------------------

def test_range_mean_covers_only_the_requested_snapshots():
    out, _ = build({0, 1, 3})
    t_Myr = np.array([100.0, 200.0, 300.0, 700.0, 900.0])
    cand = builder.alpha_det_candidates(SNAP, out["mean_alpha"], t_Myr, snap_lo=1, snap_hi=3)
    assert cand["n_range"] == 3
    assert cand["plain_range"] == pytest.approx(np.mean(out["mean_alpha"][1:4]))
    assert cand["plain_all"] == pytest.approx(np.mean(out["mean_alpha"]))
    expected_trap = builder._trapz(out["mean_alpha"][1:4], t_Myr[1:4]) / (700.0 - 200.0)
    assert cand["trapezoid_range"] == pytest.approx(expected_trap)


def test_range_mean_is_taken_in_snapshot_order_not_row_order():
    """Rows arriving in descending-z (= ascending-snap) order or shuffled must agree."""
    out, _ = build({0, 1, 3})
    t_Myr = np.array([100.0, 200.0, 300.0, 700.0, 900.0])
    perm = np.array([3, 0, 4, 1, 2])
    a = builder.alpha_det_candidates(SNAP, out["mean_alpha"], t_Myr, 1, 3)
    b = builder.alpha_det_candidates(SNAP[perm], out["mean_alpha"][perm], t_Myr[perm], 1, 3)
    assert b["plain_range"] == pytest.approx(a["plain_range"])
    assert b["trapezoid_range"] == pytest.approx(a["trapezoid_range"])


# ---------------------------------------------------------------------------
# read_table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_comments", [1, 2])
def test_read_table_skips_any_number_of_comment_lines(tmp_path, n_comments):
    path = tmp_path / "t.txt"
    path.write_text("".join(f"# comment {i}\n" for i in range(n_comments))
                    + "z  n_halos  mean_alpha\n"
                    + "5.5  10  0.49\n6.5  20  0.47\n")
    cols = builder.read_table(path)
    assert sorted(cols) == ["mean_alpha", "n_halos", "z"]
    assert cols["z"] == pytest.approx([5.5, 6.5])
    assert cols["mean_alpha"] == pytest.approx([0.49, 0.47])
