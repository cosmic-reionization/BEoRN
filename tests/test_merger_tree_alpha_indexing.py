"""Tests for merger-tree alpha fitting: snapshot indexing and branch termination.

Three defects lived in ``MergerTreeLoader.get_halo_accretion_rate_from_tree``:

A. the progenitor walk stored the progenitor's mass before the halo's own, shifting
   the mass history one snapshot relative to the redshifts it was fitted against;
B. ``tree_snap_num == redshift_index`` compared a raw simulation snapshot number
   against a position in the (filtered) ``redshifts`` array, selecting halos from the
   wrong snapshot and mapping their alphas onto another snapshot's groups;
C. a ``-1`` progenitor terminator was used as an array index, wrapping around to the
   last entry of the cache and splicing an unrelated halo onto the branch.

The fixture below is built so that each defect changes the answer:

- redshifts are **non-uniformly** spaced, which is what makes shift A observable. Under
  uniform spacing the no-intercept fit is invariant to a rigid shift of both the samples
  and the reference, so a uniform grid would silently pass;
- the loader exposes a **subset** of snapshots, so a position in ``redshifts`` is never
  equal to the raw snapshot number (defect B);
- one halo's branch terminates mid-window (defect C).
"""
import sys
import types

sys.modules.setdefault("MAS_library", types.SimpleNamespace(MASL=None))

import numpy as np
import pytest

from beorn.load_input_data.merger_tree_base import MergerTreeLoader
from beorn.structs.parameters import Parameters


# Descending, deliberately non-uniform -- gaps shrink toward low z, as in THESAN.
ALL_REDSHIFTS = np.array(
    [20.0, 18.0, 16.5, 15.2, 14.0, 13.0, 12.2, 11.5, 10.9, 10.4, 10.0, 9.7, 9.5, 9.4]
)
N_SNAPSHOTS = ALL_REDSHIFTS.size
N_HALOS = 5
# All below the clamp ceiling (alpha_grid[-2] = 1.0) so nothing is clipped.
TRUE_ALPHAS = np.array([0.30, 0.40, 0.50, 0.60, 0.70])
LOOKBACK = 5
# The loader exposes only snapshots 3.. , so redshift_index != raw snapshot number.
FIRST_EXPOSED_SNAPSHOT = 3


def entry_index(snapshot: int, halo: int) -> int:
    """Flat cache index of halo ``halo`` at snapshot ``snapshot``."""
    return snapshot * N_HALOS + halo


class SyntheticTreeLoader(MergerTreeLoader):
    """Loader over a synthetic tree with an exactly exponential mass history.

    Every halo follows ``M(z) = exp(-alpha * z)``, so ``vectorized_alpha_fit`` must
    recover ``alpha`` exactly from any window -- any residual is an indexing error,
    not a fitting inaccuracy.

    Args:
        parameters (Parameters): BEoRN parameter container.
        terminate: optional ``(halo, snapshot)`` -- that halo's branch has no
            progenitor below ``snapshot``, cutting its history short.
    """

    def __init__(self, parameters, terminate=None):
        super().__init__(parameters)
        self.terminate = terminate

    @property
    def redshifts(self):
        return ALL_REDSHIFTS[FIRST_EXPOSED_SNAPSHOT:]

    @property
    def snapshot_numbers(self):
        return np.arange(FIRST_EXPOSED_SNAPSHOT, N_SNAPSHOTS)

    @property
    def all_snapshot_redshifts(self):
        return ALL_REDSHIFTS

    def load_tree_cache(self):
        n = N_SNAPSHOTS * N_HALOS
        halo_ids = np.tile(np.arange(N_HALOS), N_SNAPSHOTS)
        snap_num = np.repeat(np.arange(N_SNAPSHOTS), N_HALOS)
        mass = np.exp(-TRUE_ALPHAS[halo_ids] * ALL_REDSHIFTS[snap_num])

        main_progenitor = np.full(n, -1, dtype=np.int64)
        for snapshot in range(1, N_SNAPSHOTS):
            for halo in range(N_HALOS):
                main_progenitor[entry_index(snapshot, halo)] = entry_index(snapshot - 1, halo)
        if self.terminate is not None:
            halo, snapshot = self.terminate
            main_progenitor[entry_index(snapshot, halo)] = -1

        is_central = np.ones(n, dtype=bool)
        return halo_ids, snap_num, mass, main_progenitor, is_central

    def get_halo_information_from_catalog(self, redshift_index):
        positions = np.zeros((N_HALOS, 3))
        masses = np.full(N_HALOS, 1e10)
        subhalo_to_group_map = np.arange(N_HALOS)
        return positions, masses, subhalo_to_group_map

    def load_density_field(self, redshift_index):
        raise AssertionError("alpha fitting must not touch the density field")


def make_parameters():
    parameters = Parameters()
    parameters.source.alpha_constant = None
    parameters.source.alpha_constant_z = None
    parameters.source.mass_accretion_lookback = LOOKBACK
    parameters.source.alpha_fallback = "mean"
    parameters.source.halo_mass_min = 1e9
    # Three edges so the clamp ceiling (alpha_grid[-2]) sits at 1.0, above every alpha.
    parameters.solver.halo_mass_accretion_alpha = np.array([0.0, 1.0, 2.0])
    return parameters


# ---------------------------------------------------------------------------
# Defects A and B: the fitted alpha must be exact despite filtering and non-uniform z
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("redshift_index", [4, 7, 10])
def test_fitted_alpha_is_exact_for_a_filtered_snapshot_list(redshift_index):
    loader = SyntheticTreeLoader(make_parameters())

    halo_ids, alphas = loader.get_halo_accretion_rate_from_tree(redshift_index)

    np.testing.assert_array_equal(halo_ids, np.arange(N_HALOS))
    np.testing.assert_allclose(alphas, TRUE_ALPHAS, rtol=1e-10)


def test_halos_are_selected_from_the_raw_snapshot_not_the_array_position():
    """The mass history must start at the halo's own mass at the raw snapshot."""
    loader = SyntheticTreeLoader(make_parameters())
    redshift_index = 7
    raw_snapshot = loader.snapshot_numbers[redshift_index]
    assert raw_snapshot != redshift_index, "fixture must not allow the two to coincide"

    # Fit a single halo by hand over the window the code should have used.
    window = np.arange(raw_snapshot, raw_snapshot - LOOKBACK, -1)
    expected_masses = np.exp(-TRUE_ALPHAS[0] * ALL_REDSHIFTS[window])

    _, alphas = loader.get_halo_accretion_rate_from_tree(redshift_index)

    from beorn.load_input_data.alpha_fitting import vectorized_alpha_fit
    reference = vectorized_alpha_fit(ALL_REDSHIFTS[window], expected_masses[None, :])
    np.testing.assert_allclose(alphas[0], reference[0], rtol=1e-10)


def test_a_uniform_grid_would_not_have_caught_the_shift():
    """Guards the fixture itself: the non-uniform spacing is what makes A observable."""
    spacings = np.diff(ALL_REDSHIFTS)
    assert not np.allclose(spacings, spacings[0]), (
        "ALL_REDSHIFTS must stay non-uniform or these tests stop detecting defect A"
    )


# ---------------------------------------------------------------------------
# Defect C: a terminated branch must not splice in the last cache entry
# ---------------------------------------------------------------------------

def test_short_branch_returns_nan_instead_of_wrapping_to_the_last_entry():
    redshift_index = 7
    raw_snapshot = FIRST_EXPOSED_SNAPSHOT + redshift_index
    # Halo 0 loses its progenitor two steps into a five-snapshot window.
    loader = SyntheticTreeLoader(make_parameters(), terminate=(0, raw_snapshot - 2))

    _, alphas = loader.get_halo_accretion_rate_from_tree(redshift_index)

    assert np.isnan(alphas[0]), "a branch shorter than the lookback must not be fitted"
    # Every other halo is untouched.
    np.testing.assert_allclose(alphas[1:], TRUE_ALPHAS[1:], rtol=1e-10)


def test_short_branch_halo_receives_the_fallback_alpha():
    redshift_index = 7
    raw_snapshot = FIRST_EXPOSED_SNAPSHOT + redshift_index
    loader = SyntheticTreeLoader(make_parameters(), terminate=(0, raw_snapshot - 2))

    catalog = loader.load_halo_catalog(redshift_index)

    assert np.all(np.isfinite(catalog.alphas)), "NaN must not reach the painted catalog"
    # alpha_fallback='mean' over the well-fitted halos only -- a NaN in that mean would
    # propagate to every halo.
    np.testing.assert_allclose(catalog.alphas[0], TRUE_ALPHAS[1:].mean(), rtol=1e-10)
    np.testing.assert_allclose(catalog.alphas[1:], TRUE_ALPHAS[1:], rtol=1e-10)


# ---------------------------------------------------------------------------
# Early snapshots: no branch fills the window, so alpha is inherited
# ---------------------------------------------------------------------------

# redshift_index 0 is raw snapshot 3, which has only 4 snapshots behind it -- one short
# of the 5-snapshot window.  Its halos must inherit from snapshot LOOKBACK - 1 = 4.
EARLY_REDSHIFT_INDEX = 0
REFERENCE_SNAPSHOT = LOOKBACK - 1


def test_early_snapshot_is_below_the_window():
    """Guards the fixture: this index must actually take the inheritance branch."""
    raw_snapshot = FIRST_EXPOSED_SNAPSHOT + EARLY_REDSHIFT_INDEX
    assert raw_snapshot + 1 < LOOKBACK
    assert REFERENCE_SNAPSHOT > raw_snapshot


def test_early_snapshot_inherits_alpha_from_the_reference_snapshot():
    loader = SyntheticTreeLoader(make_parameters())

    halo_ids, alphas = loader.get_halo_accretion_rate_from_tree(EARLY_REDSHIFT_INDEX)

    np.testing.assert_array_equal(halo_ids, np.arange(N_HALOS))
    # Each halo takes the alpha fitted for its own descendant at snapshot 4, which the
    # exponential fixture recovers exactly.
    np.testing.assert_allclose(alphas, TRUE_ALPHAS, rtol=1e-10)


def test_early_snapshot_is_not_painted_at_one_constant():
    """The old behaviour returned 0.04 for every halo, clamped to the grid floor."""
    loader = SyntheticTreeLoader(make_parameters())

    catalog = loader.load_halo_catalog(EARLY_REDSHIFT_INDEX)

    assert catalog.alphas.std() > 0, "halo-to-halo scatter must survive at early times"
    grid_floor = make_parameters().solver.halo_mass_accretion_alpha[0]
    assert not np.any(catalog.alphas == grid_floor)


def test_early_snapshot_halo_without_a_descendant_gets_the_fallback():
    """Halo 0 is not on any branch rooted at the reference snapshot."""
    # Cutting halo 0's progenitor link *at* snapshot 4 leaves its own branch one entry
    # long (so it is never fitted) and stops it reaching snapshot 3.
    loader = SyntheticTreeLoader(make_parameters(), terminate=(0, REFERENCE_SNAPSHOT))

    _, alphas = loader.get_halo_accretion_rate_from_tree(EARLY_REDSHIFT_INDEX)
    catalog = loader.load_halo_catalog(EARLY_REDSHIFT_INDEX)

    assert np.isnan(alphas[0]), "nothing to inherit must stay NaN, not a made-up value"
    np.testing.assert_allclose(alphas[1:], TRUE_ALPHAS[1:], rtol=1e-10)
    assert np.all(np.isfinite(catalog.alphas)), "NaN must not reach the painted catalog"
    np.testing.assert_allclose(catalog.alphas[0], TRUE_ALPHAS[1:].mean(), rtol=1e-10)
    np.testing.assert_allclose(catalog.alphas[1:], TRUE_ALPHAS[1:], rtol=1e-10)


def test_nearest_reference_wins_over_later_ones():
    """References are tried nearest first; a later one only fills what is still missing.

    Snapshot 4 already covers all five halos, so no branch walk beyond it may overwrite
    an alpha that is already set.
    """
    loader = SyntheticTreeLoader(make_parameters())
    references = []
    original = loader._walk_main_branches

    def recording_walk(root_rows, *args, **kwargs):
        result = original(root_rows, *args, **kwargs)
        references.append(int(result[1][0, 0]))
        return result

    loader._walk_main_branches = recording_walk
    _, alphas = loader.get_halo_accretion_rate_from_tree(EARLY_REDSHIFT_INDEX)

    assert references[0] == REFERENCE_SNAPSHOT, "the nearest reference must be tried first"
    np.testing.assert_allclose(alphas, TRUE_ALPHAS, rtol=1e-10)
    # Everything was covered by the first reference, so the loop stopped there.
    assert references == [REFERENCE_SNAPSHOT]


def test_later_references_fill_halos_the_nearest_one_misses():
    """A halo whose branch is too short at the nearest reference is picked up later.

    Halo 0's link out of snapshot 2 is cut, one snapshot *below* the painted one. The
    reference at snapshot 4 then sees only a three-snapshot branch (4, 3, 2) and cannot
    fit it, while the reference at snapshot 6 spans 6..2 -- a full window that still
    passes through the painted snapshot 3 -- so the fill loop must recover it.
    """
    loader = SyntheticTreeLoader(make_parameters(), terminate=(0, 2))
    references = []
    original = loader._walk_main_branches

    def recording_walk(root_rows, *args, **kwargs):
        result = original(root_rows, *args, **kwargs)
        references.append(int(result[1][0, 0]))
        return result

    loader._walk_main_branches = recording_walk
    _, alphas = loader.get_halo_accretion_rate_from_tree(EARLY_REDSHIFT_INDEX)

    assert references[0] == REFERENCE_SNAPSHOT
    assert len(references) > 1, "the loop must go on to later references"
    np.testing.assert_allclose(alphas, TRUE_ALPHAS, rtol=1e-10)


def test_inherited_alphas_are_not_recomputed_per_call():
    loader = SyntheticTreeLoader(make_parameters())
    walks = []
    original = loader._walk_main_branches

    def counting_walk(*args, **kwargs):
        walks.append(args[0])
        return original(*args, **kwargs)

    loader._walk_main_branches = counting_walk
    loader.get_halo_accretion_rate_from_tree(EARLY_REDSHIFT_INDEX)
    n_first = len(walks)
    loader.get_halo_accretion_rate_from_tree(EARLY_REDSHIFT_INDEX)

    assert n_first >= 1
    assert len(walks) == n_first, "the second call must reuse the cached alphas"


# ---------------------------------------------------------------------------
# Backward compatibility of the new properties
# ---------------------------------------------------------------------------

def test_snapshot_number_properties_default_to_the_identity_mapping():
    """Loaders that expose every snapshot need no override."""

    class PlainLoader(MergerTreeLoader):
        @property
        def redshifts(self):
            return np.array([9.0, 8.0, 7.0])

        def load_tree_cache(self):
            raise NotImplementedError

        def get_halo_information_from_catalog(self, redshift_index):
            raise NotImplementedError

        def load_density_field(self, redshift_index):
            raise NotImplementedError

    loader = PlainLoader(make_parameters())
    np.testing.assert_array_equal(loader.snapshot_numbers, [0, 1, 2])
    np.testing.assert_array_equal(loader.all_snapshot_redshifts, [9.0, 8.0, 7.0])
