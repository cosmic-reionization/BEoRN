"""Tests for the subhalo→FoF projection that assigns tree alphas to painted groups.

THESAN's LHaloTree is built on SUBFIND *subhalos*; the painted catalog is built from
*FoF groups*.  Several subhalos share one ``SubhaloGrNr``, so the projection is
many-to-one unless the tree roots are filtered to FoF centrals first.

Before the fix, ``ThesanLoader.load_tree_cache`` dropped the cache's ``tree_is_central``
dataset.  Every subhalo was then fitted and written into its group's row, and NumPy's
last-write-wins handed the row to the highest ``SubhaloNumber`` — the least-bound
satellite, whose stripped mass history fits an alpha at or below zero.  See
``fst_stochastic/docs/central_satellite.md``.

The fixture encodes exactly that geometry: group 0 holds a central plus two satellites
whose masses *decrease* with time, group 1 holds a lone central, and group 2 has no tree
entry at all.
"""
import sys
import types

sys.modules.setdefault("MAS_library", types.SimpleNamespace(MASL=None))

import logging

import h5py
import numpy as np
import pytest

from beorn.load_input_data.cosmo_sim_thesan import ThesanLoader
from beorn.load_input_data.merger_tree_base import MergerTreeLoader
from beorn.structs.parameters import Parameters


# Descending and non-uniform, as in THESAN.
ALL_REDSHIFTS = np.array([12.0, 11.2, 10.5, 9.9, 9.4, 9.0, 8.7, 8.5])
N_SNAPSHOTS = ALL_REDSHIFTS.size
LOOKBACK = 4
FALLBACK = 0.123

N_SUBHALOS = 5
# subhalo -> FoF group.  0,1,2 share group 0 (central + two satellites); 3 is the lone
# central of group 1; 4 belongs to group 2 and never appears in the tree.
SUBHALO_TO_GROUP = np.array([0, 0, 0, 1, 2])
N_GROUPS = 3

# Central subhalos grow (positive alpha); satellites are stripped, so their mass rises
# toward high z and the fit returns a negative alpha that clamps to the grid floor.
SUBHALO_ALPHAS = np.array([0.50, -0.40, -0.60, 0.80, np.nan])
CENTRAL_SUBHALOS = {0, 3}


def entry_index(snapshot: int, subhalo: int) -> int:
    """Flat cache index of ``subhalo`` at ``snapshot``."""
    return snapshot * N_SUBHALOS + subhalo


def build_cache(
    *,
    with_central_mask: bool = True,
    demote_progenitors_of: int | None = None,
    duplicate_central: bool = False,
    invalid_subhalo_id: bool = False,
):
    """Build the flat tree arrays.

    Args:
        with_central_mask: return five arrays instead of four.
        demote_progenitors_of: mark this subhalo as a satellite at every snapshot
            *before* the last one, leaving it central only at the painted snapshot.
        duplicate_central: also mark satellite subhalo 1 as central, so two centrals
            map onto group 0.
        invalid_subhalo_id: give subhalo 2 a ``SubhaloNumber`` outside the catalog.
    """
    n = N_SNAPSHOTS * N_SUBHALOS
    halo_ids = np.tile(np.arange(N_SUBHALOS), N_SNAPSHOTS)
    snap_num = np.repeat(np.arange(N_SNAPSHOTS), N_SUBHALOS)

    alphas = np.where(np.isnan(SUBHALO_ALPHAS), 0.0, SUBHALO_ALPHAS)
    mass = np.exp(-alphas[halo_ids] * ALL_REDSHIFTS[snap_num])
    # Subhalo 4 is absent from the tree: zero mass drops it from every root mask.
    mass[halo_ids == 4] = 0.0

    main_progenitor = np.full(n, -1, dtype=np.int64)
    for snapshot in range(1, N_SNAPSHOTS):
        for subhalo in range(N_SUBHALOS):
            main_progenitor[entry_index(snapshot, subhalo)] = entry_index(snapshot - 1, subhalo)

    if invalid_subhalo_id:
        halo_ids = halo_ids.copy()
        halo_ids[halo_ids == 2] = 99

    if not with_central_mask:
        return halo_ids, snap_num, mass, main_progenitor

    is_central = np.isin(halo_ids, sorted(CENTRAL_SUBHALOS))
    if duplicate_central:
        is_central |= halo_ids == 1
    if demote_progenitors_of is not None:
        earlier = (halo_ids == demote_progenitors_of) & (snap_num < N_SNAPSHOTS - 1)
        is_central = is_central & ~earlier
    return halo_ids, snap_num, mass, main_progenitor, is_central


class CentralMappingLoader(MergerTreeLoader):
    """Loader over the synthetic subhalo tree above."""

    def __init__(self, parameters, cache):
        super().__init__(parameters)
        self._cache = cache

    @property
    def redshifts(self):
        return ALL_REDSHIFTS

    def load_tree_cache(self):
        return self._cache

    def get_halo_information_from_catalog(self, redshift_index):
        positions = np.zeros((N_GROUPS, 3))
        masses = np.full(N_GROUPS, 1e10)
        return positions, masses, SUBHALO_TO_GROUP

    def load_density_field(self, redshift_index):
        raise AssertionError("alpha assignment must not touch the density field")


def make_parameters():
    parameters = Parameters()
    parameters.source.alpha_constant = None
    parameters.source.alpha_constant_z = None
    parameters.source.mass_accretion_lookback = LOOKBACK
    parameters.source.alpha_fallback = FALLBACK
    parameters.source.halo_mass_min = 1e9
    parameters.source.halo_mass_max = 1e16
    # Grid floor 0.0, clamp ceiling (alpha_grid[-2]) 1.0 -- above every central alpha.
    parameters.solver.halo_mass_accretion_alpha = np.array([0.0, 1.0, 2.0])
    return parameters


def alphas_at_last_snapshot(cache):
    loader = CentralMappingLoader(make_parameters(), cache)
    return loader.load_halo_catalog(N_SNAPSHOTS - 1).alphas


# ---------------------------------------------------------------------------
# The central alpha must survive; a satellite must not overwrite it
# ---------------------------------------------------------------------------

def test_group_with_satellites_receives_the_central_alpha():
    alphas = alphas_at_last_snapshot(build_cache())
    assert alphas[0] == pytest.approx(SUBHALO_ALPHAS[0], abs=1e-9)
    assert alphas[1] == pytest.approx(SUBHALO_ALPHAS[3], abs=1e-9)


def test_without_the_mask_the_last_satellite_wins():
    """Documents the legacy four-array behaviour the central mask exists to prevent."""
    alphas = alphas_at_last_snapshot(build_cache(with_central_mask=False))
    # Subhalo 2 has the highest SubhaloNumber in group 0 and a stripped history, so the
    # group is painted at the grid floor instead of the central's 0.50.
    assert alphas[0] == pytest.approx(0.0, abs=1e-9)
    assert alphas[0] != pytest.approx(SUBHALO_ALPHAS[0], abs=1e-3)


def test_legacy_four_array_cache_warns_but_does_not_raise(caplog):
    with caplog.at_level(logging.WARNING, logger="beorn.load_input_data.merger_tree_base"):
        alphas_at_last_snapshot(build_cache(with_central_mask=False))
    assert any("more than one tree-derived alpha" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Projection invariants
# ---------------------------------------------------------------------------

def test_duplicate_centrals_onto_one_group_raise():
    with pytest.raises(ValueError, match="central mask"):
        alphas_at_last_snapshot(build_cache(duplicate_central=True))


def test_group_without_a_tree_central_gets_the_fallback():
    alphas = alphas_at_last_snapshot(build_cache())
    assert alphas[2] == pytest.approx(FALLBACK, abs=1e-9)


def test_out_of_range_subhalo_ids_are_dropped_not_indexed():
    alphas = alphas_at_last_snapshot(build_cache(invalid_subhalo_id=True))
    # Subhalo 99 does not exist in the catalog; the central of group 0 still wins.
    assert alphas[0] == pytest.approx(SUBHALO_ALPHAS[0], abs=1e-9)


# ---------------------------------------------------------------------------
# Root selection must not truncate the branch
# ---------------------------------------------------------------------------

def test_central_root_follows_a_branch_that_was_satellite_earlier():
    """Only the root at the painted snapshot must be central.

    LHaloTree's ``FirstProgenitor`` chain is the subhalo's own track; a progenitor
    classified as a satellite at an earlier snapshot is still part of it.
    """
    alphas = alphas_at_last_snapshot(build_cache(demote_progenitors_of=0))
    assert alphas[0] == pytest.approx(SUBHALO_ALPHAS[0], abs=1e-9)


# ---------------------------------------------------------------------------
# ThesanLoader.load_tree_cache
# ---------------------------------------------------------------------------

def write_v2_cache(path, *, include_central: bool, format_version: int = 2):
    halo_ids, snap_num, mass, main_progenitor, is_central = build_cache()
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = format_version
        f.create_dataset("tree_halo_ids", data=halo_ids)
        f.create_dataset("tree_snap_num", data=snap_num)
        f.create_dataset("tree_mass", data=mass)
        f.create_dataset("tree_main_progenitor", data=main_progenitor)
        if include_central:
            f.create_dataset("tree_is_central", data=is_central)
    return path


def make_bare_thesan_loader(cache_path):
    """A ThesanLoader with only what ``load_tree_cache`` touches.

    The full constructor walks a THESAN directory tree, which is not available here.
    """
    loader = ThesanLoader.__new__(ThesanLoader)
    loader.cached_tree = cache_path
    loader.logger = logging.getLogger("beorn.test.thesan_loader")
    loader._tree_cache_arrays = None
    return loader


def test_v2_cache_returns_and_memoizes_five_arrays(tmp_path):
    path = write_v2_cache(tmp_path / "tree_v2.hdf5", include_central=True)
    loader = make_bare_thesan_loader(path)

    arrays = loader.load_tree_cache()
    assert len(arrays) == 5
    assert arrays[4].dtype == bool
    assert arrays[4].sum() == np.isin(arrays[0], sorted(CENTRAL_SUBHALOS)).sum()

    # Second call is memoized: the same objects come back, and deleting the file on disk
    # must not matter.
    path.unlink()
    again = loader.load_tree_cache()
    assert all(a is b for a, b in zip(arrays, again))


def test_v2_cache_missing_the_central_mask_raises(tmp_path):
    path = write_v2_cache(tmp_path / "tree_v2_nocentral.hdf5", include_central=False)
    loader = make_bare_thesan_loader(path)
    with pytest.raises(ValueError, match="tree_is_central"):
        loader.load_tree_cache()


def test_unversioned_cache_without_the_mask_warns_and_returns_four(tmp_path, caplog):
    path = write_v2_cache(
        tmp_path / "tree_v1.hdf5", include_central=False, format_version=0
    )
    loader = make_bare_thesan_loader(path)
    with caplog.at_level(logging.WARNING, logger="beorn.test.thesan_loader"):
        arrays = loader.load_tree_cache()
    assert len(arrays) == 4
    assert any("tree_is_central" in r.message for r in caplog.records)
