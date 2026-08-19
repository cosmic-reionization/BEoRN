"""Tests for the src/beorn changes that support the THESAN-1 four-case runs.

See fst_stochastic/docs/thesan1_four_cases.md section 4 for what each change is for.
"""
import sys
import types

sys.modules.setdefault("MAS_library", types.SimpleNamespace(MASL=None))

from pathlib import Path

import numpy as np
import pytest

from beorn.load_input_data.cosmo_sim_thesan import ThesanLoader
from beorn.load_input_data.merger_tree_base import MergerTreeLoader
from beorn.structs.helpers import bin_centers
from beorn.structs.parameters import Parameters


# ---------------------------------------------------------------------------
# source.alpha_constant  (section 4.1)
# ---------------------------------------------------------------------------

class TreeCounterLoader(MergerTreeLoader):
    """Minimal merger-tree loader that records whether the tree was ever read."""

    def __init__(self, parameters, n_halos=5):
        super().__init__(parameters)
        self.tree_reads = 0
        self.n_halos = n_halos

    @property
    def redshifts(self):
        return np.array([8.0, 7.0, 6.0])

    def load_tree_cache(self):
        self.tree_reads += 1
        # One entry per halo at every snapshot, each its own (terminating) branch.
        n = self.n_halos * self.redshifts.size
        tree_halo_ids = np.tile(np.arange(self.n_halos), self.redshifts.size)
        tree_snap_num = np.repeat(np.arange(self.redshifts.size), self.n_halos)
        tree_mass = np.full(n, 1e10)
        tree_main_progenitor = np.full(n, -1)
        return tree_halo_ids, tree_snap_num, tree_mass, tree_main_progenitor

    def get_halo_information_from_catalog(self, redshift_index):
        positions = np.zeros((self.n_halos, 3))
        masses = np.full(self.n_halos, 1e10)
        subhalo_to_group_map = np.arange(self.n_halos)
        return positions, masses, subhalo_to_group_map

    def load_density_field(self, redshift_index):
        raise AssertionError("load_halo_catalog must not touch the density field")


def make_parameters(alpha_constant=None, alpha_constant_z=None, alpha_grid=(0.4077, 0.5077)):
    parameters = Parameters()
    parameters.source.alpha_constant = alpha_constant
    parameters.source.alpha_constant_z = alpha_constant_z
    parameters.source.halo_mass_min = 1e9
    parameters.solver.halo_mass_accretion_alpha = np.array(alpha_grid)
    return parameters


def test_alpha_constant_assigns_the_same_alpha_to_every_halo():
    parameters = make_parameters(alpha_constant=0.4577)
    loader = TreeCounterLoader(parameters)

    catalog = loader.load_halo_catalog(0)

    np.testing.assert_allclose(catalog.alphas, 0.4577)


def test_alpha_constant_never_reads_the_merger_tree():
    """This is the whole point: the tree cache is multi-GB and read per snapshot."""
    parameters = make_parameters(alpha_constant=0.4577)
    loader = TreeCounterLoader(parameters)

    loader.load_halo_catalog(0)
    loader.load_halo_catalog(1)

    assert loader.tree_reads == 0


def test_alpha_from_tree_when_alpha_constant_is_unset():
    parameters = make_parameters(alpha_constant=None, alpha_grid=(0.0, 0.5, 1.0))
    loader = TreeCounterLoader(parameters)

    loader.load_halo_catalog(2)

    assert loader.tree_reads > 0


def test_alpha_constant_outside_the_paintable_grid_raises():
    """A mis-binned constant alpha would paint every halo with the wrong profile."""
    parameters = make_parameters(alpha_constant=2.0, alpha_grid=(0.4077, 0.5077))
    loader = TreeCounterLoader(parameters)

    with pytest.raises(ValueError, match="alpha_constant"):
        loader.load_halo_catalog(0)


def test_alpha_constant_is_not_a_paint_only_key():
    """It changes the profiles, so it must invalidate the profile cache."""
    assert "alpha_constant" not in Parameters._PAINT_ONLY_SOURCE_KEYS

    base = Parameters()
    other = Parameters()
    other.source.alpha_constant = 0.4577
    assert base.profiles_hash() != other.profiles_hash()
    assert base.profiles_fstar_hash() != other.profiles_fstar_hash()


# ---------------------------------------------------------------------------
# source.alpha_constant_z  (fst_stochastic 'alpha_z' case, docs/thesan1_four_cases.md)
# ---------------------------------------------------------------------------

def test_alpha_constant_z_interpolates_per_snapshot():
    """Every halo at a snapshot gets np.interp'd alpha for that snapshot's redshift."""
    table = np.array([[6.0, 7.0, 8.0], [0.40, 0.45, 0.50]])
    parameters = make_parameters(alpha_constant_z=table, alpha_grid=(0.0, 1.0))
    loader = TreeCounterLoader(parameters)  # loader.redshifts == [8.0, 7.0, 6.0]

    np.testing.assert_allclose(loader.load_halo_catalog(0).alphas, 0.50)  # z=8.0
    np.testing.assert_allclose(loader.load_halo_catalog(1).alphas, 0.45)  # z=7.0
    np.testing.assert_allclose(loader.load_halo_catalog(2).alphas, 0.40)  # z=6.0


def test_alpha_constant_z_never_reads_the_merger_tree():
    table = np.array([[6.0, 7.0, 8.0], [0.40, 0.45, 0.50]])
    parameters = make_parameters(alpha_constant_z=table, alpha_grid=(0.0, 1.0))
    loader = TreeCounterLoader(parameters)

    loader.load_halo_catalog(0)
    loader.load_halo_catalog(1)

    assert loader.tree_reads == 0


def test_alpha_constant_z_extrapolates_flat_beyond_table_range():
    """Redshifts outside the table's coverage clip to the nearest edge value."""
    table = np.array([[6.0, 7.0], [0.40, 0.45]])
    parameters = make_parameters(alpha_constant_z=table, alpha_grid=(0.0, 1.0))
    loader = TreeCounterLoader(parameters)  # z=8.0 (index 0) is above the table's max z=7.0

    np.testing.assert_allclose(loader.load_halo_catalog(0).alphas, 0.45)


def test_alpha_constant_takes_precedence_over_alpha_constant_z():
    table = np.array([[6.0, 7.0, 8.0], [0.40, 0.45, 0.50]])
    parameters = make_parameters(
        alpha_constant=0.4577, alpha_constant_z=table, alpha_grid=(0.0, 1.0)
    )
    loader = TreeCounterLoader(parameters)

    np.testing.assert_allclose(loader.load_halo_catalog(0).alphas, 0.4577)


def test_alpha_constant_z_is_not_a_paint_only_key():
    """It changes the profiles, so it must invalidate the profile cache."""
    assert "alpha_constant_z" not in Parameters._PAINT_ONLY_SOURCE_KEYS

    base = Parameters()
    other = Parameters()
    other.source.alpha_constant_z = np.array([[6.0, 7.0], [0.40, 0.45]])
    assert base.profiles_hash() != other.profiles_hash()
    assert base.profiles_fstar_hash() != other.profiles_fstar_hash()


# ---------------------------------------------------------------------------
# load_tree_cache memoization  (section 4.2)
# ---------------------------------------------------------------------------

def test_thesan_load_tree_cache_reads_the_file_once(tmp_path):
    h5py = pytest.importorskip("h5py")

    cache_file = tmp_path / "tree_cache.hdf5"
    with h5py.File(cache_file, "w") as f:
        f["tree_halo_ids"] = np.arange(4)
        f["tree_snap_num"] = np.zeros(4, dtype=int)
        f["tree_mass"] = np.full(4, 1e10)
        f["tree_main_progenitor"] = np.full(4, -1)

    loader = ThesanLoader.__new__(ThesanLoader)
    loader.parameters = Parameters()
    loader.cached_tree = cache_file
    loader.logger = __import__("logging").getLogger("test")

    first = loader.load_tree_cache()
    # Deleting the file proves the second call cannot be re-reading it.
    cache_file.unlink()
    second = loader.load_tree_cache()

    for a, b in zip(first, second):
        assert a is b


# ---------------------------------------------------------------------------
# Snapshots with a cached density mesh but no snapdir  (section 4.3)
# ---------------------------------------------------------------------------

def make_density_loader(cache_dir, snapdir_name="snapdir_069", ncell=8):
    loader = ThesanLoader.__new__(ThesanLoader)
    loader.parameters = Parameters()
    loader.parameters.simulation.Ncell = ncell
    loader.subbox = None
    loader.density_cache_dir = Path(cache_dir) if cache_dir is not None else None
    loader._density_directories = [Path("/nonexistent") / snapdir_name]
    loader.logger = __import__("logging").getLogger("test")
    return loader


def test_density_cache_path_for_snapdir_name_matches_the_indexed_form(tmp_path):
    """__init__ needs the cache key before _density_directories exists."""
    loader = make_density_loader(tmp_path)
    assert (
        loader._density_cache_path_for_snapdir_name("snapdir_069")
        == loader._density_cache_path(0)
    )


def test_load_density_field_raises_when_snapdir_and_cache_are_both_missing(tmp_path):
    """Without this guard an empty mesh yields delta = nan everywhere, silently."""
    loader = make_density_loader(tmp_path)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        loader.load_density_field(0)


# ---------------------------------------------------------------------------
# bin_centers uniform-spacing detection  (section 4.5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "edges, expected_first_two",
    [
        (np.linspace(0.0, 1.5, 16), (0.05, 0.15)),   # THESAN-1 tree alpha grid
        (np.linspace(0.0, 5.0, 26), (0.1, 0.3)),     # the old Arizona alpha grid
        (np.array([0.4077, 0.5077]), (0.4577, None)),  # constant-alpha single bin
    ],
)
def test_uniform_grids_get_arithmetic_centers(edges, expected_first_two):
    """np.linspace output never satisfies exact-equality spacing, which used to send
    every uniform alpha grid down the geometric branch (first centre = 0)."""
    centers = bin_centers(edges)
    assert centers[0] == pytest.approx(expected_first_two[0])
    if expected_first_two[1] is not None:
        assert centers[1] == pytest.approx(expected_first_two[1])


def test_log_spaced_grids_still_get_geometric_centers():
    edges = np.logspace(8, 17, 40)
    centers = bin_centers(edges)
    np.testing.assert_allclose(centers, np.sqrt(edges[:-1] * edges[1:]))


# ---------------------------------------------------------------------------
# PaintingCoordinator.resolve_painting_indices  (section 4.4)
# ---------------------------------------------------------------------------

def test_resolve_painting_indices_is_public_and_delegates():
    from beorn.painting.coordinator import PaintingCoordinator

    painter = PaintingCoordinator.__new__(PaintingCoordinator)
    calls = []
    painter._resolve_painting_indices = lambda subset: calls.append(subset) or [1, 2]

    assert painter.resolve_painting_indices([7.0, 6.0]) == [1, 2]
    assert calls == [[7.0, 6.0]]
