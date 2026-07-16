"""Tests for the per-snapshot density-mesh disk cache in ThesanLoader."""
import sys
import types

sys.modules.setdefault("MAS_library", types.SimpleNamespace(MASL=None))

from pathlib import Path

import numpy as np

from beorn.load_input_data.cosmo_sim_thesan import ThesanLoader, SubboxConfig
from beorn.structs.parameters import Parameters


def make_loader(cache_dir=None, subbox=None, ncell=8):
    """Bare loader with just the state _density_cache_path/load_density_field need."""
    loader = ThesanLoader.__new__(ThesanLoader)
    loader.parameters = Parameters()
    loader.parameters.simulation.Ncell = ncell
    loader.subbox = subbox
    loader.density_cache_dir = Path(cache_dir) if cache_dir is not None else None
    loader._density_directories = [Path("/nonexistent/snapdir_060")]
    return loader


def test_density_cache_path_none_without_cache_dir():
    loader = make_loader(cache_dir=None)
    assert loader._density_cache_path(0) is None


def test_density_cache_path_keyed_by_snapshot_mesh_and_scheme(tmp_path):
    loader = make_loader(cache_dir=tmp_path)
    path = loader._density_cache_path(0)

    assert path.parent == tmp_path
    assert path.suffix == ".npy"
    assert "snapdir_060" in path.name
    assert "N8" in path.name

    loader.parameters.simulation.Ncell = 16
    assert loader._density_cache_path(0) != path

    loader.parameters.simulation.Ncell = 8
    loader.parameters.cosmo_sim.halo_catalogs_thesan_mass_assignment = "NGP"
    assert loader._density_cache_path(0) != path


def test_density_cache_path_keyed_by_subbox(tmp_path):
    full_box = make_loader(cache_dir=tmp_path)
    subbox = make_loader(
        cache_dir=tmp_path,
        subbox=SubboxConfig(
            origin=np.zeros(3), size=47.75, buffer=10.0, lbox_full=95.5,
        ),
    )
    assert subbox._density_cache_path(0) != full_box._density_cache_path(0)
    assert "subbox" in subbox._density_cache_path(0).name


def test_load_density_field_returns_cached_mesh_without_reading_particles(tmp_path):
    loader = make_loader(cache_dir=tmp_path)
    delta = np.random.default_rng(0).normal(size=(8, 8, 8))
    np.save(loader._density_cache_path(0), delta)

    # The snapshot directory does not exist, so any particle I/O would raise --
    # a successful return proves the cache short-circuited the computation.
    result = loader.load_density_field(0)
    np.testing.assert_array_equal(result, delta)
