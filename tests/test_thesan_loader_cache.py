"""Tests for the per-snapshot density-mesh disk cache in ThesanLoader."""
import sys
import types

sys.modules.setdefault("MAS_library", types.SimpleNamespace(MASL=None))

from pathlib import Path

import h5py
import numpy as np
import pytest

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
            origin=np.zeros(3), size=32.0, buffer=10.0, lbox_full=64.6917,
        ),
    )
    assert subbox._density_cache_path(0) != full_box._density_cache_path(0)
    assert "subbox" in subbox._density_cache_path(0).name


def _make_catalog_chunk(tmp_path, box_ckpc_h):
    """Write a minimal group-catalog chunk with Header/BoxSize and return its dir."""
    cat_dir = tmp_path / "groups_060"
    cat_dir.mkdir()
    with h5py.File(cat_dir / "fof_subhalo_tab_060.0.hdf5", "w") as f:
        f.create_group("Header").attrs["BoxSize"] = float(box_ckpc_h)
    return cat_dir


def _guard_loader(tmp_path, box_ckpc_h, lbox, subbox=None):
    loader = ThesanLoader.__new__(ThesanLoader)
    loader.parameters = Parameters()
    loader.parameters.simulation.Lbox = lbox
    loader.subbox = subbox
    loader._catalog_directories = [_make_catalog_chunk(tmp_path, box_ckpc_h)]
    loader.logger = types.SimpleNamespace(info=lambda *a, **k: None)
    return loader


def test_box_guard_passes_on_matching_lbox(tmp_path):
    # 64691.7 ckpc/h -> 64.6917 cMpc/h matches Lbox: no raise.
    _guard_loader(tmp_path, box_ckpc_h=64691.7, lbox=64.6917)._verify_box_size()


def test_box_guard_raises_on_mismatched_lbox(tmp_path):
    # The historical bug: Lbox left at the cMpc value 95.5 must be rejected.
    loader = _guard_loader(tmp_path, box_ckpc_h=64691.7, lbox=95.5)
    with pytest.raises(ValueError, match="BoxSize"):
        loader._verify_box_size()


def test_box_guard_uses_subbox_full_box(tmp_path):
    # In subbox mode the guard compares against subbox.lbox_full, not Lbox (=lbox_eff).
    subbox = SubboxConfig(origin=np.zeros(3), size=32.0, buffer=10.0, lbox_full=64.6917)
    loader = _guard_loader(tmp_path, box_ckpc_h=64691.7, lbox=52.0, subbox=subbox)
    loader._verify_box_size()  # matches lbox_full=64.6917 -> no raise


def test_load_density_field_returns_cached_mesh_without_reading_particles(tmp_path):
    loader = make_loader(cache_dir=tmp_path)
    delta = np.random.default_rng(0).normal(size=(8, 8, 8))
    np.save(loader._density_cache_path(0), delta)

    # The snapshot directory does not exist, so any particle I/O would raise --
    # a successful return proves the cache short-circuited the computation.
    result = loader.load_density_field(0)
    np.testing.assert_array_equal(result, delta)
