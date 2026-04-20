import sys
import types
import pickle
import h5py

# Optional dependency stub: importing beorn may pull in the pylians backend.
if "MAS_library" not in sys.modules:
    sys.modules["MAS_library"] = types.SimpleNamespace(MASL=None)

import numpy as np
import pytest
from types import SimpleNamespace

from beorn.painting.coordinator import PaintingCoordinator
from beorn.structs.coeval_cube import CoevalCube
from beorn.structs.parameters import Parameters
from beorn.structs.radiation_profiles import RadiationProfilesFStarGrid
from beorn.structs.temporal_cube import TemporalCube


class DummyHaloCatalog:
    def __init__(self, masses, alpha_vals, selected=None):
        self.masses = np.asarray(masses)
        self.alpha_vals = np.asarray(alpha_vals)
        if selected is None:
            self.selected = np.arange(self.masses.size)
        else:
            self.selected = np.asarray(selected)

    @property
    def size(self):
        return self.selected.size

    def get_halo_indices(self, alpha_range, mass_range):
        alpha_low, alpha_high = alpha_range
        mass_low, mass_high = mass_range
        mask = (
            (self.alpha_vals[self.selected] >= alpha_low)
            & (self.alpha_vals[self.selected] < alpha_high)
            & (self.masses[self.selected] >= mass_low)
            & (self.masses[self.selected] < mass_high)
        )
        return np.where(mask)[0]

    def at_indices(self, idx):
        idx = np.asarray(idx)
        return DummyHaloCatalog(self.masses, self.alpha_vals, self.selected[idx])

    def to_mesh(self):
        return np.zeros((4, 4, 4), dtype=float)


class DummyLoader:
    def __init__(self, halo_catalog):
        self._halo_catalog = halo_catalog
        self.redshifts = np.array([12.0, 10.0])

    def load_halo_catalog(self, z_index):
        return self._halo_catalog

    def load_density_field(self, z_index):
        return np.zeros((4, 4, 4), dtype=float)


class DummyHandler:
    file_root = "dummy.h5"
    write_kwargs = {}

    def write_file(self, parameters, obj, **kwargs):
        return None

    def load_file(self, parameters, cls, **kwargs):
        return None


def make_parameters(seed=12345):
    params = SimpleNamespace(
        simulation=SimpleNamespace(
            Ncell=4,
            Lbox=10.0,
            cores=1,
            fft_backend='numpy',
            store_grids=["Grid_xHII", "Grid_Temp", "Grid_xal"],
            minimum_grid_size_heat=1,
            minimum_grid_size_lyal=1,
            compute_s_alpha_fluctuations=False,
            halo_mass_bins=np.array([1.0e8, 1.0e9, 1.0e10]),
            halo_mass_accretion_alpha=np.array([0.5, 1.0]),
        ),
        source=SimpleNamespace(
            min_xHII_value=0.0,
            f_st=0.1,
            f_st_grid_min=0.01,
            f_st_grid_max=0.2,
            f_st_paint_distribution="lognormal",
            f_st_paint_sigma=0.4,
            f_st_paint_min=0.01,
            f_st_paint_max=0.2,
            f_st_paint_seed=seed,
        ),
    )
    params.unique_hash = lambda: "testhash"
    return params


def make_profiles():
    z_n = 2
    mass_n = 2
    alpha_n = 1
    f_n = 4
    r_n = 3
    rlya_n = 5

    return RadiationProfilesFStarGrid(
        parameters=SimpleNamespace(),
        z_history=np.array([12.0, 10.0]),
        halo_mass_bins=np.array(
            [
                [[1.0e8, 1.0e8]],
                [[1.0e9, 1.0e9]],
                [[1.0e10, 1.0e10]],
            ]
        ),
        f_st_grid=np.array([0.01, 0.05, 0.1, 0.2]),
        rho_xray=np.ones((r_n, mass_n, alpha_n, f_n, z_n)),
        rho_heat=np.ones((r_n, mass_n, alpha_n, f_n, z_n)),
        rho_alpha=np.ones((rlya_n, mass_n, alpha_n, f_n, z_n)),
        R_bubble=np.ones((mass_n, alpha_n, f_n, z_n)),
        r_lyal=np.logspace(-5, 2, rlya_n),
        r_grid_cell=np.logspace(-2, 1, r_n),
    )


def make_coordinator(seed=12345, halo_catalog=None):
    if halo_catalog is None:
        halo_catalog = DummyHaloCatalog(
            masses=np.array([2.0e8, 3.0e8, 4.0e8, 5.0e8, 6.0e8]),
            alpha_vals=np.array([0.7, 0.7, 0.7, 0.7, 0.7]),
        )
    params = make_parameters(seed=seed)
    loader = DummyLoader(halo_catalog)
    return PaintingCoordinator(params, loader, DummyHandler(), cache_handler=None)


def test_make_f_st_rng_reproducible_per_snapshot():
    coordinator = make_coordinator(seed=2468)

    rng_a = coordinator._make_f_st_rng(0)
    rng_b = coordinator._make_f_st_rng(0)
    rng_c = coordinator._make_f_st_rng(1)

    seq_a = rng_a.lognormal(mean=np.log(0.1), sigma=0.4, size=8)
    seq_b = rng_b.lognormal(mean=np.log(0.1), sigma=0.4, size=8)
    seq_c = rng_c.lognormal(mean=np.log(0.1), sigma=0.4, size=8)

    np.testing.assert_allclose(seq_a, seq_b)
    assert not np.allclose(seq_a, seq_c)


def test_sample_f_st_for_halos_reproducible_given_snapshot_rng():
    coordinator = make_coordinator(seed=999)

    rng1 = coordinator._make_f_st_rng(1)
    rng2 = coordinator._make_f_st_rng(1)

    s1 = coordinator._sample_f_st_for_halos(20, rng1)
    s2 = coordinator._sample_f_st_for_halos(20, rng2)

    np.testing.assert_allclose(s1, s2)
    assert np.all(s1 >= coordinator.parameters.source.f_st_paint_min)
    assert np.all(s1 <= coordinator.parameters.source.f_st_paint_max)


def test_paint_single_fstar_uses_reproducible_fst_assignment_per_snapshot(monkeypatch):
    halo_catalog = DummyHaloCatalog(
        masses=np.array([2.0e8, 2.5e8, 3.0e8, 3.5e8, 4.0e8, 4.5e8]),
        alpha_vals=np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7]),
    )
    coordinator = make_coordinator(seed=13579, halo_catalog=halo_catalog)
    profiles = make_profiles()

    picked_groups_run1 = []
    picked_groups_run2 = []
    run_state = {"count": 0}

    def fake_profiles_of_halo_bin(self, z_index, alpha_index, mass_index, f_st_index):
        tag = float(f_st_index)
        return (
            np.array(tag),
            np.array([tag]),
            np.array([tag]),
        )

    def fake_paint_single_mass_bin(self, halo_catalog, z, radial_grid, r_lyal, profiles_of_bin, **kwargs):
        record = (int(np.asarray(profiles_of_bin[0]).item()), tuple(halo_catalog.selected.tolist()))
        if run_state["count"] == 0:
            picked_groups_run1.append(record)
        else:
            picked_groups_run2.append(record)
        return (None, None, None)

    monkeypatch.setattr(RadiationProfilesFStarGrid, "profiles_of_halo_bin", fake_profiles_of_halo_bin)
    monkeypatch.setattr(PaintingCoordinator, "paint_single_mass_bin", fake_paint_single_mass_bin)
    monkeypatch.setattr("beorn.painting.coordinator.spreading_excess_fast", lambda parameters, grid: grid)
    monkeypatch.setattr("beorn.painting.coordinator.T_adiab_fluctu", lambda z, parameters, delta_b: np.zeros_like(delta_b))
    monkeypatch.setattr("beorn.painting.coordinator.S_alpha", lambda z, temp, neutral_frac: 1.0)

    coordinator.paint_single_fstar(0, profiles)
    run_state["count"] = 1
    coordinator.paint_single_fstar(0, profiles)

    assert picked_groups_run1 == picked_groups_run2
    assert len(picked_groups_run1) > 0

    picked_groups_run3 = []

    def fake_paint_single_mass_bin_run3(self, halo_catalog, z, radial_grid, r_lyal, profiles_of_bin, **kwargs):
        picked_groups_run3.append((int(np.asarray(profiles_of_bin[0]).item()), tuple(halo_catalog.selected.tolist())))
        return (None, None, None)

    monkeypatch.setattr(PaintingCoordinator, "paint_single_mass_bin", fake_paint_single_mass_bin_run3)
    coordinator.paint_single_fstar(1, profiles)

    assert len(picked_groups_run3) > 0
    assert picked_groups_run3 != picked_groups_run1



def test_nearest_f_st_indices_maps_to_expected_grid_points():
    coordinator = make_coordinator(seed=111)
    f_st_grid = np.array([0.01, 0.05, 0.1, 0.2])
    sampled = np.array([0.011, 0.049, 0.08, 0.19])

    mapped = coordinator._nearest_f_st_indices(sampled, f_st_grid)

    np.testing.assert_array_equal(mapped, np.array([0, 1, 2, 3]))


def test_sample_f_st_for_halos_uniform_distribution_respects_bounds():
    coordinator = make_coordinator(seed=202)
    coordinator.parameters.source.f_st_paint_distribution = "uniform"
    coordinator.parameters.source.f_st_paint_min = 0.03
    coordinator.parameters.source.f_st_paint_max = 0.07

    rng = coordinator._make_f_st_rng(0)
    sampled = coordinator._sample_f_st_for_halos(1000, rng)

    assert np.all(sampled >= 0.03)
    assert np.all(sampled <= 0.07)
    assert np.unique(sampled).size > 10


def test_temporal_cube_append_works_after_reloading_manifest(tmp_path):
    params = Parameters()
    params.simulation.Ncell = 4
    params.simulation.Lbox = 10.0
    params.simulation.store_grids = ["Grid_xHII", "Grid_Temp", "Grid_xal"]
    output_dir = tmp_path / "output"
    cube = TemporalCube.create_empty(params, output_dir, snapshot_number=2)

    # Reopen from disk to reproduce the MPI master append path.
    reloaded = TemporalCube.read(file_path=cube._file_path)
    snapshot = CoevalCube(
        z=10.0,
        parameters=params,
        delta_b=np.zeros((4, 4, 4), dtype=float),
        Grid_Temp=np.ones((4, 4, 4), dtype=float),
        Grid_xHII=np.full((4, 4, 4), 0.25, dtype=float),
        Grid_xal=np.full((4, 4, 4), 2.0, dtype=float),
    )

    reloaded.append(snapshot, 0)

    snapshot_file = cube.snapshot_path(10.0)
    assert snapshot_file.exists()


def test_coeval_cube_to_arrays_makes_cached_cube_picklable(tmp_path):
    from beorn.io.handler import Handler

    params = Parameters()
    params.simulation.Ncell = 4
    params.simulation.Lbox = 10.0
    params.simulation.store_grids = ["Grid_xHII", "Grid_Temp", "Grid_xal"]

    handler = Handler(file_root=tmp_path)
    cube = CoevalCube(
        z=10.0,
        parameters=params,
        delta_b=np.zeros((4, 4, 4), dtype=float),
        Grid_Temp=np.ones((4, 4, 4), dtype=float),
        Grid_xHII=np.full((4, 4, 4), 0.25, dtype=float),
        Grid_xal=np.full((4, 4, 4), 2.0, dtype=float),
    )
    handler.write_file(params, cube, z_index=0)

    cached = handler.load_file(params, CoevalCube, z_index=0)
    cached.to_arrays()
    payload = pickle.dumps(cached)

    assert isinstance(payload, bytes)
    assert isinstance(cached.Grid_xHII, np.ndarray)


def test_sample_f_st_for_halos_normal_distribution_is_clipped():
    coordinator = make_coordinator(seed=303)
    coordinator.parameters.source.f_st_paint_distribution = "normal"
    coordinator.parameters.source.f_st = 0.1
    coordinator.parameters.source.f_st_paint_sigma = 10.0
    coordinator.parameters.source.f_st_paint_min = 0.02
    coordinator.parameters.source.f_st_paint_max = 0.15

    rng = coordinator._make_f_st_rng(0)
    sampled = coordinator._sample_f_st_for_halos(1000, rng)

    assert np.all(sampled >= 0.02)
    assert np.all(sampled <= 0.15)
    assert np.any(np.isclose(sampled, 0.02)) or np.any(np.isclose(sampled, 0.15))


def test_sample_f_st_for_halos_lognormal_distribution_is_clipped():
    coordinator = make_coordinator(seed=404)
    coordinator.parameters.source.f_st_paint_distribution = "lognormal"
    coordinator.parameters.source.f_st = 0.1
    coordinator.parameters.source.f_st_paint_sigma = 3.0
    coordinator.parameters.source.f_st_paint_min = 0.01
    coordinator.parameters.source.f_st_paint_max = 0.12

    rng = coordinator._make_f_st_rng(0)
    sampled = coordinator._sample_f_st_for_halos(1000, rng)

    assert np.all(sampled >= 0.01)
    assert np.all(sampled <= 0.12)
    assert np.any(np.isclose(sampled, 0.12))


def test_sample_f_st_for_halos_unknown_distribution_raises():
    coordinator = make_coordinator(seed=505)
    coordinator.parameters.source.f_st_paint_distribution = "not-a-distribution"

    rng = coordinator._make_f_st_rng(0)

    try:
        coordinator._sample_f_st_for_halos(5, rng)
    except ValueError as exc:
        assert "Unknown f_st painting distribution" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown f_st painting distribution")


def test_paint_single_fstar_mixed_mass_bins_paints_all_halos_once(monkeypatch):
    halo_catalog = DummyHaloCatalog(
        masses=np.array([2.0e8, 3.0e8, 4.0e8, 2.0e9, 3.0e9, 4.0e9]),
        alpha_vals=np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7]),
    )
    coordinator = make_coordinator(seed=606, halo_catalog=halo_catalog)
    profiles = make_profiles()

    painted_halo_ids = []

    def fake_profiles_of_halo_bin(self, z_index, alpha_index, mass_index, f_st_index):
        tag = float(100 * mass_index + 10 * alpha_index + f_st_index)
        return (
            np.array(tag),
            np.array([tag]),
            np.array([tag]),
        )

    def fake_paint_single_mass_bin(self, halo_catalog, z, radial_grid, r_lyal, profiles_of_bin, **kwargs):
        painted_halo_ids.extend(halo_catalog.selected.tolist())
        return (None, None, None)

    monkeypatch.setattr(RadiationProfilesFStarGrid, "profiles_of_halo_bin", fake_profiles_of_halo_bin)
    monkeypatch.setattr(PaintingCoordinator, "paint_single_mass_bin", fake_paint_single_mass_bin)
    monkeypatch.setattr("beorn.painting.coordinator.spreading_excess_fast", lambda parameters, grid: grid)
    monkeypatch.setattr("beorn.painting.coordinator.T_adiab_fluctu", lambda z, parameters, delta_b: np.zeros_like(delta_b))
    monkeypatch.setattr("beorn.painting.coordinator.S_alpha", lambda z, temp, neutral_frac: 1.0)

    coordinator.paint_single_fstar(0, profiles)

    np.testing.assert_array_equal(np.sort(np.asarray(painted_halo_ids)), np.arange(halo_catalog.size))


class DummyLegacyProfiles:
    def __init__(self):
        self.z_history = np.array([12.0, 10.0])
        self.halo_mass_bins = np.array(
            [
                [[1.0e8, 1.0e8]],
                [[1.0e9, 1.0e9]],
                [[1.0e10, 1.0e10]],
            ]
        )
        self.r_grid_cell = np.logspace(-2, 1, 3)
        self.r_lyal = np.logspace(-5, 2, 5)

    def profiles_of_halo_bin(self, z_index, alpha_index, mass_index):
        return (
            np.array(1.0),
            np.array([1.0]),
            np.array([1.0]),
        )



def test_paint_single_dispatches_fstar_and_legacy_paths(monkeypatch):
    coordinator = make_coordinator(seed=707)
    fstar_profiles = make_profiles()
    legacy_profiles = DummyLegacyProfiles()

    calls = []

    def fake_paint_single_fstar(self, z_index, profiles):
        calls.append(("fstar", z_index, type(profiles).__name__))
        return "fstar-result"

    def fake_legacy_paint_single_mass_bin(self, halo_catalog, z, radial_grid, r_lyal, profiles_of_bin, **kwargs):
        return (None, None, None)

    monkeypatch.setattr(PaintingCoordinator, "paint_single_fstar", fake_paint_single_fstar)
    monkeypatch.setattr(PaintingCoordinator, "paint_single_mass_bin", fake_legacy_paint_single_mass_bin)
    monkeypatch.setattr("beorn.painting.coordinator.spreading_excess_fast", lambda parameters, grid: grid)
    monkeypatch.setattr("beorn.painting.coordinator.T_adiab_fluctu", lambda z, parameters, delta_b: np.zeros_like(delta_b))
    monkeypatch.setattr("beorn.painting.coordinator.S_alpha", lambda z, temp, neutral_frac: 1.0)

    result_fstar = coordinator.paint_single(0, profiles=fstar_profiles)
    assert result_fstar == "fstar-result"
    assert calls == [("fstar", 0, "RadiationProfilesFStarGrid")]

    calls.clear()
    result_legacy = coordinator.paint_single(0, profiles=legacy_profiles)
    assert result_legacy is not None
    assert calls == []

def test_paint_cache_namespace_legacy_vs_fstar():
    coordinator = make_coordinator(seed=808)
    fstar_profiles = make_profiles()
    legacy_profiles = DummyLegacyProfiles()

    legacy_ns = coordinator._paint_cache_namespace(legacy_profiles)
    fstar_ns = coordinator._paint_cache_namespace(fstar_profiles)

    assert legacy_ns == "painted_output_legacy"
    assert fstar_ns.startswith("painted_output_fstar_dist_lognormal_sigma_")
    assert "seed_808" in fstar_ns


def test_paint_cache_namespace_changes_with_distribution_sigma_seed():
    coordinator = make_coordinator(seed=909)
    profiles = make_profiles()

    ns1 = coordinator._paint_cache_namespace(profiles)

    coordinator.parameters.source.f_st_paint_distribution = "uniform"
    coordinator.parameters.source.f_st_paint_sigma = 1.25
    coordinator.parameters.source.f_st_paint_seed = 123
    ns2 = coordinator._paint_cache_namespace(profiles)

    assert ns1 != ns2
    assert "dist_uniform" in ns2
    assert "sigma_1.25" in ns2
    assert "seed_123" in ns2


def test_paint_single_legacy_cache_uses_legacy_namespace():
    coordinator = make_coordinator(seed=1001)
    legacy_profiles = DummyLegacyProfiles()
    calls = {}

    class CachedCube:
        def to_arrays(self):
            return None

    def fake_load_file(parameters, cls, **kwargs):
        calls["kwargs"] = kwargs
        return CachedCube()

    coordinator.cache_handler = types.SimpleNamespace(
        load_file=fake_load_file,
        write_file=lambda *args, **kwargs: None,
    )

    result = coordinator.paint_single(0, profiles=legacy_profiles)
    assert isinstance(result, CachedCube)
    assert calls["kwargs"]["cache_namespace"] == "painted_output_legacy"


def test_paint_single_fstar_cache_uses_fstar_namespace():
    coordinator = make_coordinator(seed=1002)
    profiles = make_profiles()
    calls = {}

    class CachedCube:
        def to_arrays(self):
            return None

    def fake_load_file(parameters, cls, **kwargs):
        calls["kwargs"] = kwargs
        return CachedCube()

    coordinator.cache_handler = types.SimpleNamespace(
        load_file=fake_load_file,
        write_file=lambda *args, **kwargs: None,
    )

    result = coordinator.paint_single_fstar(0, profiles)
    assert isinstance(result, CachedCube)
    assert calls["kwargs"]["cache_namespace"] == coordinator._paint_cache_namespace(profiles)


def test_paint_single_fstar_profiles_path_uses_nearest_profile_redshift(tmp_path, monkeypatch):
    coordinator = make_coordinator(seed=1003)
    coordinator.loader.redshifts = np.array([15.0, 10.0])

    profiles_path = tmp_path / "fstar_profiles.h5"
    z_history = np.array([18.0, 15.0, 12.0, 9.0])
    halo_mass_bins = np.zeros((3, 1, 4), dtype=float)
    halo_mass_bins[..., 0] = 100.0
    halo_mass_bins[..., 1] = 200.0
    halo_mass_bins[..., 2] = 300.0
    halo_mass_bins[..., 3] = 400.0

    with h5py.File(profiles_path, "w") as h5:
        h5.create_dataset("r_grid_cell", data=np.logspace(-2, 1, 3))
        h5.create_dataset("r_lyal", data=np.logspace(-5, 2, 5))
        h5.create_dataset("f_st_grid", data=np.array([0.01, 0.05]))
        h5.create_dataset("z_history", data=z_history)
        h5.create_dataset("halo_mass_bins", data=halo_mass_bins)
        h5.create_dataset("R_bubble", data=np.ones((2, 1, 2, 4)))
        h5.create_dataset("rho_xray", data=np.ones((3, 2, 1, 2, 4)))
        h5.create_dataset("rho_heat", data=np.ones((3, 2, 1, 2, 4)))
        h5.create_dataset("rho_alpha", data=np.ones((5, 2, 1, 2, 4)))

    captured = {}

    def fake_paint_single_fstar(self, z_index, profiles):
        captured["z_index"] = z_index
        captured["z_history"] = profiles.z_history.copy()
        captured["halo_mass_bins"] = profiles.halo_mass_bins.copy()
        return "fstar-from-path"

    monkeypatch.setattr(PaintingCoordinator, "paint_single_fstar", fake_paint_single_fstar)

    result = coordinator.paint_single(1, profiles_path=profiles_path)

    assert result == "fstar-from-path"
    assert captured["z_index"] == 1
    np.testing.assert_allclose(captured["z_history"], np.array([9.0]))
    np.testing.assert_allclose(captured["halo_mass_bins"], halo_mass_bins[..., 3][..., None])



# Integration test for Handler cache namespace isolation
def test_handler_cache_namespace_isolation_roundtrip(tmp_path):
    # This is a tiny integration test for the Handler cache namespace feature.
    # It writes two objects with the same logical name into two namespaces and
    # verifies that they do not collide and that each can be read back.
    from beorn.io.handler import Handler

    class DummyCached:
        def __init__(self, payload: str):
            self.payload = payload

        def write(self, directory, parameters, **kwargs):
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "dummy_cache.txt"
            path.write_text(self.payload)

        @classmethod
        def read(cls, directory, parameters, **kwargs):
            path = directory / "dummy_cache.txt"
            if not path.exists():
                raise FileNotFoundError(path)
            return cls(path.read_text())

    handler = Handler(file_root=tmp_path)
    params = SimpleNamespace()  # not used by DummyCached

    handler.write_file(params, DummyCached("legacy"), cache_namespace="painted_output_legacy")
    handler.write_file(params, DummyCached("fstar"), cache_namespace="painted_output_fstar_dist_lognormal_sigma_0.4_seed_123")

    legacy = handler.load_file(params, DummyCached, cache_namespace="painted_output_legacy")
    fstar = handler.load_file(params, DummyCached, cache_namespace="painted_output_fstar_dist_lognormal_sigma_0.4_seed_123")

    assert legacy.payload == "legacy"
    assert fstar.payload == "fstar"

    # Ensure isolation on disk.
    assert (tmp_path / "painted_output_legacy" / "dummy_cache.txt").exists()
    assert (tmp_path / "painted_output_fstar_dist_lognormal_sigma_0.4_seed_123" / "dummy_cache.txt").exists()

    # Ensure no collision: contents differ.
    assert (tmp_path / "painted_output_legacy" / "dummy_cache.txt").read_text() != (
        tmp_path / "painted_output_fstar_dist_lognormal_sigma_0.4_seed_123" / "dummy_cache.txt"
    ).read_text()



# New end-to-end cache roundtrip test for stochastic f_st painting
def test_paint_single_fstar_end_to_end_cache_roundtrip(tmp_path, monkeypatch):
    """Tiny end-to-end workflow test for stochastic f_st painting with real cache IO.

    First call should compute and write a cached coeval cube. Second call should
    load from cache without invoking the painting kernel again.
    """
    from beorn.io.handler import Handler
    from beorn.structs.coeval_cube import CoevalCube

    halo_catalog = DummyHaloCatalog(
        masses=np.array([2.0e8, 2.5e8, 3.0e8, 3.5e8]),
        alpha_vals=np.array([0.7, 0.7, 0.7, 0.7]),
    )
    params = make_parameters(seed=4242)
    loader = DummyLoader(halo_catalog)
    cache_handler = Handler(file_root=tmp_path)
    coordinator = PaintingCoordinator(params, loader, DummyHandler(), cache_handler=cache_handler)
    profiles = make_profiles()

    paint_calls = {"count": 0}

    def fake_paint_single_mass_bin(self, halo_catalog, z, radial_grid, r_lyal, profiles_of_bin, **kwargs):
        paint_calls["count"] += 1
        shape = (4, 4, 4)
        fa = np.fft.rfftn(np.ones(shape))
        return (fa, 3.0 * fa, 2.0 * fa)

    monkeypatch.setattr(PaintingCoordinator, "paint_single_mass_bin", fake_paint_single_mass_bin)
    monkeypatch.setattr("beorn.painting.coordinator.spreading_excess_fast", lambda parameters, grid: grid)
    monkeypatch.setattr("beorn.painting.coordinator.T_adiab_fluctu", lambda z, parameters, delta_b: np.zeros_like(delta_b))
    monkeypatch.setattr("beorn.painting.coordinator.S_alpha", lambda z, temp, neutral_frac: 1.0)

        # Patch CoevalCube IO for this integration test: we only need to confirm
    # that caching round-trips the painted grids and respects namespaces.
    def _test_cube_write(self, directory, parameters, **kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        z_index = kwargs.get("z_index", 0)
        path = directory / f"CoevalCube_{parameters.unique_hash()}_z{z_index}.npz"
        np.savez(
            path,
            z=float(self.z),
            Grid_xHII=self.Grid_xHII,
            Grid_Temp=self.Grid_Temp,
            Grid_xal=self.Grid_xal,
        )

    @classmethod
    def _test_cube_read(cls, directory, parameters, **kwargs):
        z_index = kwargs.get("z_index", 0)
        path = directory / f"CoevalCube_{parameters.unique_hash()}_z{z_index}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        # minimal object with the fields used by the coordinator/tests
        return cls(
            parameters=parameters,
            z=float(data["z"]),
            delta_b=np.zeros((4, 4, 4), dtype=float),
            Grid_Temp=data["Grid_Temp"],
            Grid_xHII=data["Grid_xHII"],
            Grid_xal=data["Grid_xal"],
        )

    monkeypatch.setattr(CoevalCube, "write", _test_cube_write, raising=True)
    monkeypatch.setattr(CoevalCube, "read", _test_cube_read, raising=True)


    # First run computes and writes to cache.
    cube1 = coordinator.paint_single_fstar(0, profiles)
    assert paint_calls["count"] > 0
    n_paint_groups = paint_calls["count"]
    np.testing.assert_allclose(cube1.Grid_xHII, float(n_paint_groups) * np.ones((4, 4, 4)))
    np.testing.assert_allclose(cube1.Grid_Temp, 2.0 * float(n_paint_groups) * np.ones((4, 4, 4)))
    np.testing.assert_allclose(cube1.Grid_xal, 3.0 * float(n_paint_groups) / (4 * np.pi) * np.ones((4, 4, 4)))

    namespace_dir = tmp_path / coordinator._paint_cache_namespace(profiles)
    assert namespace_dir.exists()
    assert any(namespace_dir.iterdir())

    # Second run should load from cache and avoid repainting.
    paint_calls_before = paint_calls["count"]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("paint_single_mass_bin should not be called when loading painted output from cache")

    monkeypatch.setattr(PaintingCoordinator, "paint_single_mass_bin", fail_if_called)

    cube2 = coordinator.paint_single_fstar(0, profiles)
    assert paint_calls["count"] == paint_calls_before
    np.testing.assert_allclose(cube2.Grid_xHII, cube1.Grid_xHII)
    np.testing.assert_allclose(cube2.Grid_Temp, cube1.Grid_Temp)
    np.testing.assert_allclose(cube2.Grid_xal, cube1.Grid_xal)
    assert cube2.z == cube1.z



# MPI integration test for painted-output caching in the stochastic f_st path.
def test_mpi_paint_single_fstar_cache_roundtrip(tmp_path, monkeypatch):
    """MPI integration test for painted-output caching in the stochastic f_st path.

    Run with:
        mpirun -n 2 pytest tests/test_coordinator.py::test_mpi_paint_single_fstar_cache_roundtrip -q -s

    This test matches the current coordinator behavior by orchestrating the MPI
    sequence externally:
      - rank 0 paints and writes the cached output
      - all ranks synchronize
      - rank 1 loads from the shared cache and must not repaint
    """
    pytest.importorskip("mpi4py")
    from mpi4py import MPI
    from pathlib import Path
    from beorn.io.handler import Handler
    from beorn.structs.coeval_cube import CoevalCube

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() < 2:
        pytest.skip("MPI cache roundtrip test requires at least 2 ranks")

    # Under mpirun, each rank runs its own pytest process, so tmp_path may differ.
    # Create one shared cache directory on rank 0 and broadcast it.
    if rank == 0:
        shared_root = tmp_path / "shared_mpi_paint_cache"
        shared_root.mkdir(parents=True, exist_ok=True)
        shared_root_str = str(shared_root)
    else:
        shared_root_str = None
    shared_root_str = comm.bcast(shared_root_str, root=0)

    halo_catalog = DummyHaloCatalog(
        masses=np.array([2.0e8, 2.5e8, 3.0e8, 3.5e8]),
        alpha_vals=np.array([0.7, 0.7, 0.7, 0.7]),
    )
    params = make_parameters(seed=5151)
    loader = DummyLoader(halo_catalog)
    cache_handler = Handler(file_root=Path(shared_root_str))
    coordinator = PaintingCoordinator(params, loader, DummyHandler(), cache_handler=cache_handler)
    profiles = make_profiles()

    paint_calls = {"count": 0}

    def fake_paint_single_mass_bin(self, halo_catalog, z, radial_grid, r_lyal, profiles_of_bin, **kwargs):
        paint_calls["count"] += 1
        shape = (4, 4, 4)
        fa = np.fft.rfftn(np.ones(shape))
        return (fa, 3.0 * fa, 2.0 * fa)

    monkeypatch.setattr(PaintingCoordinator, "paint_single_mass_bin", fake_paint_single_mass_bin)
    monkeypatch.setattr("beorn.painting.coordinator.spreading_excess_fast", lambda parameters, grid: grid)
    monkeypatch.setattr("beorn.painting.coordinator.T_adiab_fluctu", lambda z, parameters, delta_b: np.zeros_like(delta_b))
    monkeypatch.setattr("beorn.painting.coordinator.S_alpha", lambda z, temp, neutral_frac: 1.0)

    # Patch CoevalCube IO for this integration test to avoid depending on full
    # BaseStruct / Parameters serialization while still using the real Handler.
    def _test_cube_write(self, directory, parameters, **kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        z_index = kwargs.get("z_index", 0)
        path = directory / f"CoevalCube_{parameters.unique_hash()}_z{z_index}.npz"
        np.savez(
            path,
            z=float(self.z),
            Grid_xHII=self.Grid_xHII,
            Grid_Temp=self.Grid_Temp,
            Grid_xal=self.Grid_xal,
        )

    @classmethod
    def _test_cube_read(cls, directory, parameters, **kwargs):
        z_index = kwargs.get("z_index", 0)
        path = directory / f"CoevalCube_{parameters.unique_hash()}_z{z_index}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        return cls(
            parameters=parameters,
            z=float(data["z"]),
            delta_b=np.zeros((4, 4, 4), dtype=float),
            Grid_Temp=data["Grid_Temp"],
            Grid_xHII=data["Grid_xHII"],
            Grid_xal=data["Grid_xal"],
        )

    monkeypatch.setattr(CoevalCube, "write", _test_cube_write, raising=True)
    monkeypatch.setattr(CoevalCube, "read", _test_cube_read, raising=True)

    if rank == 0:
        cube = coordinator.paint_single_fstar(0, profiles)
        assert paint_calls["count"] > 0
        n_paint_groups = paint_calls["count"]
        np.testing.assert_allclose(cube.Grid_xHII, float(n_paint_groups) * np.ones((4, 4, 4)))
        np.testing.assert_allclose(cube.Grid_Temp, 2.0 * float(n_paint_groups) * np.ones((4, 4, 4)))
        np.testing.assert_allclose(cube.Grid_xal, 3.0 * float(n_paint_groups) / (4 * np.pi) * np.ones((4, 4, 4)))

    # Ensure rank 0 finished writing before rank 1 tries to load.
    comm.Barrier()

    if rank == 1:
        def fail_if_called(*args, **kwargs):
            raise AssertionError("rank 1 should load painted output from cache without repainting")

        monkeypatch.setattr(PaintingCoordinator, "paint_single_mass_bin", fail_if_called)
        cube = coordinator.paint_single_fstar(0, profiles)
        assert paint_calls["count"] == 0

    namespace_dir = Path(shared_root_str) / coordinator._paint_cache_namespace(profiles)
    assert namespace_dir.exists()

    payload = (
        float(np.sum(cube.Grid_xHII)),
        float(np.sum(cube.Grid_Temp)),
        float(np.sum(cube.Grid_xal)),
        float(cube.z),
    )
    gathered = comm.gather(payload, root=0)
    if rank == 0:
        assert len(gathered) == 2
        assert gathered[0] == gathered[1]
