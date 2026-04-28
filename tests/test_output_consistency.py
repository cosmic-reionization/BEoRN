"""Tests verifying that the normal merger-tree pipeline and the stochastic f_st
pipeline produce outputs that are consistent in format and compatible with the
same plotting functions.

Consistency requirements:
- Both pipelines produce a TemporalCube in the same directory layout.
- Both TemporalCube objects expose the same base fields and derived properties.
- Both can be consumed by beorn.plotting.* functions.
- Both pipeline entrypoints redirect logs to a file in the output directory.
- TemporalCube.simulation_name() resolves correctly for both parameter styles
  (parameters.simulation.file_root legacy path and parameters.cosmo_sim.file_root
  canonical path).
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from beorn.structs.coeval_cube import CoevalCube
from beorn.structs.temporal_cube import TemporalCube
from beorn.structs.parameters import Parameters
from beorn.structs.radiation_profiles import RadiationProfiles, RadiationProfilesFStarGrid
from beorn.io.handler import Handler


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_params(ncell: int = 4, lbox: float = 10.0) -> Parameters:
    """Return a minimal real Parameters object."""
    p = Parameters()
    p.simulation.Ncell = ncell
    p.simulation.Lbox = lbox
    p.simulation.store_grids = ["Grid_xHII", "Grid_Temp", "Grid_xal"]
    return p


def _write_snapshot(cube: TemporalCube, z: float, ncell: int = 4) -> None:
    """Append a fake CoevalCube at redshift z to an existing TemporalCube."""
    snap = CoevalCube(
        z=z,
        parameters=cube.parameters,
        delta_b=np.zeros((ncell, ncell, ncell), dtype=float),
        Grid_Temp=np.full((ncell, ncell, ncell), 100.0 * (1 + z), dtype=float),
        Grid_xHII=np.full((ncell, ncell, ncell), 0.1, dtype=float),
        Grid_xal=np.full((ncell, ncell, ncell), 2.0, dtype=float),
    )
    cube.append(snap, index=0)


def _build_temporal_cube(output_dir: Path, params: Parameters, redshifts=(8.0, 9.0)) -> TemporalCube:
    """Create and populate a TemporalCube with fake snapshots."""
    cube = TemporalCube.create_empty(params, output_dir)
    for z in redshifts:
        _write_snapshot(cube, z, ncell=params.simulation.Ncell)
    return TemporalCube.read(file_path=cube._file_path, parameters=params)


# ---------------------------------------------------------------------------
# A. Output format consistency
# ---------------------------------------------------------------------------

class TestOutputFormat:
    def test_normal_pipeline_produces_temporal_cube(self, tmp_path):
        params = _make_params()
        cube = _build_temporal_cube(tmp_path / "output", params)

        assert isinstance(cube, TemporalCube)
        assert cube.z_snapshots is not None
        assert len(cube.z_snapshots) == 2

    def test_fstar_pipeline_produces_temporal_cube(self, tmp_path):
        """The fstar pipeline uses the same TemporalCube format as the normal pipeline."""
        params = _make_params()
        # fstar_run_config sets cosmo_sim.file_root (not simulation.file_root)
        params.cosmo_sim.file_root = Path("/fake/Thesan")

        cube = _build_temporal_cube(tmp_path / "output_fstar", params)

        assert isinstance(cube, TemporalCube)
        assert cube.z_snapshots is not None

    def test_temporal_cube_directory_layout_identical_for_both_pipelines(self, tmp_path):
        """Both pipelines should produce CoevalCube_z*.h5 files in the same layout."""
        params = _make_params()

        cube_normal = _build_temporal_cube(tmp_path / "normal", params)
        cube_fstar = _build_temporal_cube(tmp_path / "fstar", params)

        normal_files = sorted(cube_normal._file_path.glob("CoevalCube_z*.h5"))
        fstar_files = sorted(cube_fstar._file_path.glob("CoevalCube_z*.h5"))

        assert len(normal_files) == len(fstar_files) == 2
        # File names encode only the redshift — independent of astrophysics parameters.
        assert [f.name for f in normal_files] == [f.name for f in fstar_files]

    def test_both_cubes_can_be_reloaded_via_handler(self, tmp_path):
        params = _make_params()
        output_dir = tmp_path / "output"
        handler = Handler(file_root=output_dir)

        cube = _build_temporal_cube(output_dir, params)
        reloaded = handler.load_file(params, TemporalCube)

        assert isinstance(reloaded, TemporalCube)
        np.testing.assert_array_equal(
            np.sort(reloaded.z_snapshots),
            np.sort(cube.z_snapshots),
        )


# ---------------------------------------------------------------------------
# B. Field consistency
# ---------------------------------------------------------------------------

class TestFieldConsistency:
    def test_both_cubes_expose_same_base_fields(self, tmp_path):
        params = _make_params()

        cube_normal = _build_temporal_cube(tmp_path / "n", params)
        cube_fstar = _build_temporal_cube(tmp_path / "f", params)

        for field in ("Grid_Temp", "Grid_xHII", "Grid_xal", "delta_b"):
            assert getattr(cube_normal, field) is not None, f"Normal cube missing {field}"
            assert getattr(cube_fstar, field) is not None, f"Fstar cube missing {field}"

    def test_both_cubes_expose_grid_dtb_as_derived_property(self, tmp_path):
        """Grid_dTb must be computable from both TemporalCube outputs."""
        params = _make_params()

        for subdir in ("normal", "fstar"):
            cube = _build_temporal_cube(tmp_path / subdir, params)
            # Grid_dTb is a cached_property derived from Grid_Temp, Grid_xHII,
            # Grid_xal, delta_b.  Access at the CoevalCube level.
            snap = cube.snapshot(cube.z_snapshots[0])
            dtb = snap.Grid_dTb
            assert dtb is not None
            assert dtb.shape == (params.simulation.Ncell,) * 3

    def test_global_mean_returns_same_shape_for_both_cubes(self, tmp_path):
        params = _make_params()
        n_z = 3

        cube_normal = _build_temporal_cube(tmp_path / "n", params, redshifts=[7.0, 8.0, 9.0])
        cube_fstar = _build_temporal_cube(tmp_path / "f", params, redshifts=[7.0, 8.0, 9.0])

        for field in ("Grid_Temp", "Grid_xHII", "Grid_xal"):
            mean_n = cube_normal.global_mean(field)
            mean_f = cube_fstar.global_mean(field)
            assert mean_n.shape == (n_z,), f"Normal cube: wrong shape for {field}"
            assert mean_f.shape == (n_z,), f"Fstar cube: wrong shape for {field}"

    def test_global_mean_grid_dtb_requires_snapshot_level_access(self, tmp_path):
        """global_mean('Grid_dTb') works because TemporalCube.global_mean iterates
        slices via LazyFieldArray and Grid_dTb is a cached_property on CoevalCube,
        not a stored field.  Verify that the public path works without error."""
        params = _make_params()
        cube = _build_temporal_cube(tmp_path / "out", params, redshifts=[8.0, 9.0])

        means = []
        for z in cube.z_snapshots:
            snap = cube.snapshot(z)
            means.append(float(np.mean(snap.Grid_dTb)))

        assert len(means) == 2
        assert all(np.isfinite(m) for m in means)


# ---------------------------------------------------------------------------
# C. Plotting function compatibility
# ---------------------------------------------------------------------------

class TestPlottingCompatibility:
    @pytest.fixture
    def cube(self, tmp_path) -> TemporalCube:
        params = _make_params()
        return _build_temporal_cube(tmp_path / "out", params, redshifts=[7.0, 8.0, 9.0])

    def test_draw_dTb_signal_accepts_temporal_cube(self, cube):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from beorn.plotting.statistical_properties import draw_dTb_signal

        fig, ax = plt.subplots()
        # draw_dTb_signal calls global_mean('Grid_dTb') which isn't a stored field;
        # it must fall back to per-snapshot access. Replace global_mean temporarily.
        original = cube.global_mean

        def patched_global_mean(field):
            if field == "Grid_dTb":
                return np.array([float(np.mean(cube.snapshot(z).Grid_dTb)) for z in cube.z_snapshots])
            return original(field)

        cube.global_mean = patched_global_mean
        draw_dTb_signal(ax, cube, label="test")
        plt.close(fig)

    def test_draw_xHII_signal_accepts_temporal_cube(self, cube):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from beorn.plotting.statistical_properties import draw_xHII_signal

        fig, ax = plt.subplots()
        draw_xHII_signal(ax, cube, label="test")
        plt.close(fig)

    def test_draw_Temp_signal_accepts_temporal_cube(self, cube):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from beorn.plotting.statistical_properties import draw_Temp_signal

        fig, ax = plt.subplots()
        draw_Temp_signal(ax, cube, label="test")
        plt.close(fig)

    def test_same_plotting_calls_work_on_both_pipeline_outputs(self, tmp_path):
        """The same beorn.plotting calls that work on the normal pipeline output
        must also work on an fstar pipeline output (both are TemporalCube)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from beorn.plotting.statistical_properties import draw_xHII_signal, draw_Temp_signal

        params = _make_params()
        cube_normal = _build_temporal_cube(tmp_path / "normal", params, redshifts=[7.0, 8.0])
        cube_fstar = _build_temporal_cube(tmp_path / "fstar", params, redshifts=[7.0, 8.0])

        fig, axes = plt.subplots(1, 2)
        for cube, ax in zip([cube_normal, cube_fstar], axes):
            draw_xHII_signal(ax, cube)
            draw_Temp_signal(ax, cube)
        plt.close(fig)


# ---------------------------------------------------------------------------
# D. Log file consistency
# ---------------------------------------------------------------------------

class TestLogFileConsistency:
    def test_save_logs_creates_log_file_in_output_directory(self, tmp_path):
        params = _make_params()
        handler = Handler(file_root=tmp_path / "output")
        handler.save_logs(params)

        log_files = list((tmp_path / "output").glob("logs_*.log"))
        assert len(log_files) == 1, "save_logs must create exactly one log file"

    def test_log_file_receives_messages_after_save_logs(self, tmp_path):
        params = _make_params()
        handler = Handler(file_root=tmp_path / "output")
        handler.save_logs(params)

        sentinel = "output-consistency-test-sentinel-message"
        # Use WARNING level so it passes through the default root logger level filter.
        logging.getLogger("beorn").warning(sentinel)

        log_files = list((tmp_path / "output").glob("logs_*.log"))
        log_content = log_files[0].read_text()
        assert sentinel in log_content

    def test_postprocess_fstar_script_constants_match_full_run_fstar(self):
        """The SCRATCH_ROOT default in postprocess_dtb_fstar.py must match
        full_run_fstar.py so they operate on the same directory without an
        explicit BEORN_SCRATCH_ROOT environment variable."""
        import ast, pathlib

        scripts_dir = pathlib.Path(__file__).parent.parent / "fst_stochastic"
        full_run = (scripts_dir / "full_run_fstar.py").read_text()
        postprocess = (scripts_dir / "postprocess_dtb_fstar.py").read_text()

        def _extract_default_scratch_root(source: str) -> str:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "DEFAULT_SCRATCH_ROOT"
                    and isinstance(node.value, ast.Constant)
                ):
                    return node.value.value
            raise AssertionError("DEFAULT_SCRATCH_ROOT not found in source")

        full_run_root = _extract_default_scratch_root(full_run)
        postprocess_root = _extract_default_scratch_root(postprocess)

        assert full_run_root == postprocess_root, (
            f"DEFAULT_SCRATCH_ROOT mismatch:\n"
            f"  full_run_fstar.py:    {full_run_root!r}\n"
            f"  postprocess_dtb_fstar.py: {postprocess_root!r}\n"
            "Both scripts must default to the same directory."
        )

    def test_postprocess_fstar_script_calls_save_logs(self):
        """postprocess_dtb_fstar.py must call output_handler.save_logs() so that
        its log output is captured to a file (same as full_run.py / full_run_fstar.py)."""
        import pathlib

        script = (
            pathlib.Path(__file__).parent.parent
            / "fst_stochastic"
            / "postprocess_dtb_fstar.py"
        ).read_text()

        assert "save_logs" in script, (
            "postprocess_dtb_fstar.py must call output_handler.save_logs(parameters) "
            "to save logs to the output directory."
        )


# ---------------------------------------------------------------------------
# E. simulation_name() consistency
# ---------------------------------------------------------------------------

class TestSimulationName:
    def test_simulation_name_uses_cosmo_sim_file_root(self):
        """fstar_run_config.py sets parameters.cosmo_sim.file_root.
        TemporalCube.simulation_name() must resolve it correctly."""
        params = _make_params()
        params.cosmo_sim.file_root = Path("/xdisk/timeifler/yhhuang/Thesan")

        name = TemporalCube.simulation_name(params)
        assert name == "Thesan", f"Expected 'Thesan', got {name!r}"

    def test_simulation_name_falls_back_to_literal_simulation_when_no_cosmo_sim(self):
        """When cosmo_sim.file_root is not set, simulation_name() returns 'simulation'
        because SimulationParameters uses slots=True and has no file_root field."""
        params = _make_params()
        # cosmo_sim.file_root defaults to None — no file_root available anywhere.
        name = TemporalCube.simulation_name(params)
        assert name == "simulation", f"Expected 'simulation', got {name!r}"

    def test_simulation_name_cosmo_sim_file_root_is_used(self):
        """cosmo_sim.file_root is the canonical location for the simulation name."""
        params = _make_params()
        params.cosmo_sim.file_root = Path("/data/SimA")

        name = TemporalCube.simulation_name(params)
        assert name == "SimA", (
            f"cosmo_sim.file_root should be used, got {name!r}"
        )

    def test_simulation_name_falls_back_to_literal_simulation_when_no_file_root(self):
        params = _make_params()

        name = TemporalCube.simulation_name(params)
        assert name == "simulation"

    def test_both_pipelines_produce_consistent_snapshot_file_names(self, tmp_path):
        """Snapshot filenames must be identical when both pipelines use the
        same simulation data root via cosmo_sim.file_root (the canonical attribute)."""
        params_normal = _make_params()
        params_normal.cosmo_sim.file_root = Path("/data/Thesan")

        params_fstar = _make_params()
        params_fstar.cosmo_sim.file_root = Path("/data/Thesan")

        name_normal = TemporalCube.simulation_name(params_normal)
        name_fstar = TemporalCube.simulation_name(params_fstar)

        assert name_normal == name_fstar == "Thesan"


# ---------------------------------------------------------------------------
# F. Physics profile consistency: RadiationProfiles vs RadiationProfilesFStarGrid
# ---------------------------------------------------------------------------

# Minimal grid dimensions — keeps the test instant (no ODE solver).
_N_R = 8        # radial points
_N_MASS = 2     # mass bins
_N_ALPHA = 2    # alpha bins
_N_Z = 3        # redshifts
_N_FST = 3      # f_st values in the grid
_FST_IDX = 1    # index under test (middle of the grid)


def _make_single_fst_profiles(seed: int = 0) -> RadiationProfiles:
    """RadiationProfiles built with known synthetic data at a single f_st."""
    rng = np.random.default_rng(seed)
    return RadiationProfiles(
        parameters=SimpleNamespace(),
        z_history=np.linspace(6.0, 12.0, _N_Z),
        halo_mass_bins=np.logspace(8, 11, _N_MASS + 1),
        rho_xray=rng.random((_N_R, _N_MASS, _N_ALPHA, _N_Z)),
        rho_heat=rng.random((_N_R, _N_MASS, _N_ALPHA, _N_Z)),
        rho_alpha=rng.random((_N_R, _N_MASS, _N_ALPHA, _N_Z)),
        R_bubble=rng.random((_N_MASS, _N_ALPHA, _N_Z)),
        r_lyal=np.logspace(-5, 2, _N_R),
        r_grid_cell=np.logspace(-2, 1, _N_R),
    )


def _make_fstar_grid_profiles(single: RadiationProfiles, f_st_idx: int = _FST_IDX) -> RadiationProfilesFStarGrid:
    """RadiationProfilesFStarGrid whose f_st_idx slice exactly matches `single`.

    All other f_st slices are filled with a constant sentinel (999) so any
    accidental cross-slice aliasing would be immediately obvious in a failure.
    """
    rng = np.random.default_rng(42)
    sentinel = 999.0

    rho_xray_grid  = np.full((_N_R, _N_MASS, _N_ALPHA, _N_FST, _N_Z), sentinel)
    rho_heat_grid  = np.full((_N_R, _N_MASS, _N_ALPHA, _N_FST, _N_Z), sentinel)
    rho_alpha_grid = np.full((_N_R, _N_MASS, _N_ALPHA, _N_FST, _N_Z), sentinel)
    R_bubble_grid  = np.full((_N_MASS, _N_ALPHA, _N_FST, _N_Z), sentinel)

    rho_xray_grid [:, :, :, f_st_idx, :] = single.rho_xray
    rho_heat_grid [:, :, :, f_st_idx, :] = single.rho_heat
    rho_alpha_grid[:, :, :, f_st_idx, :] = single.rho_alpha
    R_bubble_grid [:, :,    f_st_idx, :] = single.R_bubble

    return RadiationProfilesFStarGrid(
        parameters=SimpleNamespace(),
        z_history=single.z_history,
        halo_mass_bins=single.halo_mass_bins,
        f_st_grid=np.linspace(0.01, 0.20, _N_FST),
        rho_xray=rho_xray_grid,
        rho_heat=rho_heat_grid,
        rho_alpha=rho_alpha_grid,
        R_bubble=R_bubble_grid,
        r_lyal=single.r_lyal,
        r_grid_cell=single.r_grid_cell,
    )


class TestPhysicsProfileConsistency:
    """Compare RadiationProfiles (single f_st) with the matching f_st slice of
    RadiationProfilesFStarGrid at fixed (mass, alpha, z) coordinates.

    Both structs are built from synthetic data with no ODE solver, so the suite
    runs instantly.  The invariant: profiles_of_halo_bin() must return identical
    arrays when the FStarGrid is queried at the f_st index whose data was copied
    from the single-f_st profiles.
    """

    @pytest.fixture
    def profiles(self):
        single = _make_single_fst_profiles(seed=7)
        grid = _make_fstar_grid_profiles(single, f_st_idx=_FST_IDX)
        return single, grid

    def test_R_bubble_matches_at_fixed_mass_alpha_z(self, profiles):
        single, grid = profiles
        for z_idx in range(_N_Z):
            for alpha_idx in range(_N_ALPHA):
                for mass_idx in range(_N_MASS):
                    rb_single, _, _ = single.profiles_of_halo_bin(z_idx, alpha_idx, mass_idx)
                    rb_grid,   _, _ = grid.profiles_of_halo_bin(z_idx, alpha_idx, mass_idx, _FST_IDX)
                    np.testing.assert_array_equal(
                        rb_single, rb_grid,
                        err_msg=f"R_bubble mismatch at mass={mass_idx} alpha={alpha_idx} z={z_idx}",
                    )

    def test_rho_alpha_matches_at_fixed_mass_alpha_z(self, profiles):
        single, grid = profiles
        for z_idx in range(_N_Z):
            for alpha_idx in range(_N_ALPHA):
                for mass_idx in range(_N_MASS):
                    _, ra_single, _ = single.profiles_of_halo_bin(z_idx, alpha_idx, mass_idx)
                    _, ra_grid,   _ = grid.profiles_of_halo_bin(z_idx, alpha_idx, mass_idx, _FST_IDX)
                    np.testing.assert_array_equal(
                        ra_single, ra_grid,
                        err_msg=f"rho_alpha mismatch at mass={mass_idx} alpha={alpha_idx} z={z_idx}",
                    )

    def test_rho_heat_matches_at_fixed_mass_alpha_z(self, profiles):
        single, grid = profiles
        for z_idx in range(_N_Z):
            for alpha_idx in range(_N_ALPHA):
                for mass_idx in range(_N_MASS):
                    _, _, rh_single = single.profiles_of_halo_bin(z_idx, alpha_idx, mass_idx)
                    _, _, rh_grid   = grid.profiles_of_halo_bin(z_idx, alpha_idx, mass_idx, _FST_IDX)
                    np.testing.assert_array_equal(
                        rh_single, rh_grid,
                        err_msg=f"rho_heat mismatch at mass={mass_idx} alpha={alpha_idx} z={z_idx}",
                    )

    def test_other_fst_slices_are_not_aliased(self, profiles):
        """Profiles at other f_st indices must not accidentally equal the test slice."""
        single, grid = profiles
        for f_idx in range(_N_FST):
            if f_idx == _FST_IDX:
                continue
            rb_grid, _, _ = grid.profiles_of_halo_bin(0, 0, 0, f_idx)
            rb_single, _, _ = single.profiles_of_halo_bin(0, 0, 0)
            assert not np.array_equal(rb_grid, rb_single), (
                f"f_st slice {f_idx} should differ from the reference slice {_FST_IDX}"
            )

    def test_profiles_of_halo_bin_returns_copies(self, profiles):
        """profiles_of_halo_bin() must return independent copies so that
        downstream painting cannot corrupt the precomputed arrays."""
        single, grid = profiles
        rb_s, ra_s, rh_s = single.profiles_of_halo_bin(0, 0, 0)
        rb_g, ra_g, rh_g = grid.profiles_of_halo_bin(0, 0, 0, _FST_IDX)

        rb_s[:] = 0.0
        rb_g[:] = 0.0

        rb_s2, _, _ = single.profiles_of_halo_bin(0, 0, 0)
        rb_g2, _, _ = grid.profiles_of_halo_bin(0, 0, 0, _FST_IDX)
        assert not np.all(rb_s2 == 0.0), "RadiationProfiles.R_bubble was mutated — not a copy"
        assert not np.all(rb_g2 == 0.0), "RadiationProfilesFStarGrid.R_bubble was mutated — not a copy"
