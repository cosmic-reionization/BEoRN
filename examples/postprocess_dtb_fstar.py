from pathlib import Path
import logging
import os

import h5py
import numpy as np

import beorn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_SCRATCH_ROOT = "/xdisk/timeifler/yhhuang/BEoRN-v2/coarse_grid_test_merged/"
DEFAULT_FILE_ROOT = "/xdisk/timeifler/yhhuang/Thesan/"
DEFAULT_TREE_CACHE_FILE = "/xdisk/timeifler/yhhuang/Thesan/postprocessing/trees/LHaloTree/tree_cache_v2.hdf5"

SCRATCH_ROOT = Path(os.environ.get("BEORN_SCRATCH_ROOT", DEFAULT_SCRATCH_ROOT))
FILE_ROOT = Path(os.environ.get("BEORN_FILE_ROOT", DEFAULT_FILE_ROOT))
TREE_CACHE_FILE = Path(os.environ.get("BEORN_TREE_CACHE_FILE", DEFAULT_TREE_CACHE_FILE))

CACHE_ROOT = SCRATCH_ROOT / "cache"
OUTPUT_ROOT = SCRATCH_ROOT / "output"
DTB_OUTPUT_ROOT = SCRATCH_ROOT / "output_dtb"

DEFAULT_PARAMETER_FILE = Path(__file__).with_name("coarse_grid.yaml")
PARAMETER_FILE = Path(os.environ.get("BEORN_PARAMETER_FILE", str(DEFAULT_PARAMETER_FILE)))


def _format_cache_value(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("/", "-")


def painted_cache_namespace(parameters) -> str:
    source = parameters.source
    distribution = _format_cache_value(getattr(source, "f_st_paint_distribution", "lognormal"))
    sigma = _format_cache_value(getattr(source, "f_st_paint_sigma", 0.5))
    seed = _format_cache_value(getattr(source, "f_st_paint_seed", None))
    return f"painted_output_fstar_dist_{distribution}_sigma_{sigma}_seed_{seed}"


def expected_snapshot_path(cache_handler, parameters, cache_namespace: str, z_index: int) -> Path:
    cache_dir = cache_handler.file_root / cache_namespace
    return beorn.structs.CoevalCube.get_file_path(
        cache_dir,
        parameters,
        z_index=z_index,
        cache_namespace=cache_namespace,
    )


def active_snapshot_indices(loader, requested_redshifts) -> list[int]:
    if requested_redshifts is None:
        return list(range(loader.redshifts.size))

    all_z = loader.redshifts
    indices = []
    for z_target in np.asarray(requested_redshifts, dtype=float):
        nearest = int(np.argmin(np.abs(all_z - z_target)))
        if nearest not in indices:
            indices.append(nearest)
    return sorted(indices)


def ensure_all_painted_snapshots_available(loader, cache_handler, parameters, cache_namespace: str) -> None:
    missing = []
    for z_index in active_snapshot_indices(loader, parameters.cosmo_sim.snapshot_redshifts):
        redshift = loader.redshifts[z_index]
        snapshot_path = expected_snapshot_path(cache_handler, parameters, cache_namespace, z_index)
        if not snapshot_path.exists():
            missing.append((z_index, float(redshift), snapshot_path))

    if missing:
        missing_text = "\n".join(
            f"z_index={z_index}, z={redshift:.6f}, path={path}"
            for z_index, redshift, path in missing
        )
        raise FileNotFoundError(
            "Painted output is incomplete. Missing painted snapshots:\n"
            f"{missing_text}"
        )


def load_temporal_output(output_root: Path, parameters):
    cube_path = beorn.structs.TemporalCube.get_file_path(output_root, parameters)
    if not cube_path.exists():
        return None
    cube = beorn.structs.TemporalCube.read(file_path=cube_path, parameters=parameters)
    if cube.z_snapshots is None or len(cube.z_snapshots) == 0:
        return None
    return cube


def main() -> None:
    parameters = beorn.structs.Parameters.from_yaml(PARAMETER_FILE)
    parameters.cosmo_sim.file_root = FILE_ROOT
    parameters.solver.redshifts = np.sort(parameters.solver.redshifts)[::-1]
    if parameters.cosmo_sim.snapshot_redshifts is not None:
        parameters.cosmo_sim.snapshot_redshifts = np.sort(parameters.cosmo_sim.snapshot_redshifts)[::-1]

    loader = beorn.load_input_data.ThesanLoader(
        parameters,
        cache_file=TREE_CACHE_FILE,
        is_high_res=False,
    )
    cache_handler = beorn.io.Handler(file_root=CACHE_ROOT)
    temporal_output = load_temporal_output(OUTPUT_ROOT, parameters)

    cache_namespace = painted_cache_namespace(parameters)
    logger.info("Using painted snapshot cache namespace %s", cache_namespace)
    snapshot_indices = active_snapshot_indices(loader, parameters.cosmo_sim.snapshot_redshifts)
    if temporal_output is None:
        ensure_all_painted_snapshots_available(loader, cache_handler, parameters, cache_namespace)
        logger.info("Using painted snapshot cache as the Grid_dTb input source.")
    else:
        logger.info("Using temporal output snapshots as the Grid_dTb input source.")
    logger.info("All %d painted snapshots are available. Starting Grid_dTb postprocessing.", len(snapshot_indices))

    DTB_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    simulation_name = beorn.structs.TemporalCube.simulation_name(parameters)
    ncell = int(parameters.simulation.Ncell)

    for snapshot_index in snapshot_indices:
        if temporal_output is not None:
            grid_snapshot = temporal_output.snapshot(float(loader.redshifts[snapshot_index]))
        else:
            grid_snapshot = cache_handler.load_file(
                parameters,
                beorn.structs.CoevalCube,
                z_index=snapshot_index,
                cache_namespace=cache_namespace,
            )
        grid_snapshot.to_arrays()
        redshift = float(grid_snapshot.z)

        logger.info(
            "Computing Grid_dTb for snapshot %03d at z=%.6f.",
            snapshot_index,
            redshift,
        )

        grid_dtb = np.asarray(grid_snapshot.Grid_dTb)
        snapshot_file = (
            f"{simulation_name}_snapshot_{snapshot_index:03d}_"
            f"z{redshift:.6f}_Grid_dTb_N{ncell}.h5"
        )
        snapshot_path = DTB_OUTPUT_ROOT / snapshot_file

        with h5py.File(snapshot_path, "w") as hdf5_file:
            hdf5_file.create_dataset("Grid_dTb", data=grid_dtb)
            hdf5_file.attrs["field"] = "Grid_dTb"
            hdf5_file.attrs["snapshot_index"] = int(snapshot_index)
            hdf5_file.attrs["redshift"] = redshift
            hdf5_file.attrs["grid_size"] = ncell
            hdf5_file.attrs["simulation_name"] = simulation_name

    logger.info("Finished writing Grid_dTb snapshots to %s", DTB_OUTPUT_ROOT)


if __name__ == "__main__":
    main()
