from pathlib import Path
import logging

import h5py
import numpy as np

import beorn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SCRATCH_ROOT = Path("/xdisk/timeifler/yhhuang/BEoRN-v2/")
FILE_ROOT = Path("/xdisk/timeifler/yhhuang/Thesan/")

OUTPUT_ROOT = SCRATCH_ROOT / "thesan_run_fstar" / "output"
DTB_OUTPUT_ROOT = SCRATCH_ROOT / "thesan_run_fstar" / "output_dtb"

PARAMETER_FILE = Path(".") / "parameters_fstar.yaml"


def main() -> None:
    parameters = beorn.structs.Parameters.from_yaml(PARAMETER_FILE)
    parameters.simulation.file_root = FILE_ROOT
    parameters.solver.redshifts = np.sort(parameters.solver.redshifts)

    output_handler = beorn.io.Handler(file_root=OUTPUT_ROOT)
    cube = output_handler.load_file(parameters, beorn.structs.TemporalCube)

    DTB_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    simulation_name = beorn.structs.TemporalCube.simulation_name(parameters)
    ncell = int(parameters.simulation.Ncell)

    redshifts = np.asarray(cube.z[:], dtype=np.float64)
    logger.info("Loaded base-field manifest for %d snapshots.", redshifts.size)

    for snapshot_index, redshift in enumerate(redshifts):
        logger.info(
            "Computing Grid_dTb for snapshot %03d at z=%.6f.",
            snapshot_index,
            redshift,
        )

        grid_snapshot = beorn.structs.CoevalCube(
            parameters=parameters,
            z=float(redshift),
            delta_b=np.asarray(cube.delta_b[snapshot_index, ...]),
            Grid_Temp=np.asarray(cube.Grid_Temp[snapshot_index, ...]),
            Grid_xHII=np.asarray(cube.Grid_xHII[snapshot_index, ...]),
            Grid_xal=np.asarray(cube.Grid_xal[snapshot_index, ...]),
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
            hdf5_file.attrs["redshift"] = float(redshift)
            hdf5_file.attrs["grid_size"] = ncell
            hdf5_file.attrs["simulation_name"] = simulation_name

    logger.info("Finished writing Grid_dTb snapshots to %s", DTB_OUTPUT_ROOT)


if __name__ == "__main__":
    main()
