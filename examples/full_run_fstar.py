from pathlib import Path
import gc
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import numpy as np
import beorn

from types import SimpleNamespace

try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
except Exception:
    _comm = None
    _rank = 0
# change the simulation-related paths here
SCRATCH_ROOT = Path("/xdisk/timeifler/yhhuang/BEoRN-v2/")
FILE_ROOT = Path("/xdisk/timeifler/yhhuang/Thesan/")

CACHE_ROOT = SCRATCH_ROOT / "thesan_run_fstar" / "cache"
OUTPUT_ROOT = SCRATCH_ROOT / "thesan_run_fstar" / "output"

# PARAMETER_FILE = Path(".") / "parameters_fstar.yaml"
PARAMETER_FILE = Path(".") / "coarse_grid.yaml"  # for a quick test run with a coarse grid - this is not meant to produce accurate results, just to test the workflow and the f_st-grid solver.


### Parameter setup
parameters = beorn.structs.Parameters.from_yaml(PARAMETER_FILE)
parameters.simulation.file_root = FILE_ROOT
parameters.solver.redshifts = np.sort(parameters.solver.redshifts)

### IO setup
loader = beorn.load_input_data.ThesanLoader(
    parameters,
    is_high_res = False  # in our example we have used the Thesan-Dark-2 simulation which is the low-res counterpart to Thesan-Dark-1
)
cache_handler = beorn.io.Handler(file_root=CACHE_ROOT)
output_handler = beorn.io.Handler(file_root=OUTPUT_ROOT)
# redirect logs to a file under the OUTPUT_ROOT directory - this can be useful for cases where you want to retrace many parallel runs later on.
output_handler.save_logs(parameters)

### In a first step, we compute the radiation profiles around sources at all redshifts of interest
# Use the f_st-grid solver so profiles are precomputed on (mass, alpha, f_st, z)
from beorn.precomputation.solver import RadiationProfileFstSolver
from beorn.structs.radiation_profiles import RadiationProfilesFStarGrid
solver = RadiationProfileFstSolver(parameters, loader.redshifts)
# the computation does not depend on the spatial information, so the profiles are reusable
# instead of recomputing them every time, we can reuse a cached version if available

profile_cache_namespace = solver.profile_cache_namespace()
profile_cache_dir = cache_handler.file_root / profile_cache_namespace
expected_profiles_path = RadiationProfilesFStarGrid.get_file_path(
    profile_cache_dir,
    parameters,
    cache_namespace=profile_cache_namespace,
)

if _rank == 0:
    cache_exists = expected_profiles_path.exists()
    logger.info("Profile cache path is %s", expected_profiles_path)
    logger.info("Profile cache exists: %s", cache_exists)
else:
    cache_exists = None

if _comm is not None:
    cache_exists = _comm.bcast(cache_exists, root=0)

if cache_exists:
    profiles_path = str(expected_profiles_path)
    logger.info("Using cached f_st-grid radiation profiles from %s", profiles_path)
else:
    logger.info("Profile cache miss. Entering collective profile generation.")
    profiles_full = solver.get_or_compute_profiles(cache_handler)
    profiles_path = str(profiles_full._file_path)
    del profiles_full
    gc.collect()
    logger.info("Profile generation finished. Using %s", profiles_path)

# Pass only the file path into painting on all ranks to avoid OOM.
profiles = SimpleNamespace(_file_path=Path(profiles_path))

### In a second step, we use the precomputed profiles to paint the desired quantities onto the simulation grids
# For RadiationProfilesFStarGrid, the PaintingCoordinator automatically switches to the
# stochastic f_st painting path, samples one f_st per halo, and maps to the nearest profile slice.
painter = beorn.painting.PaintingCoordinator(
    parameters,
    loader = loader,
    cache_handler = cache_handler,
    output_handler = output_handler
)

final_output = painter.paint_full(profiles)

# The final_output object contains all individual grids (corresponding to each computed quantity) for each redshift
# This object has been written to an .hdf5 format and can be inspected manually
# You can also load it from within your code by running:
output = output_handler.load_file(parameters, beorn.structs.TemporalCube)
# the data associated to this run is uniquely identified by the hash of the parameters - no need to manually specify the path
