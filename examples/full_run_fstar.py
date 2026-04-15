from pathlib import Path
import gc
import logging
import os
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
DEFAULT_SCRATCH_ROOT = "/xdisk/timeifler/yhhuang/BEoRN-v2/coarse_grid_test_main/"
DEFAULT_FILE_ROOT = "/xdisk/timeifler/yhhuang/Thesan/"
DEFAULT_TREE_CACHE_FILE = "/xdisk/timeifler/yhhuang/Thesan/postprocessing/trees/LHaloTree/tree_cache_v2.hdf5"

SCRATCH_ROOT = Path(os.environ.get("BEORN_SCRATCH_ROOT", DEFAULT_SCRATCH_ROOT))
FILE_ROOT = Path(os.environ.get("BEORN_FILE_ROOT", DEFAULT_FILE_ROOT))
TREE_CACHE_FILE = Path(os.environ.get("BEORN_TREE_CACHE_FILE", DEFAULT_TREE_CACHE_FILE))

CACHE_ROOT = SCRATCH_ROOT / "cache"
OUTPUT_ROOT = SCRATCH_ROOT / "output"

DEFAULT_PARAMETER_FILE = Path(__file__).with_name("coarse_grid.yaml")
PARAMETER_FILE = Path(os.environ.get("BEORN_PARAMETER_FILE", str(DEFAULT_PARAMETER_FILE)))


### Parameter setup
parameters = beorn.structs.Parameters.from_yaml(PARAMETER_FILE)
parameters.cosmo_sim.file_root = FILE_ROOT
parameters.solver.redshifts = np.sort(parameters.solver.redshifts)[::-1]
if parameters.cosmo_sim.snapshot_redshifts is not None:
    parameters.cosmo_sim.snapshot_redshifts = np.sort(parameters.cosmo_sim.snapshot_redshifts)[::-1]

SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)

### IO setup
loader = beorn.load_input_data.ThesanLoader(
    parameters,
    cache_file=TREE_CACHE_FILE,
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

profiles_full = None
if cache_exists:
    profiles_path = str(expected_profiles_path)
    logger.info("Using cached f_st-grid radiation profiles from %s", profiles_path)
else:
    mpi_size = _comm.Get_size() if _comm is not None else 1
    if mpi_size > 1:
        if _rank == 0:
            logger.info("Profile cache miss. Rank 0 will generate the f_st-grid profiles.")
            profiles_full = solver.get_or_compute_profiles(cache_handler)
            profiles_path = str(profiles_full._file_path)
            logger.info("Profile generation finished. Using %s", profiles_path)
        else:
            logger.info("Profile cache miss. Rank %d is waiting for rank 0 to finish profile generation.", _rank)
            profiles_path = None
        _comm.barrier()
        profiles_path = _comm.bcast(profiles_path, root=0)
        cache_exists = True
    else:
        logger.info("Profile cache miss. Entering profile generation.")
        profiles_full = solver.get_or_compute_profiles(cache_handler)
        profiles_path = str(profiles_full._file_path)
        logger.info("Profile generation finished. Using %s", profiles_path)

mpi_size = _comm.Get_size() if _comm is not None else 1
if mpi_size > 1:
    # MPI workers load profiles from disk to avoid replicating the ~GB-sized
    # profile cube across ranks in memory.
    profiles = SimpleNamespace(_file_path=Path(profiles_path))
    if profiles_full is not None:
        del profiles_full
        gc.collect()
else:
    if profiles_full is None:
        profiles = cache_handler.load_file(parameters, RadiationProfilesFStarGrid, cache_namespace=profile_cache_namespace)
    else:
        profiles = profiles_full

### In a second step, we use the precomputed profiles to paint the desired quantities onto the simulation grids
# For RadiationProfilesFStarGrid, the PaintingCoordinator automatically switches to the
# stochastic f_st painting path, samples one f_st per halo, and maps to the nearest profile slice.
painter = beorn.painting.PaintingCoordinator(
    parameters,
    loader = loader,
    cache_handler = cache_handler,
    output_handler = output_handler
)

redshift_subset = None
if parameters.cosmo_sim.snapshot_redshifts is not None:
    redshift_subset = parameters.cosmo_sim.snapshot_redshifts.tolist()
    logger.info(
        "Painting a reduced snapshot subset: %d requested redshifts from z=%.2f to z=%.2f",
        len(redshift_subset),
        redshift_subset[0],
        redshift_subset[-1],
    )

final_output = painter.paint_full(profiles, redshift_subset=redshift_subset)

# The final_output object contains all individual grids (corresponding to each computed quantity) for each redshift
# This object has been written to an .hdf5 format and can be inspected manually
# You can also load it from within your code by running:
output = output_handler.load_file(parameters, beorn.structs.TemporalCube)
# the data associated to this run is uniquely identified by the hash of the parameters - no need to manually specify the path
