import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import beorn

FILE_ROOT = Path("/xdisk/timeifler/yhhuang/BEoRN-v2/")
CACHE_ROOT = FILE_ROOT / "cache"
OUTPUT_ROOT = FILE_ROOT / "output"
PARAMETER_FILE = Path(".") / "parameters.yaml"

parameters = beorn.structs.Parameters.from_yaml(PARAMETER_FILE)

cache_handler = beorn.io.Handler(CACHE_ROOT)
loader = beorn.load_input_data.ArtificialHaloLoader(parameters, halo_count=100)

solver = beorn.precomputation.RadiationProfileSolver(parameters, loader.redshifts)
profiles = solver.get_or_compute_profiles(cache_handler)

output_handler = beorn.io.Handler(OUTPUT_ROOT)
p = beorn.painting.PaintingCoordinator(
    parameters,
    loader=loader,
    cache_handler=cache_handler,
    output_handler=output_handler
)
multi_z_quantities = p.paint_full(profiles)

z = 9
z_index = np.argmin(np.abs(loader.redshifts - z))

print('plotting the xHII grid at z=9')
xHII_grid = multi_z_quantities.Grid_xHII[z_index, ...]
plt.figure()
plt.imshow(xHII_grid[:, 64, :], origin='lower', cmap='viridis')
plt.savefig(OUTPUT_ROOT / 'test_xHII_slice.png')
plt.clf()
