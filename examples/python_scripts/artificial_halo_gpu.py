"""Run BEoRN's artificial-halo pipeline as a plain script, GPU-accelerated.

Script counterpart to ``../artificial_halos.ipynb`` (same source model and
halo catalog) and the practical, "just run it" side of
``../artificial_halo_gpu.ipynb`` (which additionally validates gradients and
the differentiable-painting path interactively). Here we only:

1. build the artificial halo catalog and precompute radiation profiles,
2. print a short GPU-vs-numpy timing benchmark for `paint_single` if a GPU
   backend (jax or torch) is available, and
3. run the full production painting pipeline and save the results to disk.

Inspect the results afterwards with ``plot_results.ipynb`` in this folder,
which reads ``build_parameters()`` / ``OUTPUT_ROOT`` / ``INPUT_TAG`` from this
module rather than re-running the pipeline.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

# JAX grabs 75% of the GPU by default - disable preallocation *before*
# importing jax (matters on shared/contended GPUs).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import beorn

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_ROOT = SCRIPT_DIR / "cache"
OUTPUT_ROOT = SCRIPT_DIR / "output"
INPUT_TAG = "artificial_halo_gpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_gpu_backend() -> str:
    """Return the fastest available backend: 'jax', 'torch', or 'numpy'."""
    try:
        import jax
        if any(d.platform != "cpu" for d in jax.devices()):
            return "jax"
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            return "torch"
    except ImportError:
        pass
    return "numpy"


def build_parameters() -> beorn.structs.Parameters:
    """Same source model and simulation volume as ``../artificial_halos.ipynb``.

    Also fixes ``simulation.backend.default``/``cores`` to the GPU backend
    this machine will actually use for the production run: `beorn_hash()` (which
    determines where `paint_full` writes/looks for output) is computed from
    the whole `simulation` section, so the reader (`plot_results.ipynb`,
    calling this same function) must reconstruct an identical `simulation`
    section to find the writer's output directory.
    """
    parameters = beorn.structs.Parameters()

    parameters.simulation.cores = 4
    parameters.simulation.Lbox = 100
    parameters.simulation.use_hunits = True  # Lbox above is in Mpc/h, not physical Mpc
    parameters.simulation.Ncell = 128

    parameters.solver.halo_mass_bin_min = 1e7
    parameters.solver.halo_mass_bin_max = 1e15
    parameters.solver.halo_mass_nbin = 200
    parameters.solver.halo_mass_accretion_alpha = np.array([0.785, 0.795])

    # Source age: how far back to integrate the emission history.
    parameters.solver.z_source_start = 35.0  # default
    parameters.source.t_source_age = None  # default (no age cap)

    # Emission
    parameters.source.Nion = 5000 * 3
    parameters.source.energy_cutoff_min_xray = 500
    parameters.source.energy_cutoff_max_xray = 2000
    parameters.source.energy_min_sed_xray = 500
    parameters.source.energy_max_sed_xray = 2000
    parameters.source.alS_xray = 1.5
    parameters.source.xray_normalisation = 3.4e40 * 3

    # Escape fraction
    parameters.source.f0_esc = 0.2
    parameters.source.pl_esc = 0

    # Star formation efficiency
    parameters.source.f_st = 1
    parameters.source.g1 = 0
    parameters.source.g2 = 0
    parameters.source.g3 = 4
    parameters.source.g4 = -1
    parameters.source.Mp = 1.6e11 * parameters.cosmology.h0
    parameters.source.Mt = 1e7

    # Minimum star-forming halo mass
    parameters.source.halo_mass_min = 1e5

    backend = detect_gpu_backend()
    parameters.simulation.backend.default = backend
    # GPU backends paint mass bins serially; forked worker processes that
    # inherit a CUDA context crash (BrokenProcessPool).
    parameters.simulation.cores = 1 if backend != "numpy" else 4

    return parameters


def benchmark_backends(parameters, loader, profiles, output_handler, z_index: int) -> None:
    """Print a paint_single timing/parity table across available fft backends."""
    backends = ["numpy"]
    try:
        import jax
        if any(d.platform != "cpu" for d in jax.devices()):
            backends.append("jax")
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            backends.append("torch")
    except ImportError:
        pass

    if len(backends) == 1:
        logger.info("No GPU backend (jax/torch) available - skipping backend benchmark.")
        return

    original_cores = parameters.simulation.cores
    # GPU backends paint mass bins serially; forked worker processes that
    # inherit a CUDA context crash (BrokenProcessPool).
    parameters.simulation.cores = 1

    fields, timings = {}, {}
    for be in backends:
        parameters.simulation.backend.default = be
        painter = beorn.painting.PaintingCoordinator(
            parameters, loader=loader, output_handler=output_handler,
            force_recompute=True,
        )
        t0 = time.time()
        cube = painter.paint_single(z_index, profiles)
        timings[be] = time.time() - t0
        fields[be] = (np.array(cube.Grid_xHII), np.array(cube.Grid_Temp), np.array(cube.Grid_xal))

    parameters.simulation.backend.default = "numpy"
    parameters.simulation.cores = original_cores

    logger.info("paint_single backend benchmark at z=%.2f:", loader.redshifts[z_index])
    for be in backends:
        if be == "numpy":
            logger.info("  numpy : %6.2f s (reference)", timings[be])
        else:
            d = max(np.max(np.abs(a - b)) / (np.abs(b).max() + 1e-30)
                    for a, b in zip(fields[be], fields["numpy"]))
            logger.info("  %-6s: %6.2f s   speedup %.1fx   max rel field diff vs numpy %.1e",
                        be, timings[be], timings["numpy"] / timings[be], d)


def main() -> None:
    parameters = build_parameters()
    backend = parameters.simulation.backend.default
    cores = parameters.simulation.cores
    logger.info("Selected backend=%s for the production run", backend)

    cache_handler = beorn.io.Handler(CACHE_ROOT)
    loader = beorn.load_input_data.ArtificialHaloLoader(parameters, halo_count=100)

    solver = beorn.precomputation.RadiationProfileSolver(parameters, loader.redshifts)
    profiles = solver.get_or_compute_profiles(cache_handler)

    output_handler = beorn.io.Handler(OUTPUT_ROOT, input_tag=INPUT_TAG)

    z_index = int(np.argmin(np.abs(loader.redshifts - 9.0)))
    benchmark_backends(parameters, loader, profiles, output_handler, z_index)

    # benchmark_backends leaves backend/cores at its own defaults - restore
    # the production values fixed by build_parameters() before painting.
    parameters.simulation.backend.default = backend
    parameters.simulation.cores = cores

    painter = beorn.painting.PaintingCoordinator(
        parameters, loader=loader, output_handler=output_handler,
        force_recompute=True,
    )
    t0 = time.time()
    multi_z_quantities = painter.paint_full(profiles)
    logger.info("paint_full finished in %.1f s (backend=%s)", time.time() - t0, backend)

    stats = beorn.structs.StatisticsEstimator(multi_z_quantities, parameters)
    stats_path = stats.save(path=OUTPUT_ROOT / f"stats_{INPUT_TAG}.h5")
    logger.info("Statistics saved to %s", stats_path)
    logger.info("Grids written under %s (input_tag=%s) - see plot_results.ipynb",
                OUTPUT_ROOT, INPUT_TAG)


if __name__ == "__main__":
    main()
