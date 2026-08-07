"""Run BEoRN's native LPT+CHMF pipeline as a plain script, driven by a YAML file.

Script counterpart to ``../lpt_native_21cm.ipynb``: generates the cosmological
density field and halo catalogues internally (Lagrangian Perturbation Theory
+ conditional halo mass function - no external simulation input required),
paints the 3D signal maps, and saves the results to disk. All simulation,
astrophysical, and cosmological parameters come from a YAML file
(``param.yaml`` next to this script by default) via
``beorn.structs.Parameters.from_yaml`` - see that file for the full parameter
schema.

Inspect the results afterwards with ``plot_results.ipynb`` in this folder,
which reads ``build_parameters()`` / ``OUTPUT_ROOT`` / ``input_tag_for()``
from this module rather than re-running the pipeline.

Usage:
    python lpt_native_21cm.py [--param-file PATH]
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import beorn

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_ROOT = SCRIPT_DIR / "cache"
OUTPUT_ROOT = SCRIPT_DIR / "output"
DEFAULT_PARAM_FILE = SCRIPT_DIR / "param.yaml"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_parameters(param_file: Path = DEFAULT_PARAM_FILE) -> beorn.structs.Parameters:
    return beorn.structs.Parameters.from_yaml(Path(param_file))


def input_tag_for(parameters: beorn.structs.Parameters, seed: int) -> str:
    return (f"lpt_native_N{parameters.simulation.Ncell}"
            f"_L{parameters.simulation.Lbox:.0f}_seed{seed}")


def main(param_file: Path = DEFAULT_PARAM_FILE) -> None:
    parameters = build_parameters(param_file)
    seed = parameters.cosmo_sim.random_seed
    input_tag = input_tag_for(parameters, seed)

    cache_handler = beorn.io.Handler(CACHE_ROOT)
    loader = beorn.load_input_data.LPTHaloLoader(
        parameters, seed=seed, halo_seed=parameters.halo_sim.random_seed,
        n_mass_bins=parameters.solver.halo_mass_nbin,
    )

    solver = beorn.precomputation.RadiationProfileSolver(
        parameters, parameters.cosmo_sim.snapshot_redshifts
    )
    profiles = solver.get_or_compute_profiles(cache_handler)

    output_handler = beorn.io.Handler(OUTPUT_ROOT, input_tag=input_tag)

    painter = beorn.painting.PaintingCoordinator(
        parameters, loader=loader, output_handler=output_handler,
    )
    t0 = time.time()
    multi_z_quantities = painter.paint_full(profiles)
    logger.info("paint_full finished in %.1f s", time.time() - t0)

    stats = beorn.structs.StatisticsEstimator(multi_z_quantities, parameters)
    stats_path = stats.save(path=OUTPUT_ROOT / f"stats_{input_tag}.h5")
    logger.info("Statistics saved to %s", stats_path)
    logger.info("Grids written under %s (input_tag=%s) - see plot_results.ipynb",
                OUTPUT_ROOT, input_tag)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--param-file", type=Path, default=DEFAULT_PARAM_FILE,
        help="Path to a BEoRN parameter YAML file (default: param.yaml next to this script).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.param_file)
