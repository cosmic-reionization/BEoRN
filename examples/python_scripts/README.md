# examples/python_scripts/

Plain-script counterparts to two of the tutorial notebooks, for running BEoRN
as a batch job (e.g. via `sbatch`/`srun`, or just `python script.py` in a
terminal) instead of interactively in Jupyter. Each script writes its results
to disk; `plot_results.ipynb` then reads those results back and reproduces
the tutorials' plots without recomputing anything.

| File | Notebook counterpart | Parameters |
|---|---|---|
| `artificial_halo_gpu.py` | `../artificial_halos.ipynb` / `../artificial_halo_gpu.ipynb` | hardcoded in `build_parameters()` |
| `lpt_native_21cm.py` | `../lpt_native_21cm.ipynb` | loaded from `param.yaml` via `beorn.structs.Parameters.from_yaml` |

`lpt_native_21cm.py` also serves as the docs' "full run from a non-interactive
script" tutorial (see `lpt_native_21cm_file.rst`, included from
`docs/tutorials.rst`) since it's fully self-contained - no external N-body
input or HPC-specific paths required, unlike a run driven by external
simulation data.

## Running

```bash
python artificial_halo_gpu.py
python lpt_native_21cm.py                       # uses param.yaml next to the script
python lpt_native_21cm.py --param-file my.yaml   # or point at your own parameter file
```

`artificial_halo_gpu.py` auto-detects a GPU backend (jax or torch): if one is
available it prints a short `paint_single` timing/parity benchmark against
numpy, then runs the full production pipeline with that backend
(`parameters.simulation.backend.default`). With no GPU it just runs on numpy.

`lpt_native_21cm.py` generates its own cosmological density field and halo
catalogues internally (Lagrangian Perturbation Theory + conditional halo mass
function) - no external simulation data needed. See `param.yaml` for the full
parameter schema and comments on which fields matter.

Each script writes under its own `cache/` (radiation-profile cache, reused
across runs with the same astrophysical parameters) and `output/` (painted
grids + a `stats_*.h5` summary-statistics file) subfolders, created
automatically on first run.

## Reading the results back

`plot_results.ipynb` does not rerun any physics - it imports each script as a
plain Python module (for its `build_parameters()` / path / input-tag helpers)
and loads the grids straight from the `output/` directory:

```python
import artificial_halo_gpu as ahg
import beorn

parameters = ahg.build_parameters()
output_handler = beorn.io.Handler(ahg.OUTPUT_ROOT, input_tag=ahg.INPUT_TAG)
multi_z_quantities = output_handler.load_file(parameters, beorn.structs.TemporalCube)
```

Run the corresponding script at least once before opening the notebook, or
the `load_file` calls will raise a file-not-found error.

## Key things to get right

- **Run the script before the notebook.** The notebook only reads
  `output/`; it has nothing to plot until a script has produced it.
- **Reuse the same `parameters` used to write the output** when reading it
  back. `Handler.load_file` locates files by a hash of the run's parameters
  (plus `input_tag` where one is set, for the `cosmo_sim`-only inputs like a
  random seed that aren't part of `Parameters` itself) - reconstructing
  `parameters` differently than the writing script did means the reader
  won't find the file. Importing the script module and calling its
  `build_parameters()` (and `input_tag_for()` for the LPT script)
  guarantees this instead of duplicating the parameter values by hand.
- **`param.yaml` only needs to list overrides.** `Parameters.from_yaml` starts
  from `beorn.structs.Parameters()`'s defaults and replaces only the
  sections/fields you specify - see `lpt_native_21cm.py`'s module docstring
  and `param.yaml`'s header comment.
