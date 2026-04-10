**Scope**
This integration branch was cleaned up to stay as close as practical to `origin/main`.
Temporary debugging and broad backward-compatibility shims were removed where they were not required for the merged functionality.

**Temporary Changes Removed**
- Removed the offset-free THESAN catalog fallback for snapshots without `offsets_*`.
- Removed the legacy `cosmology.h -> cosmology.h0` YAML compatibility shim.
- Removed the solver-side automatic redshift reordering shim.

**Intentional Differences Still Present**
- [src/beorn/painting/coordinator.py](/home/u15/yhhuang/cosmology/BEoRN-v2/src/beorn/painting/coordinator.py)
  Keeps the stochastic `f_st` painting path and its associated runtime fixes:
  HDF5-backed small-array handling (`z_history`, `f_st_grid`), empty-halo early return, and explicit propagation of `ProcessPoolExecutor` worker failures.

- [src/beorn/precomputation/solver.py](/home/u15/yhhuang/cosmology/BEoRN-v2/src/beorn/precomputation/solver.py)
  Keeps the `np.asarray(sol.y)` fix before `.clip()` in `R_bubble()`, which avoids list/array type failures during profile generation.

- [src/beorn/load_input_data/cosmo_sim_thesan.py](/home/u15/yhhuang/cosmology/BEoRN-v2/src/beorn/load_input_data/cosmo_sim_thesan.py)
  Keeps the merge-essential THESAN loader fixes:
  robust snapshot-index parsing for filenames like `offsets_039.hdf5`,
  storage on private attributes instead of assigning to the base class `catalogs` property,
  and use of `cosmo_sim.halo_catalogs_thesan_mass_assignment` / `particle_mapping_backend`.

- [src/beorn/structs/temporal_cube.py](/home/u15/yhhuang/cosmology/BEoRN-v2/src/beorn/structs/temporal_cube.py)
  Keeps the `append()` fix so per-redshift output snapshots are written to the temporal output directory even when the same `CoevalCube` was already written to cache.

- [src/beorn/plotting/radiation_profiles.py](/home/u15/yhhuang/cosmology/BEoRN-v2/src/beorn/plotting/radiation_profiles.py)
  Keeps the `f_st`-grid-aware plotting support merged from this branch.

- [tests/test_coordinator.py](/home/u15/yhhuang/cosmology/BEoRN-v2/tests/test_coordinator.py)
- [tests/test_solver.py](/home/u15/yhhuang/cosmology/BEoRN-v2/tests/test_solver.py)
  Keep the coverage for stochastic `f_st` profile generation and painting behavior.

**Example / Validation Files**
- The example files under [examples/coarse_grid.yaml](/home/u15/yhhuang/cosmology/BEoRN-v2/examples/coarse_grid.yaml), [examples/full_run_fstar.py](/home/u15/yhhuang/cosmology/BEoRN-v2/examples/full_run_fstar.py), [examples/postprocess_dtb_fstar.py](/home/u15/yhhuang/cosmology/BEoRN-v2/examples/postprocess_dtb_fstar.py), [examples/run_beorn_hpc.slurm](/home/u15/yhhuang/cosmology/BEoRN-v2/examples/run_beorn_hpc.slurm), and [examples/validate.ipynb](/home/u15/yhhuang/cosmology/BEoRN-v2/examples/validate.ipynb) remain branch-specific validation tooling for the coarse-grid merge test.
