=========
Changelog
=========

Unreleased
----------

Corrections from the 2026-09-14 internal review of the painting and profile code. **This is not a
complete record of changes since v2.1.0** -- it lists only that review's user-visible results.

Behaviour
~~~~~~~~~

* The stacked heating and Lyman-alpha painting kernels now include each source's contribution to
  its own cell. That cell used to be a point sample of a steeply rising (~1/r^2) profile at r = 0,
  which the interpolators returned as zero, so a source deposited nothing in the cell it occupies.
  ``Grid_Temp``, ``Grid_xal`` and the derived ``Grid_dTb`` change as a result -- most at small
  scales -- while ``Grid_xHII`` is unaffected. Maps painted with earlier versions are not
  comparable in T_k or x_al.
* The X-ray and Lyman-alpha emission histories no longer receive a duplicate zero anchor at
  ``solver.z_source_start`` when the profile redshift grid already starts there. The duplicate made
  the interpolator ramp the star-formation rate to zero across the top grid interval instead of
  using the computed values. The painted effect is small (<= 3e-5 per cell where measured), but it
  changes profile-cube contents.
* ``rho_alpha_profile`` honours ``solver.z_source_start`` instead of a hardcoded 35.

Caches -- action required when upgrading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``profiles_hash`` and ``profiles_fstar_hash`` now cover ``solver.ode_rtol``, ``ode_atol`` and
  ``ode_method``, and the f_st cube namespace records the grid spacing (``_log`` / ``_lin``).
  Previously a tolerance change silently reused a cube solved at other tolerances. **Existing
  profile cubes therefore resolve to new names and will not be found**: copy them to the new names
  or let them be recomputed. The renaming alone does not change cube contents.
* Note the emission-anchor fix above *does* change cube contents without changing the hash, so a
  cube solved by an earlier version is still found under its old name. Use
  ``--force-recompute-profiles`` or a fresh cache root if you need corrected profiles.

Fixes
~~~~~

* MPI painting: the resume check now tests the filename ``paint_single`` actually writes -- the
  nearest profile redshift -- as the serial loop already did. An MPI run whose loader redshifts
  differ from the profile grid no longer repaints snapshots it has or skips ones it lacks.
* The per-bin forward FFTs no longer run with ``workers=-1`` when several ranks share a node, which
  oversubscribed the node's CPUs; the worker count is bounded by the CPUs allocated to each rank.
* ``ThesanLoader`` sizes the subhalo-to-group map from ``Header/Nsubgroups_Total`` and verifies the
  count, instead of guessing 1.5x the last file offset.
* ``PaintingCoordinator``, ``RadiationProfileSolver`` and ``RadiationProfileFstSolver`` have class
  docstrings again; they sat after the first method, so ``__doc__`` was ``None``.
* Removed an unreachable branch in the ionisation-kernel path, and an assertion message that
  contradicted its own check.

Documentation
~~~~~~~~~~~~~

* Reading a struct from HDF5 is eager: every dataset is materialised and the file closed, so a
  profile cube costs its full size in RAM (~10.9 GB for the largest THESAN-1 cube). Painting with
  several MPI ranks avoids this -- each rank reads one redshift slice -- and a single-rank run now
  warns when the cube is large relative to the memory available to the job.
* ``beorn_hash`` includes ``simulation.cores`` and ``simulation.fft_backend``, so changing either
  mid-run renames the output directory and silently breaks resume. Documented with the rule not to
  change them between a run's launch and its resume.
* Corrected parameter docstrings: halo masses are in Msun/h; ``fXh = 'constant'`` gives
  (2e-4)**0.225 ~ 0.147, not 0.11; ``halo_mass_nbin`` counts bin *edges*; ``ThesanLoader`` reads
  full-hydro THESAN, not THESAN-DARK.

v2.1.0
------
* Per-halo accretion rate α fitting from merger trees (Moll 2025, Master's thesis, ETH Zürich)
* New ``MergerTreeLoader`` abstract base class for simulations with merger trees
* THESAN-DARK merger tree tutorial (``thesan_merger_tree_postprocessing`` notebook)
* THESAN-DARK N-body data exploration tutorial (``thesan_nbody_data_exploration`` notebook)
* Refactored ``beorn.particle_mapping`` module with unified ``map_particles_to_mesh`` dispatcher; pure-NumPy default backend; optional ``numba``, ``pylians``, ``torch``, ``jax`` backends
* ``force_recompute`` parameter added to ``RadiationProfileSolver.get_or_compute_profiles``
* ``CosmologyParameters.h`` renamed to ``h0`` for clarity
* ``pylians`` is no longer a required dependency; it is an optional extra

v1.2.0
------
* Post-processing of any N-body simulation data
* Tested tutorial for pkdgrav3 N-body data (``nbody_simulation_halos`` notebook)
* Halo catalog loading abstracted via a general N-body base class

v1.1.0
------
* Stable interface with 21cmFAST v3 (``>=3.4.0, <4``)
* Tested tutorial for 21cmFAST halo catalogues (``21cmfast_halos`` notebook)
* Artificial halo quick-start tutorial (``artificial_halos`` notebook)
* Radiation profile pre-computation and caching
* 3D signal map painting (xHII, Tk, Lyman-alpha, dTb)
* Lightcone construction and plotting utilities

v0.1.0
------
* Initial form of the package
* Compatible with python 3 only
