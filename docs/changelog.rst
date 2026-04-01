=========
Changelog
=========

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

v0.1
----
* Initial form of the package
* Compatible with python 3 only
