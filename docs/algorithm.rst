Algorithm and Parallelisation
==============================

The BEoRN pipeline has two phases.  **Phase 1** (precomputation) runs once and
is cached to HDF5; **Phase 2** (painting) reads the cache and produces 3D IGM
maps, exploiting two independent parallelism axes.

.. mermaid::

   flowchart TD
       P([Parameters])

       subgraph pre["Phase 1 — Precomputation  ·  result cached to HDF5"]
           direction TB
           Solver["RadiationProfileSolver.solve()"]
           MA["mass_accretion()\nmass × alpha × z grid"]
           RB["R_bubble()\nODE · solve_ivp"]
           RX["rho_xray()\nfrequency integral ∫dν"]
           RH["rho_heat()\nODE · solve_ivp"]
           RA["rho_alpha()\nLy-α profile ∫dr"]
           RP[("RadiationProfiles\n.h5")]

           Solver --> MA
           MA --> RB
           MA --> RX
           MA --> RA
           RX --> RH
           RB --> RP
           RH --> RP
           RA --> RP
       end

       subgraph paint["Phase 2 — Painting  (PaintingCoordinator.paint_full)"]
           direction TB
           Coord["paint_full()"]

           subgraph mpi["Parallelism L1 — MPI  (one snapshot per rank)"]
               SZ["snapshot at redshift z"]
           end

           subgraph snap["paint_single()  —  per snapshot"]
               Load["load_halo_catalog()\nload_density_field()"]
               Gate{"fft_backend"}

               subgraph cpu_path["CPU path  (numpy)"]
                   Pool["ProcessPoolExecutor\ncores workers"]
               end

               subgraph gpu_path["GPU path  (jax / torch)"]
                   GPUloop["serial mass-bin loop\nGPU parallelism inside FFT"]
               end

               subgraph bin["paint_single_mass_bin()  —  per mass × alpha bin"]
                   direction LR
                   TM["to_mesh()\nparticle → grid"] --> FFT["precompute_fft()\nforward rFFT"]
                   FFT --> FMK["fourier_multiply_kernel()\nfa × FFT(kernel) × renorm"]
                   FMK --> Acc["Fourier accumulator\n+= contribution"]
               end

               Load --> Gate
               Gate -- "numpy  CPU" --> Pool
               Gate -- "jax / torch  GPU" --> GPUloop
               Pool --> bin
               GPUloop --> bin
               Acc --> IFFT["3 × ifft_field()\nxHII · Temp · xal"]
               IFFT --> Post["spreading_excess_fast()\n+ post-processing"]
               Post --> CC[("CoevalCube .h5")]
           end

           Coord --> mpi
           SZ --> Load
       end

       P --> Solver
       P --> Coord
       RP --> Coord
       CC --> TC[("TemporalCube .h5")]


Parallelism levels
------------------

**Level 1 — MPI across snapshots**
   When launched with ``mpirun``, :class:`~beorn.painting.coordinator.PaintingCoordinator`
   distributes redshift snapshots across MPI ranks via
   :class:`mpi4py.futures.MPICommExecutor`.  Each rank independently paints one
   snapshot and sends the result back to rank 0, which assembles the
   :class:`~beorn.structs.temporal_cube.TemporalCube`.

**Level 2 — mass-bin parallelism within a snapshot**
   Within a single snapshot, painting work is split by mass × alpha bin:

   - **CPU (numpy backend):** a :class:`~concurrent.futures.ProcessPoolExecutor`
     with ``parameters.simulation.cores`` workers processes bins concurrently.
     Each worker calls :meth:`~beorn.painting.coordinator.PaintingCoordinator.paint_single_mass_bin`
     and returns a Fourier-space contribution; the main process accumulates them.

   - **GPU (jax / torch backend):** bins are processed serially in the main
     process.  The GPU provides internal parallelism for each FFT, and all
     Fourier operations remain on-device until the final inverse transforms.

**Fourier-space accumulation**
   Instead of performing an inverse FFT per bin (which would cost
   :math:`N_\text{bins} \times 3` IFFTs per snapshot), all per-bin contributions
   are summed in Fourier space.  Only **3 inverse FFTs** are performed at the
   end of each snapshot — one each for ``xHII``, ``Temp``, and ``xal`` — saving
   an :math:`O(N_\text{bins})` factor in transform cost.

ODE solver options
------------------

Both :meth:`~beorn.precomputation.solver.RadiationProfileSolver.R_bubble` and
:meth:`~beorn.precomputation.solver.RadiationProfileSolver.rho_heat` use
:func:`scipy.integrate.solve_ivp`.  The method and tolerances are configurable:

.. code-block:: python

   parameters.solver.ode_method = 'LSODA'   # auto-detects stiffness
   parameters.solver.ode_rtol   = 1e-2
   parameters.solver.ode_atol   = 1e-2

``'LSODA'`` is a good all-round choice; ``'Radau'`` or ``'BDF'`` are faster
when the system is strongly stiff (high-z runs with dense IGM or fine redshift
grids).
