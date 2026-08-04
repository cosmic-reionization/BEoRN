======================================
Full run example (non-interactive)
======================================

This file serves as a reference for a full run, in a plain Python script instead
of a notebook. This can be used on remote systems, e.g. in an HPC environment
(via ``sbatch``/``srun``, or just ``python lpt_native_21cm.py`` in a terminal).

Unlike a run driven by external N-body input, this example is fully
self-contained: it generates its own cosmological density field and halo
catalogues internally (Lagrangian Perturbation Theory + conditional halo mass
function), so no external simulation data or ``file_root`` needs to be set up
first.

This example makes the following assumptions:

 - the global parameters for this run are specified as a YAML file under
   ``./param.yaml`` (next to the script; see that file for the full parameter
   schema and comments on every field)
 - intermediate and output data is written to ``./cache/`` and ``./output/``,
   created automatically next to the script on first run

Inspect the results afterwards with ``plot_results.ipynb`` in the same folder,
which reads the saved grids back without recomputing anything.

.. literalinclude:: lpt_native_21cm.py
