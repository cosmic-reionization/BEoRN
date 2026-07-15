# scripts/

Standalone helper scripts for working with a BEoRN checkout — not part of the
installed package. Organized into sections below; add new sections here as
more scripts are added.

## HPC setup scripts

Generic SLURM job templates for setting up and validating a BEoRN environment
on a cluster where the **login node** you edit/submit from may not match the
**compute node** your job actually runs on — different CPU architecture,
different (often shared) GPU, or both. See issue #44 for the background and a
worked example (the "Arrhenius cluster (NAISS) case" comment).

Run in order the first time you set up a new cluster:

| Script | Purpose |
|---|---|
| `hpc_setup_01_check_node_environment.sh` | Reports the compute node's CPU architecture, GPU visibility/contention, and what a site-provided Python module (if any) already includes. |
| `hpc_setup_02_check_wheel_availability.sh` | Dry-run `pip install` checks (no install) that BEoRN's GPU-capable dependencies — especially `jaxlib` / `jax-cuda12-plugin` / `jax-cuda12-pjrt`, the pieces that actually give jax GPU support — have prebuilt wheels for this architecture, before committing to a full build. |
| `hpc_setup_03_build_beorn_env.sh` | Builds the environment: a venv layered on the site's Python module (if any), the missing core deps, `torch`/`jax[cuda12]`, and an editable `--no-deps` install of BEoRN itself. Ends with a GPU-visibility check for both frameworks. |
| `hpc_setup_04_run_output_consistency.sh` | Validates the environment built in step 3 by running the MPI-parallel `tests/test_output_consistency.py` as a SLURM job. |

All four use the same placeholder convention — `YOUR_HPC_ACCOUNT`,
`/path/to/...` — copy them, fill in the placeholders for your cluster, and
drop the copies wherever you keep job scripts (they don't need to live
inside this repo).

### Key things to get right

- **Check the architecture gap first** (script 01) before building anything.
  If `uname -m` differs between login and compute node, an environment built
  on one will not run on the other.
- **Prefer the site's own Python/module bundle** for `numpy`/`scipy`/`mpi4py`
  if one exists — it's usually built against the site's actual interconnect
  and CPU features, which a generic PyPI wheel is not.
- **Install BEoRN with `--no-deps`** once its other dependencies are
  satisfied (script 03). A plain `pip install -e .` will try to satisfy every
  pin in `pyproject.toml`, which can silently upgrade a site-provided package
  (`mpi4py` in particular) to a generic build detached from the site's MPI
  stack.
- **Decide the environment's final location before building.** A venv's
  `bin/activate` hardcodes its own absolute path at creation time — moving
  the environment's folder afterwards breaks activation (it still runs, it
  just silently points `PATH` at the old, now-missing location). If you do
  need to move it, `rm -rf` and rebuild rather than trying to patch the
  hardcoded paths — pip's local wheel cache makes the rebuild fast.
- **Verify GPU visibility explicitly**, not just that the install succeeded —
  both `torch` and `jax` can install cleanly and silently fall back to CPU if
  the CUDA plugin didn't match the driver or architecture.

## HPC job examples

Once an environment exists (the setup scripts above, or your own), these are
templates for actually *using* it day to day — three different ways to run
something in a SLURM job:

| Script | Purpose |
|---|---|
| `hpc_example_job_run.sh` | Plainest form: run a Python script (shown: `examples/python_scripts/lpt_native_21cm.py`) as a batch job — submit and walk away. |
| `hpc_example_job_notebook.sh` | Execute a notebook headlessly via `nbclient` (shown: `examples/artificial_halo_gpu.ipynb`), overwriting it in place with fresh outputs — good for validation/benchmarking notebooks that don't need a human watching. |
| `hpc_example_job_interactive.sh` | Start a live Jupyter server on a compute node and keep it running for the job's walltime, so you can `sbatch` once and then keep working interactively in a notebook — see below. |

### Running a sbatch job to work in a notebook interactively

Yes — `hpc_example_job_interactive.sh` is a batch job whose only "work" is
running `jupyter lab` in the foreground, which keeps the SLURM allocation
alive for as long as Jupyter runs (until `--time` expires or you `scancel`
it). No `srun --pty` interactive shell is involved. The recipe:

1. `sbatch hpc_example_job_interactive.sh`.
2. Once the job starts, its log shows the compute node's hostname, the port,
   and (from Jupyter itself) a token URL.
3. From your laptop, SSH-tunnel *through the login node* to that compute node
   and port (compute nodes usually aren't reachable directly from outside
   the cluster): `ssh -N -L 8888:gpu-042:8888 you@login.yourcluster.edu`.
4. Open the token URL in your browser, or point VS Code's "Jupyter: Connect
   to Existing Server" at it — either way you're talking to the compute
   node's Jupyter, with the exact GPU and environment the job was allocated.
5. `scancel` the job when you're done.

Jupyter needs `--ip=0.0.0.0` for step 3's tunnel to reach it (a compute
node's `127.0.0.1` isn't visible from the login node), but its own token
auth still protects it, and compute-node networks are typically not
internet-reachable to begin with.

### Key things to get right

- **`--time` here means "how long you want to keep working"**, not "how long
  one computation takes" — size it like an interactive session, not a batch
  computation.
- **Read the job's log for the actual hostname/port/token** before tunneling
  — don't assume the placeholders in the script match what SLURM allocated.
- **`scancel` when done.** An interactive job holds a GPU allocation for its
  full `--time` unless you end it explicitly.

