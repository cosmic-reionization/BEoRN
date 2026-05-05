"""Painting coordinator: orchestrate conversion of 1D profiles into 3D maps."""
import os
import time
from datetime import timedelta
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.integrate import trapezoid as _trapz
from scipy.interpolate import interp1d
import h5py
from pathlib import Path
from tqdm.auto import tqdm
try:
    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor
    MPI_ENABLED = True
except (ImportError, RuntimeError):
    MPI_ENABLED = False

from .helpers import (
    TQDM_KWARGS,
    _GPU_BACKENDS,
    _resolve_fft_backend,
    precompute_fft,
    fourier_multiply_kernel,
    ifft_field,
    make_fourier_accumulator,
    profile_to_3Dkernel,
    stacked_lyal_kernel,
    stacked_T_kernel,
)
from .spread  import spreading_excess_fast
from ..cosmo import T_adiab_fluctu
from ..couplings import S_alpha
from ..io.handler import Handler
from ..structs.radiation_profiles import RadiationProfiles, RadiationProfilesFStarGrid
from ..structs.parameters import Parameters
from ..structs.coeval_cube import CoevalCube
from ..structs.temporal_cube import TemporalCube
from ..structs.halo_catalog import HaloCatalog
from ..load_input_data.base import BaseLoader


class PaintingCoordinator:
    @staticmethod
    def _profile_redshift_array(z_history) -> np.ndarray:
        """Materialize profile redshifts as a small in-memory array.

        Loaded profile objects may expose ``z_history`` either as a NumPy array
        or as an h5py Dataset. The coordinator only needs this small 1D vector
        for nearest-neighbor matching, so materializing it here keeps the rest
        of the large profile cube lazy.
        """
        return np.asarray(z_history[...]) if isinstance(z_history, h5py.Dataset) else np.asarray(z_history)

    @staticmethod
    def _small_profile_array(values) -> np.ndarray:
        """Materialize a small 1D HDF5-backed profile array when needed."""
        return np.asarray(values[...]) if isinstance(values, h5py.Dataset) else np.asarray(values)

    def _solver_grid_config(self):
        """Return the parameter section holding halo mass / alpha bin definitions."""
        return getattr(self.parameters, "solver", self.parameters.simulation)

    def _raise_if_halos_outside_profile_coverage(
        self,
        halo_catalog: HaloCatalog,
        mass_range: np.ndarray,
        zgrid: float,
    ) -> None:
        """Fail early when halos cannot be assigned to any precomputed mass/alpha bin."""
        if halo_catalog.size == 0:
            return

        solver_cfg = self._solver_grid_config()
        alpha_edges = self._small_profile_array(solver_cfg.halo_mass_accretion_alpha)
        alpha_indices = np.searchsorted(alpha_edges, halo_catalog.alphas, side="right") - 1
        alpha_ok = (alpha_indices >= 0) & (alpha_indices < alpha_edges.size - 1)
        alpha_indices_clipped = np.clip(alpha_indices, 0, alpha_edges.size - 2)

        supported_mass_min = np.full(halo_catalog.size, np.nan, dtype=float)
        supported_mass_max = np.full(halo_catalog.size, np.nan, dtype=float)
        supported_mass_min[alpha_ok] = mass_range[0, alpha_indices_clipped[alpha_ok]]
        supported_mass_max[alpha_ok] = mass_range[-1, alpha_indices_clipped[alpha_ok]]

        covered = alpha_ok & (halo_catalog.masses >= supported_mass_min) & (halo_catalog.masses < supported_mass_max)
        unsupported = np.where(~covered)[0]
        if unsupported.size == 0:
            return

        sample_details = []
        for halo_index in unsupported[:5]:
            mass = float(halo_catalog.masses[halo_index])
            alpha = float(halo_catalog.alphas[halo_index])
            if alpha_ok[halo_index]:
                alpha_bin = int(alpha_indices_clipped[halo_index])
                sample_details.append(
                    "halo[{idx}] mass={mass:.3e} Msol alpha={alpha:.3f} is outside the "
                    "supported mass range [{mass_min:.3e}, {mass_max:.3e}) Msol for "
                    "alpha bin [{alpha_min:.3f}, {alpha_max:.3f})".format(
                        idx=int(halo_index),
                        mass=mass,
                        alpha=alpha,
                        mass_min=float(supported_mass_min[halo_index]),
                        mass_max=float(supported_mass_max[halo_index]),
                        alpha_min=float(alpha_edges[alpha_bin]),
                        alpha_max=float(alpha_edges[alpha_bin + 1]),
                    )
                )
            else:
                sample_details.append(
                    "halo[{idx}] mass={mass:.3e} Msol alpha={alpha:.3f} is outside the "
                    "supported alpha range [{alpha_min:.3f}, {alpha_max:.3f})".format(
                        idx=int(halo_index),
                        mass=mass,
                        alpha=alpha,
                        alpha_min=float(alpha_edges[0]),
                        alpha_max=float(alpha_edges[-1]),
                    )
                )

        suggestion = (
            "Increase solver.halo_mass_bin_max and recompute the profiles, "
            "or narrow the halo_mass_accretion_alpha grid."
        )
        current_mass_bin_max = getattr(solver_cfg, "halo_mass_bin_max", None)
        unsupported_with_valid_alpha = unsupported[alpha_ok[unsupported]]
        if current_mass_bin_max is not None and unsupported_with_valid_alpha.size > 0:
            required_ratio = np.max(
                halo_catalog.masses[unsupported_with_valid_alpha]
                / supported_mass_max[unsupported_with_valid_alpha]
            )
            suggested_mass_bin_max = float(current_mass_bin_max) * float(required_ratio) * 1.1
            suggestion = (
                f"Increase solver.halo_mass_bin_max to at least {suggested_mass_bin_max:.3e} "
                "Msol and recompute the profiles, or narrow the halo_mass_accretion_alpha grid."
            )

        raise RuntimeError(
            f"Halo catalog at z={float(zgrid):.3f} contains {unsupported.size} halo(s) outside "
            "the precomputed mass/alpha coverage. Painting would later drop those halos and "
            "fail the halo-count invariant. Coverage is alpha-dependent even when the global "
            f"profile mass range [{float(np.min(mass_range)):.3e}, {float(np.max(mass_range)):.3e}] "
            f"Msol looks wide enough. Examples: {'; '.join(sample_details)}. {suggestion}"
        )

    def _nearest_profile_redshift_index(self, z_history: np.ndarray, z_index: int) -> int:
        """Return the nearest profile redshift index for a loader snapshot index."""
        z_history = self._profile_redshift_array(z_history)
        snap_z = float(self.loader.redshifts[z_index])
        if z_history.size == 1:
            return 0
        return int(np.argmin(np.abs(z_history - snap_z)))

    def _warn_if_profile_redshift_far(self, z_history: np.ndarray, profile_z_index: int, z_index: int) -> None:
        """Warn when the nearest profile redshift is far from the target snapshot."""
        z_history = self._profile_redshift_array(z_history)
        snap_z = float(self.loader.redshifts[z_index])
        profile_z = float(z_history[profile_z_index])
        if abs(profile_z - snap_z) > 0.5:
            self.logger.warning(
                f"Snapshot z={snap_z:.3f} is more than Δz=0.5 away from the nearest "
                f"profile redshift z={profile_z:.3f}. "
                "Consider adding more steps to solver.redshifts."
            )

    @staticmethod
    def _accumulate_bin_result(result, Grid_xHII, Grid_Temp, Grid_xal):
        """Add per-bin contribution arrays returned by paint_single_mass_bin into the running totals."""
        xHII_c, lyal_c, temp_c = result
        if xHII_c is not None:
            Grid_xHII += xHII_c
        if lyal_c is not None:
            Grid_xal += lyal_c
        if temp_c is not None:
            Grid_Temp += temp_c

    def _load_fstar_profiles_z_slice(self, profiles_path: Path, profile_z_index: int) -> RadiationProfilesFStarGrid:
        """Load only one redshift slice of f_st-grid profiles from disk.

        This avoids loading the full (mass, alpha, f_st, z) profile grid into memory.
        The returned object keeps a trailing z-dimension of length 1 so existing
        painting code can reuse the same indexing pattern by using `z_local=0`.
        """
        profiles_path = Path(profiles_path)
        with h5py.File(profiles_path, "r") as h5:
            r_grid_cell = h5["r_grid_cell"][...]
            r_lyal = h5["r_lyal"][...]
            f_st_grid = h5["f_st_grid"][...]

            z_all = h5["z_history"][...]
            z_val = float(z_all[profile_z_index])
            z_history = np.array([z_val], dtype=z_all.dtype)

            halo_mass_bins = h5["halo_mass_bins"][..., profile_z_index][..., None]
            R_bubble = h5["R_bubble"][..., profile_z_index][..., None]
            rho_xray = h5["rho_xray"][..., profile_z_index][..., None]
            rho_heat = h5["rho_heat"][..., profile_z_index][..., None]
            rho_alpha = h5["rho_alpha"][..., profile_z_index][..., None]

        # Use the coordinator parameters (small object) rather than materializing
        # Parameters from the HDF5 file.
        obj = RadiationProfilesFStarGrid(
            parameters=self.parameters,
            z_history=z_history,
            halo_mass_bins=halo_mass_bins,
            f_st_grid=f_st_grid,
            rho_xray=rho_xray,
            rho_heat=rho_heat,
            rho_alpha=rho_alpha,
            R_bubble=R_bubble,
            r_lyal=r_lyal,
            r_grid_cell=r_grid_cell,
        )
        obj._file_path = profiles_path
        return obj


    def _format_cache_value(self, value) -> str:
        """Format cache-key values into stable path-safe strings."""
        if value is None:
            return "none"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("/", "-")

    def _paint_cache_namespace(self, profiles: RadiationProfiles | RadiationProfilesFStarGrid) -> str:
        """Return the cache namespace for painted coeval outputs."""
        if isinstance(profiles, RadiationProfilesFStarGrid):
            source = self.parameters.source
            distribution = self._format_cache_value(getattr(source, "f_st_paint_distribution", "lognormal"))
            sigma = self._format_cache_value(getattr(source, "f_st_paint_sigma", 0.5))
            seed = self._format_cache_value(getattr(source, "f_st_paint_seed", None))
            return f"painted_output_fstar_dist_{distribution}_sigma_{sigma}_seed_{seed}"
        return "painted_output_legacy"
    
    """Orchestrate painting of 1D radiation profiles to 3D grids.

    The coordinator handles loading halo catalogs / density fields,
    dispatching painting work across processes (or MPI ranks), and
    writing the resulting :class:`CoevalCube` / :class:`TemporalCube`
    outputs.

    Attributes:
        parameters (Parameters): Simulation parameters.
        loader (BaseLoader): Loader class providing halo catalogs and density fields for each redshift.
        output_handler (Handler): IO handler used to save results.
        cache_handler (Handler|None): Optional cache for intermediate painted outputs.
    """
    logger = logging.getLogger(__name__)

    def __init__(
            self,
            parameters: Parameters,
            loader: type[BaseLoader],
            output_handler: Handler,
            cache_handler: Handler = None,
            force_recompute: bool = False,
        ):
        """
        Initialize the Painter class with the given parameters.

        Args:
            parameters (Parameters): The parameters object containing cosmological and simulation parameters.
            loader (BaseLoader): The loader class responsible for providing halo catalogs and density fields.
            output_handler (Handler): The handler for saving the painted output data.
            cache_handler (Handler, optional): The handler for loading and saving cache data that can be reused between runs.
            force_recompute (bool): If True, repaint all snapshots even when output files already exist.
                Defaults to False (existing snapshots are reused).
        """
        self.parameters = parameters
        self.output_handler = output_handler
        self.cache_handler = cache_handler
        self.loader = loader
        self.snapshot_count = self.loader.redshifts.size
        self.force_recompute = force_recompute


    def paint_full(
        self,
        radiation_profiles: RadiationProfiles,
        redshift_subset: "list[float] | None" = None,
    ) -> TemporalCube:
        """Paint redshift snapshots and return a :class:`TemporalCube`.

        By default all snapshots known to the loader are painted.  Pass
        ``redshift_subset`` to paint only the snapshots nearest to the
        requested redshifts — useful when data is sparse or you want a
        quick run at a few selected epochs.

        The method chooses an MPI-enabled execution path when MPI is available;
        otherwise it runs a local loop.

        Args:
            radiation_profiles (RadiationProfiles): Precomputed 1D profiles.
            redshift_subset (list[float] | None): If given, only the
                **halo-bearing** snapshots nearest to each requested redshift
                are painted.  Density-only snapshots are excluded from
                matching.  A warning is emitted and the entry skipped when no
                halo snapshot is within Δz=1.  ``None`` (default) paints all
                snapshots.

        Returns:
            TemporalCube: HDF5-backed collection of painted 3D snapshots.
        """
        active_indices = self._resolve_painting_indices(redshift_subset)
        self.logger.info(self.parameters.summary_str())
        # Use MPI only when there are actually multiple ranks; a single-rank
        # MPICommExecutor runs futures in the main process, which causes HDF5
        # file-handle conflicts with the already-open TemporalCube file.
        if MPI_ENABLED and MPI.COMM_WORLD.Get_size() > 1:
            return self.paint_mpi(radiation_profiles, active_indices)
        else:
            return self.paint_simple_loop(radiation_profiles, active_indices)

    def _resolve_painting_indices(
        self, redshift_subset: "list[float] | None"
    ) -> list:
        """Return the loader snapshot indices to paint.

        If ``redshift_subset`` is ``None`` every snapshot is included.
        Otherwise, for each requested redshift the nearest **halo-bearing**
        snapshot is selected (density-only snapshots are excluded from
        matching).  A warning is emitted and the entry is skipped when no
        halo snapshot is within Δz=1.  Duplicates are removed and the result
        is sorted.
        """
        if redshift_subset is None:
            return list(range(self.snapshot_count))

        all_z = self.loader.redshifts
        # Indices that actually have a halo catalog attached.
        halo_indices = [
            i for i, cat in enumerate(self.loader.catalogs) if cat is not None
        ]
        if not halo_indices:
            self.logger.warning(
                "redshift_subset provided but the loader has no halo catalogs at any snapshot."
            )
            return []

        halo_z = all_z[halo_indices]
        indices = []
        for z_target in redshift_subset:
            nearest_pos = int(np.argmin(np.abs(halo_z - z_target)))
            i = halo_indices[nearest_pos]
            if abs(halo_z[nearest_pos] - z_target) > 1.0:
                self.logger.warning(
                    f"Requested z={z_target:.3f} has no halo snapshot within Δz=1.0 "
                    f"(nearest halo snapshot is z={halo_z[nearest_pos]:.3f}) — skipping."
                )
            elif i not in indices:
                indices.append(i)
        indices = sorted(indices)
        self.logger.info(
            f"redshift_subset: painting {len(indices)} of {self.snapshot_count} "
            f"available snapshot(s) ({len(halo_indices)} have halo catalogs)."
        )
        return indices


    def paint_mpi(self, radiation_profiles: RadiationProfiles, active_indices: list) -> TemporalCube|None:
        """MPI-enabled painting: distribute redshift snapshots across ranks.

        Only the master rank writes and returns the final HDF5 dataset; worker ranks perform painting and send partial results to be appended by the master. This function assumes an MPI communicator is available.

        Args:
            radiation_profiles (RadiationProfiles): Precomputed profiles.
            active_indices (list): Snapshot indices to paint (subset of range(snapshot_count)).

        Returns:
            TemporalCube: The assembled temporal cube (returned only on master).
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # barrier to ensure all processes reach this point before proceeding: this ensures that all prerequisites are met
        # probably optional
        MPI.COMM_WORLD.Barrier()

        self.logger.debug(f"Starting painter process on rank {rank}.")
        if rank == 0:
            self.logger.info(f"Setting up {comm.Get_size()} painting processes for MPI.")
            cube = TemporalCube.create_empty(
                self.parameters,
                self.output_handler.file_root,
                **self.output_handler.write_kwargs
            )
            if self.force_recompute:
                missing_indices = list(active_indices)
                n_cached = sum(
                    1 for i in active_indices
                    if cube.snapshot_path(self.loader.redshifts[i]).exists()
                )
                if n_cached:
                    self.logger.info(
                        f"force_recompute=True: repainting {n_cached} already-present snapshot(s) "
                        f"plus {len(active_indices) - n_cached} new snapshot(s)."
                    )
            else:
                missing_indices = [
                    i for i in active_indices
                    if not cube.snapshot_path(self.loader.redshifts[i]).exists()
                ]
                n_cached = len(active_indices) - len(missing_indices)
                if n_cached:
                    self.logger.info(
                        f"Found {n_cached} already-painted snapshot(s) — skipping "
                        f"(set force_recompute=True to repaint them)."
                    )
            self.logger.info(
                f"Submitting {len(missing_indices)}/{self.snapshot_count} snapshot(s) to MPI workers."
            )
        else:
            missing_indices = None

        missing_indices = comm.bcast(missing_indices, root=0)

        # since workers load radiation profiles from file, ensure the file exists
        if radiation_profiles._file_path is None:
            self.output_handler.write_file(self.parameters, radiation_profiles)

        with MPICommExecutor(comm) as executor:
            if executor is not None:
                futures = {
                    executor.submit(self.paint_single, index, profiles_path=radiation_profiles._file_path): index
                    for index in missing_indices
                }

                if rank == 0:
                    for future in as_completed(futures):
                        loop_index = futures[future]
                        grid_data = future.result()
                        self.logger.info(
                            "Received finished snapshot from worker for z_index=%d (z=%.2f). Appending to temporal output.",
                            loop_index,
                            float(grid_data.z),
                        )
                        cube.append(grid_data, loop_index)
                        self.logger.info(
                            "Appended snapshot to temporal output for z_index=%d (z=%.2f).",
                            loop_index,
                            float(grid_data.z),
                        )

                    self.logger.info(f"Painting of {self.snapshot_count} snapshots done.")
                    return TemporalCube.read(file_path=cube._file_path, parameters=self.parameters)

        return None


    def paint_simple_loop(self, radiation_profiles: RadiationProfiles, active_indices: list) -> TemporalCube:
        """Paint snapshots in a single (possibly multi-process) loop.

        Creates the ``igm_data_*/`` output directory and iterates over
        redshift snapshots.  Snapshots whose ``CoevalCube_z{z:.3f}.h5`` file
        already exists are skipped unless ``force_recompute=True`` was passed
        to the constructor, allowing interrupted runs to resume automatically.

        Args:
            radiation_profiles (RadiationProfiles): Precomputed profiles.
            active_indices (list): Snapshot indices to paint (subset of range(snapshot_count)).

        Returns:
            TemporalCube: The assembled temporal cube loaded from the output directory.
        """
        cube = TemporalCube.create_empty(
            self.parameters,
            self.output_handler.file_root,
            **self.output_handler.write_kwargs
        )

        self.logger.info(
            f"Painting profiles onto grid for {len(active_indices)} of "
            f"{self.snapshot_count} redshift snapshots. "
            f"Using {self.parameters.simulation.cores} processes on a single node."
        )

        with tqdm(active_indices, **TQDM_KWARGS) as pbar:
            for loop_index in pbar:
                z = self.loader.redshifts[loop_index]
                z_history = self._profile_redshift_array(radiation_profiles.z_history)
                # Use the profile-matched redshift for the filename check: paint_single
                # writes CoevalCube with z=zgrid (nearest profile z), so we must look
                # for that name rather than the raw loader redshift which may differ.
                profile_z_index = int(np.argmin(np.abs(z_history - z)))
                zgrid = float(z_history[profile_z_index])
                pbar.set_postfix(z=f"{zgrid:.3f}", refresh=False)
                if cube.snapshot_path(zgrid).exists():
                    if self.force_recompute:
                        self.logger.info(f"Found painted output for z={zgrid:.3f} — repainting (force_recompute=True).")
                    else:
                        self.logger.info(f"Found painted output for z={zgrid:.3f} — skipping (set force_recompute=True to repaint).")
                        continue
                grid_data = self.paint_single(loop_index, radiation_profiles)
                cube.append(grid_data, loop_index)

        self.logger.info(f"Painting of {len(active_indices)} snapshots done.")
        return TemporalCube.read(file_path=cube._file_path, parameters=self.parameters)


    def paint_single(self, z_index: int, profiles: RadiationProfiles = None, profiles_path: Path = None) -> CoevalCube:
        """Paint a single redshift snapshot into a :class:`CoevalCube`.

        The method loads (or receives) the radiation profiles, dispatches per-mass-bin
        painting tasks and performs post-processing (excess spreading,
        background addition, and derived-field computation).

        Args:
            z_index (int): Index of the snapshot to paint.
            profiles (RadiationProfiles|None): Profiles object (optional).
            profiles_path (Path|None): Path to on-disk profiles (used by MPI workers) (optional).

        Returns:
            CoevalCube: Painted coeval cube for the requested snapshot.

        Notes:
            This method requires the information about the radiation profiles. It needs to be passed either as an object or as a path from which to load it.
        """
        if profiles is None and profiles_path is None:
            raise ValueError("Either profiles or profiles_path must be provided to paint_single.")
        if profiles is None:
            # Detect f_st-grid profile files by dataset presence and slice-load only the needed z.
            try:
                with h5py.File(profiles_path, "r") as h5:
                    is_fstar = "f_st_grid" in h5
                    if is_fstar:
                        profile_z_index = self._nearest_profile_redshift_index(h5["z_history"][...], z_index)
            except Exception:
                is_fstar = False
            if is_fstar:
                profiles = self._load_fstar_profiles_z_slice(profiles_path, profile_z_index)
            else:
                profiles = RadiationProfiles.read(profiles_path)

        if isinstance(profiles, RadiationProfilesFStarGrid):
            return self.paint_single_fstar(z_index, profiles)

        if self.cache_handler:
            try:
                grid_data = self.cache_handler.load_file(
                    self.parameters, 
                    CoevalCube, 
                    z_index=z_index,
                    cache_namespace=self._paint_cache_namespace(profiles)
                )
                self.logger.info(f"Found painted output in cache for {z_index=}. Skipping.")
                grid_data.to_arrays()
                return grid_data

            except FileNotFoundError:
                # there is no cache or the cache does not contain the halo catalog - compute it fresh
                self.logger.debug("Painted output not found in cache. Processing now")


        iteration_start_time = time.time()
        zero_grid = np.zeros((self.parameters.simulation.Ncell, self.parameters.simulation.Ncell, self.parameters.simulation.Ncell))

        halo_catalog = self.loader.load_halo_catalog(z_index)
        delta_b = self.loader.load_density_field(z_index)

        if profiles.z_history.size == 1:
            profile_z_index = 0
        else:
            # Find the profile index whose redshift is nearest to this snapshot's redshift.
            # When solver.redshifts and simulation.snapshot_redshifts are the same grid this is a
            # direct lookup; when snapshot_redshifts is a coarser subset the nearest
            # profile step is used.
            profile_z_index = self._nearest_profile_redshift_index(profiles.z_history, z_index)
        self._warn_if_profile_redshift_far(profiles.z_history, profile_z_index, z_index)

        zgrid = profiles.z_history[profile_z_index]
        mass_range = profiles.halo_mass_bins[..., profile_z_index]
        solver_cfg = self._solver_grid_config()

        # log some information about the current "paintable range"
        alphas = solver_cfg.halo_mass_accretion_alpha
        self.logger.debug(
            f"Got {mass_range.shape[0]}x{mass_range.shape[1]} profiles. Range: "
            f"alpha={alphas[0]:.2f} [{mass_range[...,0].min():.2e} - {mass_range[..., 0].max():.2e} Msun] and "
            f"alpha={alphas[-1]:.2f} [{mass_range[...,-1].min():.2e} - {mass_range[..., -1].max():.2e} Msun]."
        )

        # # TODO - describe the relevance of coef
        # coef = constants.rhoc0 * self.parameters.cosmology.h0 ** 2 * self.parameters.cosmology.Ob * (1 + zgrid) ** 3 * constants.M_sun / constants.cm_per_Mpc ** 3 / constants.m_H


        # since we want to paint the halo profiles in grouped mass bins, we need to know which halos are in which mass bin
        # but there are a few short-circuits:
        # 1. if there are no halos at all -> skip the painting
        # 2. if there are halos but they lie outside the mass range -> raise an error

        if halo_catalog.masses.size == 0:
            self.logger.info(f'No halos at z={zgrid:.2f}. Returning empty grids.')
            grid_data = CoevalCube(
                parameters=self.parameters,
                z=zgrid,
                delta_b=delta_b,
                Grid_Temp=T_adiab_fluctu(zgrid, self.parameters, delta_b),
                Grid_xHII=zero_grid.copy(),
                Grid_xal=zero_grid.copy(),
            )
            return grid_data

        self._raise_if_halos_outside_profile_coverage(halo_catalog, mass_range, zgrid)

        self.logger.info(f'Painting {halo_catalog.size} halos at {zgrid=:.2f} ({z_index=:.0f}).')

        nGrid = self.parameters.simulation.Ncell
        store_grids = self.parameters.simulation.store_grids
        cores = self.parameters.simulation.cores
        fft_backend = _resolve_fft_backend(self.parameters.simulation.fft_backend)
        # GPU backends (jax/torch) run bins serially in the main process; the GPU
        # provides internal parallelism for each FFT.  CPU numpy uses ProcessPoolExecutor.
        use_gpu = fft_backend in _GPU_BACKENDS
        use_multiprocess = cores > 1 and not use_gpu
        fft_workers = 1 if use_gpu else (-1 if cores <= 1 else max(1, (os.cpu_count() or 1) // cores))

        # Fourier-space accumulators: contributions from all bins are summed here,
        # and exactly 3 inverse FFTs are performed at the end (regardless of N_bins).
        fourier_shape = (nGrid, nGrid, nGrid // 2 + 1)
        accum_xHII = make_fourier_accumulator(fourier_shape, fft_backend) if "Grid_xHII" in store_grids else None
        accum_lyal  = make_fourier_accumulator(fourier_shape, fft_backend) if "Grid_xal"  in store_grids else None
        accum_temp  = make_fourier_accumulator(fourier_shape, fft_backend) if "Grid_Temp" in store_grids else None

        def _add_fourier(result):
            nonlocal accum_xHII, accum_lyal, accum_temp
            fa_x, fa_l, fa_t = result
            if fa_x is not None and accum_xHII is not None:
                accum_xHII = accum_xHII + fa_x
            if fa_l is not None and accum_lyal is not None:
                accum_lyal = accum_lyal + fa_l
            if fa_t is not None and accum_temp is not None:
                accum_temp = accum_temp + fa_t

        futures = []
        total_halos = 0
        executor = ProcessPoolExecutor(max_workers=cores) if use_multiprocess else None

        alpha_indices = range(len(solver_cfg.halo_mass_accretion_alpha) - 1)
        mass_indices = range(len(solver_cfg.halo_mass_bins) - 1)

        self.logger.debug(
            f"Fourier-accumulation painting: backend={fft_backend!r}, "
            f"{'GPU serial' if use_gpu else f'{cores} workers' if use_multiprocess else 'serial CPU'}."
        )
        start_time = time.time()

        try:
            for alpha_index in alpha_indices:
                loop_alpha_range = [
                    solver_cfg.halo_mass_accretion_alpha[alpha_index],
                    solver_cfg.halo_mass_accretion_alpha[alpha_index + 1]
                ]
                for mass_index in mass_indices:
                    loop_mass_range = [
                        mass_range[mass_index, alpha_index],
                        mass_range[mass_index + 1, alpha_index]
                    ]
                    halo_indices = halo_catalog.get_halo_indices(loop_alpha_range, loop_mass_range)
                    if halo_indices.size == 0:
                        continue
                    total_halos += halo_indices.size

                    profiles_of_bin = profiles.profiles_of_halo_bin(profile_z_index, alpha_index, mass_index)
                    assert not np.any(np.isnan(profiles_of_bin[0])), "R_bubble at the current range seem to be malformed (got nan values)"
                    assert not np.any(np.isnan(profiles_of_bin[1])), "rho_alpha at the current range seem to be malformed (got nan values)"
                    assert not np.any(np.isnan(profiles_of_bin[2])), "rho_heat at the current range seem to be malformed (got nan values)"

                    radial_grid = profiles.r_grid_cell[:] / (1 + zgrid)  # pMpc/h
                    kwargs = {
                        "halo_catalog": halo_catalog.at_indices(halo_indices),
                        "z": zgrid,
                        "radial_grid": radial_grid,
                        "r_lyal": profiles.r_lyal[:],
                        "profiles_of_bin": profiles_of_bin,
                        "store_grids": store_grids,
                    }

                    if executor is not None:
                        f = executor.submit(self.paint_single_mass_bin, **kwargs)
                        futures.append(f)
                    else:
                        _add_fourier(self.paint_single_mass_bin(**kwargs))

            if futures:
                for future in as_completed(futures):
                    _add_fourier(future.result())
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        assert total_halos == halo_catalog.size, (
            f"Number of painted halos ({total_halos}) does not match the halo catalog size "
            f"({halo_catalog.size}). {halo_catalog.size - total_halos} halo(s) fell outside "
            f"all mass/alpha bins. Catalog mass range: "
            f"[{halo_catalog.masses.min():.3e}, {halo_catalog.masses.max():.3e}] M☉ — "
            f"profile bins cover [{self.parameters.solver.halo_mass_bin_min:.3e}, "
            f"{self.parameters.solver.halo_mass_bin_max:.3e}] M☉. "
            "Widen halo_mass_bin_min/max in the parameters and recompute profiles "
            "(force_recompute=True)."
        )

        # 3 inverse FFTs total — regardless of how many mass bins were processed.
        Grid_xHII = ifft_field(accum_xHII, (nGrid, nGrid, nGrid), backend=fft_backend, workers=fft_workers) if accum_xHII is not None else zero_grid.copy()
        Grid_xal  = ifft_field(accum_lyal,  (nGrid, nGrid, nGrid), backend=fft_backend, workers=fft_workers) if accum_lyal  is not None else zero_grid.copy()
        Grid_Temp = ifft_field(accum_temp,  (nGrid, nGrid, nGrid), backend=fft_backend, workers=fft_workers) if accum_temp  is not None else zero_grid.copy()

        self.logger.info(f'Profile painting took {timedelta(seconds=time.time() - start_time)}.')

        ## Excess spreading
        start_time = time.time()
        Grid_xHII = spreading_excess_fast(self.parameters, Grid_xHII)

        self.logger.info(f'Redistributing excess photons from the overlapping regions took {timedelta(seconds=time.time() - start_time)}.')

        ## Post processing of the already filled grids
        start_time = time.time()

        # take into account the background temperature
        Grid_Temp += T_adiab_fluctu(zgrid, self.parameters, delta_b)

        # Enforce a minimum ionization fraction
        Grid_xHII[Grid_xHII < self.parameters.source.min_xHII_value] = self.parameters.source.min_xHII_value

        # Include fluctuations
        if self.parameters.simulation.compute_s_alpha_fluctuations:
            self.logger.debug('Including Salpha fluctuations in dTb')
            Grid_xal *= S_alpha(zgrid, Grid_Temp, 1 - Grid_xHII) / (4 * np.pi)
            # We divide by 4pi to go to sr**-1 units
        else:
            self.logger.debug('NOT including Salpha fluctuations in dTb')
            Grid_xal *= S_alpha(zgrid, np.mean(Grid_Temp), 1 - np.mean(Grid_xHII)) / (4 * np.pi)


        # if Rsmoothing > 0:
        #     self.logger.info(f'Smoothing the fields with {Rsmoothing=}')
        #     Grid_xal = smooth_field(Grid_xal, Rsmoothing, LBox, nGrid)
        #     Grid_Temp = smooth_field(Grid_Temp, Rsmoothing, LBox, nGrid)
        #     #Grid_xHII = smooth_field(Grid_xHII, Rsmoothing, LBox, nGrid)
        #     #delta_b   = smooth_field(delta_b, Rsmoothing, LBox, nGrid)
        #     # TODO - why are the other fields not smoothed?


        self.logger.info(f'Postprocessing of the grids took {timedelta(seconds=time.time() - start_time)}.')
        self.logger.info(f'Current snapshot took {timedelta(seconds=time.time() - iteration_start_time)}.')

        grid_data = CoevalCube(
            parameters = self.parameters,
            z = zgrid,
            delta_b = delta_b,
            Grid_Temp = Grid_Temp,
            Grid_xHII = Grid_xHII,
            Grid_xal = Grid_xal,
        )

        if self.cache_handler:
            self.logger.info(
                "Writing painted snapshot to cache for z_index=%d (z=%.2f).",
                z_index,
                float(zgrid),
            )
            self.cache_handler.write_file(
                self.parameters, 
                grid_data, 
                z_index=z_index,
                cache_namespace=self._paint_cache_namespace(profiles)
            )
            self.logger.info(
                "Finished writing painted snapshot to cache for z_index=%d (z=%.2f).",
                z_index,
                float(zgrid),
            )
        return grid_data
    
    def paint_single_fstar(self, z_index: int, profiles: RadiationProfilesFStarGrid) -> CoevalCube:
        """Paint a single snapshot using stochastic per-halo f_st sampling."""
        if self.cache_handler:
            try:
                grid_data = self.cache_handler.load_file(
                    self.parameters, 
                    CoevalCube, 
                    z_index=z_index,
                    cache_namespace=self._paint_cache_namespace(profiles)
                )
                self.logger.info(f"Found painted output in cache for {z_index=}. Skipping.")
                grid_data.to_arrays()
                return grid_data
            except FileNotFoundError:
                self.logger.debug("Painted output not found in cache. Processing now")
    
        iteration_start_time = time.time()
        zero_grid = np.zeros((
            self.parameters.simulation.Ncell,
            self.parameters.simulation.Ncell,
            self.parameters.simulation.Ncell
        ))

        halo_catalog = self.loader.load_halo_catalog(z_index)
        delta_b = self.loader.load_density_field(z_index)

        if profiles.z_history.size == 1:
            profile_z_index = 0
        else:
            profile_z_index = self._nearest_profile_redshift_index(profiles.z_history, z_index)
        self._warn_if_profile_redshift_far(profiles.z_history, profile_z_index, z_index)
        zgrid = profiles.z_history[profile_z_index]
        mass_range = profiles.halo_mass_bins[..., profile_z_index]
        solver_cfg = self._solver_grid_config()

        rng = self._make_f_st_rng(z_index)  # create a random number generator for sampling f_st values, seeded by the redshift index to ensure reproducibility

        alphas = solver_cfg.halo_mass_accretion_alpha
        self.logger.debug(
            f"Got {mass_range.shape[0]}x{mass_range.shape[1]} profile-bin edges. Range: "
            f"alpha={alphas[0]:.2f} [{mass_range[...,0].min():.2e} - {mass_range[..., 0].max():.2e} Msun] and "
            f"alpha={alphas[-1]:.2f} [{mass_range[...,-1].min():.2e} - {mass_range[..., -1].max():.2e} Msun]."
        )

        if halo_catalog.masses.size == 0:
            self.logger.info(f'No halos at z={float(zgrid):.2f}. Returning empty grids.')
            return CoevalCube(
                parameters=self.parameters,
                z=zgrid,
                delta_b=delta_b,
                Grid_Temp=T_adiab_fluctu(zgrid, self.parameters, delta_b),
                Grid_xHII=zero_grid.copy(),
                Grid_xal=zero_grid.copy(),
            )

        self._raise_if_halos_outside_profile_coverage(halo_catalog, mass_range, zgrid)

        self.logger.info(f'Painting {halo_catalog.size} halos at {zgrid=:.2f} ({z_index=:.0f}) using stochastic f_st sampling.')

        nGrid = self.parameters.simulation.Ncell
        store_grids = self.parameters.simulation.store_grids
        cores = self.parameters.simulation.cores
        fft_backend = _resolve_fft_backend(self.parameters.simulation.fft_backend)
        use_gpu = fft_backend in _GPU_BACKENDS
        use_multiprocess = cores > 1 and not use_gpu
        fft_workers = 1 if use_gpu else (-1 if cores <= 1 else max(1, (os.cpu_count() or 1) // cores))

        fourier_shape = (nGrid, nGrid, nGrid // 2 + 1)
        accum_xHII = make_fourier_accumulator(fourier_shape, fft_backend) if "Grid_xHII" in store_grids else None
        accum_lyal  = make_fourier_accumulator(fourier_shape, fft_backend) if "Grid_xal"  in store_grids else None
        accum_temp  = make_fourier_accumulator(fourier_shape, fft_backend) if "Grid_Temp" in store_grids else None

        def _add_fourier(result):
            nonlocal accum_xHII, accum_lyal, accum_temp
            fa_x, fa_l, fa_t = result
            if fa_x is not None and accum_xHII is not None:
                accum_xHII = accum_xHII + fa_x
            if fa_l is not None and accum_lyal is not None:
                accum_lyal = accum_lyal + fa_l
            if fa_t is not None and accum_temp is not None:
                accum_temp = accum_temp + fa_t

        futures = []
        total_halos = 0
        executor = ProcessPoolExecutor(max_workers=cores) if use_multiprocess else None

        alpha_indices = range(len(solver_cfg.halo_mass_accretion_alpha) - 1)
        mass_indices = range(len(solver_cfg.halo_mass_bins) - 1)

        self.logger.debug(
            f"Fourier-accumulation painting (fstar): backend={fft_backend!r}, "
            f"{'GPU serial' if use_gpu else f'{cores} workers' if use_multiprocess else 'serial CPU'}."
        )
        start_time = time.time()

        try:
            for alpha_index in alpha_indices:
                loop_alpha_range = [
                    solver_cfg.halo_mass_accretion_alpha[alpha_index],
                    solver_cfg.halo_mass_accretion_alpha[alpha_index + 1]
                ]

                for mass_index in mass_indices:
                    loop_mass_range = [
                        mass_range[mass_index, alpha_index],
                        mass_range[mass_index + 1, alpha_index]
                    ]

                    halo_indices = halo_catalog.get_halo_indices(loop_alpha_range, loop_mass_range)
                    if halo_indices.size == 0:
                        continue

                    halo_subset = halo_catalog.at_indices(halo_indices)

                    sampled_f_st = self._sample_f_st_for_halos(halo_subset.size, rng)
                    f_st_indices = self._nearest_f_st_indices(sampled_f_st, profiles.f_st_grid)

                    self.logger.debug(
                        "Sampled f_st for %d halos in mass bin %d / alpha bin %d; using %d unique profile slices.",
                        halo_subset.size,
                        mass_index,
                        alpha_index,
                        np.unique(f_st_indices).size,
                    )

                    for f_st_index in np.unique(f_st_indices):
                        subhalo_local_indices = np.where(f_st_indices == f_st_index)[0]
                        if subhalo_local_indices.size == 0:
                            continue

                        total_halos += subhalo_local_indices.size

                        profiles_of_bin = profiles.profiles_of_halo_bin(
                            profile_z_index,
                            alpha_index,
                            mass_index,
                            int(f_st_index)
                        )

                        assert not np.any(np.isnan(profiles_of_bin[0])), "R_bubble at the current range seem to be malformed (got nan values)"
                        assert not np.any(np.isnan(profiles_of_bin[1])), "rho_alpha at the current range seem to be malformed (got nan values)"
                        assert not np.any(np.isnan(profiles_of_bin[2])), "rho_heat at the current range seem to be malformed (got nan values)"

                        radial_grid = profiles.r_grid_cell[:] / (1 + zgrid)

                        kwargs = {
                            "halo_catalog": halo_subset.at_indices(subhalo_local_indices),
                            "z": zgrid,
                            "radial_grid": radial_grid,
                            "r_lyal": profiles.r_lyal[:],
                            "profiles_of_bin": profiles_of_bin,
                            "store_grids": store_grids,
                        }

                        if executor is not None:
                            f = executor.submit(self.paint_single_mass_bin, **kwargs)
                            futures.append(f)
                        else:
                            _add_fourier(self.paint_single_mass_bin(**kwargs))

            if futures:
                for future in as_completed(futures):
                    _add_fourier(future.result())
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        assert total_halos == halo_catalog.size, (
            f"Number of painted halos ({total_halos}) does not match the halo catalog size ({halo_catalog.size})."
        )

        self.logger.info(f'Profile painting took {timedelta(seconds=time.time() - start_time)}.')

        Grid_xHII = ifft_field(accum_xHII, (nGrid, nGrid, nGrid), backend=fft_backend, workers=fft_workers) if accum_xHII is not None else zero_grid.copy()
        Grid_xal  = ifft_field(accum_lyal,  (nGrid, nGrid, nGrid), backend=fft_backend, workers=fft_workers) if accum_lyal  is not None else zero_grid.copy()
        Grid_Temp = ifft_field(accum_temp,  (nGrid, nGrid, nGrid), backend=fft_backend, workers=fft_workers) if accum_temp  is not None else zero_grid.copy()

        start_time = time.time()
        Grid_xHII = spreading_excess_fast(self.parameters, Grid_xHII)
        self.logger.info(f'Redistributing excess photons from the overlapping regions took {timedelta(seconds=time.time() - start_time)}.')

        start_time = time.time()

        Grid_Temp += T_adiab_fluctu(zgrid, self.parameters, delta_b)
        Grid_xHII[Grid_xHII < self.parameters.source.min_xHII_value] = self.parameters.source.min_xHII_value

        if self.parameters.simulation.compute_s_alpha_fluctuations:
            self.logger.debug('Including Salpha fluctuations in dTb')
            Grid_xal *= S_alpha(zgrid, Grid_Temp, 1 - Grid_xHII) / (4 * np.pi)
        else:
            self.logger.debug('NOT including Salpha fluctuations in dTb')
            Grid_xal *= S_alpha(zgrid, np.mean(Grid_Temp), 1 - np.mean(Grid_xHII)) / (4 * np.pi)

        self.logger.info(f'Postprocessing of the grids took {timedelta(seconds=time.time() - start_time)}.')
        self.logger.info(f'Current snapshot took {timedelta(seconds=time.time() - iteration_start_time)}.')

        grid_data = CoevalCube(
            parameters=self.parameters,
            z=zgrid,
            delta_b=delta_b,
            Grid_Temp=Grid_Temp,
            Grid_xHII=Grid_xHII,
            Grid_xal=Grid_xal,
        )

        if self.cache_handler:
            self.logger.info(
                "Writing painted snapshot to cache for z_index=%d (z=%.2f).",
                z_index,
                float(zgrid),
            )
            self.cache_handler.write_file(
                self.parameters, 
                grid_data, 
                z_index=z_index,
                cache_namespace=self._paint_cache_namespace(profiles)
            )
            self.logger.info(
                "Finished writing painted snapshot to cache for z_index=%d (z=%.2f).",
                z_index,
                float(zgrid),
            )

        return grid_data
    
    def _sample_f_st_for_halos(self, halo_count: int, rng: np.random.Generator) -> np.ndarray:
        """Sample one stellar-fraction value per halo for f_st-grid painting.

        Optional source parameters:
        - f_st_paint_distribution: 'lognormal', 'normal', or 'uniform'
        - f_st_paint_sigma
        - f_st_paint_min
        - f_st_paint_max
        - f_st_paint_seed

        Defaults:
        - distribution = 'lognormal'
        - sigma = 0.5
        - clipping range = [f_st_grid_min, f_st_grid_max]
        """
        source = self.parameters.source
        distribution = getattr(source, "f_st_paint_distribution", "lognormal").lower()
        sigma = float(getattr(source, "f_st_paint_sigma", 0.5))
        f_st_center = float(source.f_st)
        f_st_min = float(getattr(source, "f_st_paint_min", getattr(source, "f_st_grid_min", 0.01)))
        f_st_max = float(getattr(source, "f_st_paint_max", getattr(source, "f_st_grid_max", 0.2)))

        if f_st_center <= 0:
            raise ValueError("parameters.source.f_st must be > 0 for stochastic f_st painting")
        if f_st_min <= 0 or f_st_max <= 0:
            raise ValueError("f_st painting bounds must be > 0")
        if f_st_max > 1.0:
            raise ValueError(f"f_st_paint_max={f_st_max} exceeds 1.0, which is unphysical for a stellar fraction")
        if f_st_min >= f_st_max:
            raise ValueError("f_st_paint_min must be smaller than f_st_paint_max")

        if distribution == "lognormal":
            sampled = rng.lognormal(mean=np.log(f_st_center) - sigma**2/2, sigma=sigma, size=halo_count)
        elif distribution == "normal":
            sampled = rng.normal(loc=f_st_center, scale=sigma, size=halo_count)
        elif distribution == "uniform":
            sampled = rng.uniform(low=f_st_min, high=f_st_max, size=halo_count)
        else:
            raise ValueError(
                f"Unknown f_st painting distribution {distribution!r}. "
                "Use 'lognormal', 'normal', or 'uniform'."
            )

        return np.clip(sampled, f_st_min, f_st_max)
    
    def _nearest_f_st_indices(self, sampled_f_st: np.ndarray, f_st_grid: np.ndarray) -> np.ndarray:
        """Map sampled f_st values to the nearest precomputed f_st-grid index."""
        f_st_grid = self._small_profile_array(f_st_grid)
        return np.abs(sampled_f_st[:, None] - f_st_grid[None, :]).argmin(axis=1)

    def _make_f_st_rng(self, z_index: int):
        seed = getattr(self.parameters.source, "f_st_paint_seed", None)
        if seed is None:
            return np.random.default_rng()
        return np.random.default_rng(int(seed) + int(z_index))


    # def _sample_f_st_for_halos(self, halo_count: int, rng: np.random.Generator) -> np.ndarray:
    #     source = self.parameters.source
    #     distribution = getattr(source, "f_st_paint_distribution", "lognormal").lower()
    #     sigma = float(getattr(source, "f_st_paint_sigma", 0.5))
    #     f_st_center = float(source.f_st)
    #     f_st_min = float(getattr(source, "f_st_paint_min", getattr(source, "f_st_grid_min", 0.01)))
    #     f_st_max = float(getattr(source, "f_st_paint_max", getattr(source, "f_st_grid_max", 0.2)))

    #     if distribution == "lognormal":
    #         sampled = rng.lognormal(mean=np.log(f_st_center), sigma=sigma, size=halo_count)
    #     elif distribution == "normal":
    #         sampled = rng.normal(loc=f_st_center, scale=sigma, size=halo_count)
    #     elif distribution == "uniform":
    #         sampled = rng.uniform(low=f_st_min, high=f_st_max, size=halo_count)
    #     else:
    #         raise ValueError(f"Unknown f_st painting distribution {distribution!r}")

    #     return np.clip(sampled, f_st_min, f_st_max)
    
    def paint_single_mass_bin(
        self,
        halo_catalog: HaloCatalog,
        z: float,
        radial_grid: np.ndarray,
        r_lyal: np.ndarray,
        profiles_of_bin: tuple[np.ndarray, np.ndarray, np.ndarray],
        store_grids: list | tuple = ('delta_b', 'Grid_Temp', 'Grid_xHII', 'Grid_xal'),
    ):
        """Paint one mass/alpha bin and return **Fourier-space** contributions.

        By returning Fourier-space arrays instead of real-space grids, the
        coordinator can accumulate contributions from all bins in Fourier space
        and perform exactly **3 inverse FFTs** per snapshot — regardless of how
        many mass bins there are.  For 200 bins this reduces 600 inverse FFTs to
        3, an O(N_bins) algorithmic saving.

        GPU backends (jax, torch) keep all data on-device throughout; the
        coordinator does a single device→CPU transfer at the final IFFT step.

        Args:
            halo_catalog (HaloCatalog): Subset of halos for this bin.
            z (float): Snapshot redshift.
            radial_grid (np.ndarray): Radial coordinate grid for profiles.
            r_lyal (np.ndarray): Radial grid for Lyman-alpha profiles.
            profiles_of_bin (tuple): ``(R_bubble, rho_alpha, Temp_profile)``.
            store_grids (sequence): Field names to compute (others yield None).

        Returns:
            tuple: ``(fa_xHII, fa_lyal, fa_temp)`` — Fourier-space contributions
            on the appropriate device, or None for skipped fields.
        """
        cores = self.parameters.simulation.cores
        fft_backend = _resolve_fft_backend(self.parameters.simulation.fft_backend)
        # GPU handles internal parallelism; don't over-subscribe CPU threads in workers.
        if fft_backend in _GPU_BACKENDS:
            fft_workers = 1
        elif cores <= 1:
            fft_workers = -1
        else:
            fft_workers = max(1, (os.cpu_count() or 1) // cores)

        nGrid = self.parameters.simulation.Ncell
        LBox = self.parameters.simulation.Lbox
        truncate = False
        R_bubble, rho_alpha_, Temp_profile = profiles_of_bin

        halo_grid = halo_catalog.to_mesh()
        fa_halo = precompute_fft(halo_grid, backend=fft_backend, workers=fft_workers)

        fa_xHII = None
        if "Grid_xHII" in store_grids:
            x_HII_profile = np.zeros(len(radial_grid))
            x_HII_profile[radial_grid < R_bubble / (1 + z)] = 1
            profile_fn = interp1d(radial_grid * (1 + z), x_HII_profile, bounds_error=False, fill_value=(1, 0))
            kernel = profile_to_3Dkernel(profile_fn, nGrid, LBox)
            if np.any(kernel > 0):
                renorm = (
                    _trapz(x_HII_profile * 4 * np.pi * radial_grid ** 2, radial_grid)
                    / (LBox / (1 + z)) ** 3 / np.mean(kernel)
                )
                fa_xHII = fourier_multiply_kernel(fa_halo, kernel, backend=fft_backend, workers=fft_workers) * renorm
            else:
                # Bubble smaller than a cell — represent direct halo weighting in Fourier space.
                scale = (
                    _trapz(x_HII_profile * 4 * np.pi * radial_grid ** 2, radial_grid)
                    / (LBox / nGrid / (1 + z)) ** 3
                )
                fa_xHII = fa_halo * scale

        fa_lyal = None
        if "Grid_xal" in store_grids:
            x_alpha_prof = 1.81e11 * rho_alpha_ / (1 + z)
            if isinstance(truncate, float):
                x_alpha_prof[r_lyal * (1 + z) < truncate] = x_alpha_prof[r_lyal * (1 + z) < truncate][-1]
            kernel = stacked_lyal_kernel(
                r_lyal * (1 + z), x_alpha_prof, LBox, nGrid,
                nGrid_min=self.parameters.simulation.minimum_grid_size_lyal,
            )
            if np.any(kernel > 0):
                renorm = (
                    _trapz(x_alpha_prof * 4 * np.pi * r_lyal ** 2, r_lyal)
                    / (LBox / (1 + z)) ** 3 / np.mean(kernel)
                )
                fa_lyal = fourier_multiply_kernel(fa_halo, kernel, backend=fft_backend, workers=fft_workers) * renorm

        fa_temp = None
        if "Grid_Temp" in store_grids:
            if isinstance(truncate, float):
                Temp_profile[radial_grid * (1 + z) < truncate] = Temp_profile[radial_grid * (1 + z) < truncate][-1]
            kernel = stacked_T_kernel(
                radial_grid * (1 + z), Temp_profile, LBox, nGrid,
                nGrid_min=self.parameters.simulation.minimum_grid_size_heat,
            )
            if np.any(kernel > 0):
                renorm = (
                    _trapz(Temp_profile * 4 * np.pi * radial_grid ** 2, radial_grid)
                    / (LBox / (1 + z)) ** 3 / np.mean(kernel)
                )
                fa_temp = fourier_multiply_kernel(fa_halo, kernel, backend=fft_backend, workers=fft_workers) * renorm

        return fa_xHII, fa_lyal, fa_temp
