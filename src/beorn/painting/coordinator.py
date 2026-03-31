"""Painting coordinator: orchestrate conversion of 1D profiles into 3D maps."""
import time
from datetime import timedelta
import logging
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, wait, as_completed
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
try:
    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor
    MPI_ENABLED = True
except RuntimeError:
    # mpi fails to import because the host system does not have it installed
    MPI_ENABLED = False

from .helpers import TQDM_KWARGS, precompute_fft
from .painters import paint_alpha_profile, paint_ionization_profile, paint_temperature_profile
from .spread  import spreading_excess_fast
from ..cosmo import T_adiab_fluctu
from ..couplings import S_alpha
from ..io.handler import Handler
from ..structs.radiation_profiles import RadiationProfiles
from ..structs.parameters import Parameters
from ..structs.coeval_cube import CoevalCube
from ..structs.temporal_cube import TemporalCube
from ..structs.halo_catalog import HaloCatalog
from ..load_input_data.base import BaseLoader


class PaintingCoordinator:
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
                        cube.append(grid_data, loop_index)

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
                # Use the profile-matched redshift for the filename check: paint_single
                # writes CoevalCube with z=zgrid (nearest profile z), so we must look
                # for that name rather than the raw loader redshift which may differ.
                profile_z_index = int(np.argmin(np.abs(radiation_profiles.z_history - z)))
                zgrid = float(radiation_profiles.z_history[profile_z_index])
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
            profiles = RadiationProfiles.read(profiles_path)



        iteration_start_time = time.time()
        zero_grid = np.zeros((self.parameters.simulation.Ncell, self.parameters.simulation.Ncell, self.parameters.simulation.Ncell))

        halo_catalog = self.loader.load_halo_catalog(z_index)
        delta_b = self.loader.load_density_field(z_index)

        # Find the profile index whose redshift is nearest to this snapshot's redshift.
        # When solver.redshifts and simulation.snapshot_redshifts are the same grid this is a
        # direct lookup; when snapshot_redshifts is a coarser subset the nearest
        # profile step is used.
        snap_z = self.loader.redshifts[z_index]
        profile_z_index = int(np.argmin(np.abs(profiles.z_history - snap_z)))
        if abs(profiles.z_history[profile_z_index] - snap_z) > 0.5:
            self.logger.warning(
                f"Snapshot z={snap_z:.3f} is more than Δz=0.5 away from the nearest "
                f"profile redshift z={profiles.z_history[profile_z_index]:.3f}. "
                "Consider adding more steps to solver.redshifts."
            )

        zgrid = profiles.z_history[profile_z_index]
        mass_range = profiles.halo_mass_bins[..., profile_z_index]

        # log some information about the current "paintable range"
        alphas = self.parameters.solver.halo_mass_accretion_alpha
        self.logger.debug(
            f"Got {mass_range.shape[0]}x{mass_range.shape[1]} profiles. Range: "
            f"alpha={alphas[0]:.2f} [{mass_range[...,0].min():.2e} - {mass_range[..., 0].max():.2e} Msun] and "
            f"alpha={alphas[-1]:.2f} [{mass_range[...,-1].min():.2e} - {mass_range[..., -1].max():.2e} Msun]."
        )

        # # TODO - describe the relevance of coef
        # coef = constants.rhoc0 * self.parameters.cosmology.h ** 2 * self.parameters.cosmology.Ob * (1 + zgrid) ** 3 * constants.M_sun / constants.cm_per_Mpc ** 3 / constants.m_H


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

        if halo_catalog.masses.max() > mass_range.max() or halo_catalog.masses.min() < mass_range.min():
            raise RuntimeError(f"The current halo catalog at z={zgrid} has a higher masse range ({halo_catalog.masses.max():.2e} - {halo_catalog.masses.min():.2e}) than the mass range of the precomputed profiles ({mass_range.max():.2e} - {mass_range.min():.2e}). You need to adjust your parameters: either increase the mass range of the profile simulation (parameters.simulation) or decrease the mass range of star forming halos (parameters.source).")

        self.logger.info(f'Painting {halo_catalog.size} halos at {zgrid=:.2f} ({z_index=:.0f}).')

        # initialise the "main" grids here. Since they will be filled in place by multiple parallel processes, we need to use shared memory
        # get the memory size of the grids

        size = zero_grid.size * np.dtype(np.float64).itemsize
        if "Grid_xHII" in self.parameters.simulation.store_grids:
            buffer_xHII = shared_memory.SharedMemory(create=True, size=size)
        else:
            buffer_xHII = None

        if "Grid_Temp" in self.parameters.simulation.store_grids:
            buffer_Temp = shared_memory.SharedMemory(create=True, size=size)
        else:
            buffer_Temp = None

        if "Grid_xal" in self.parameters.simulation.store_grids:
            buffer_xal = shared_memory.SharedMemory(create=True, size=size)
        else:
            buffer_xal = None

        with ProcessPoolExecutor(max_workers=self.parameters.simulation.cores) as executor:
            # if only one process is used, we won't make use of the executor
            futures = []
            total_halos = 0

            ## iterate over the range of mass and alpha bins that the profiles are available for
            # the alpha bins are constant so we can use the ones from the parameters
            # the mass bins are more tricky - they follow the mass accretion history i.e. they shift with each redshift step
            alpha_indices = range(len(self.parameters.solver.halo_mass_accretion_alpha) - 1)
            mass_indices = range(len(self.parameters.solver.halo_mass_bins) - 1)

            # now each profile was computed for a precise mass/alpha value that we set to be the center points of the bins
            # => in the actual profile the shape is (l-1)x(m-1)x(n-1) where l,m,n are the number of bins in mass, alpha and redshift
            # For each profile we have a range of mass and alpha values where we can pick haloes from
            # We just need to ensure that all haloes are considered in the end (hence the total_halos check)

            self.logger.debug(f"Using {self.parameters.simulation.cores} processes for painting.")
            start_time = time.time()

            for alpha_index in alpha_indices:
                # the alpha range is simply defined by the parameters
                loop_alpha_range = [
                    self.parameters.solver.halo_mass_accretion_alpha[alpha_index],
                    self.parameters.solver.halo_mass_accretion_alpha[alpha_index + 1]
                ]
                for mass_index in mass_indices:

                    # the mass range shifts with the redshift so we need to take the mass range for the current redshift and take the bins from there
                    loop_mass_range = [
                        mass_range[mass_index, alpha_index],
                        mass_range[mass_index + 1, alpha_index]
                    ]
                    halo_indices = halo_catalog.get_halo_indices(loop_alpha_range, loop_mass_range)
                    # shortcut: don't copy any memory if there are no halos to begin with
                    if halo_indices.size == 0:
                        continue
                    total_halos += halo_indices.size

                    # since the profiles are large and copied in the multiprocessing approach, we only pass the relevant slice
                    profiles_of_bin = profiles.profiles_of_halo_bin(profile_z_index, alpha_index, mass_index)
                    assert not np.any(np.isnan(profiles_of_bin[0])), "R_bubble at the current range seem to be malformed (got nan values)"
                    assert not np.any(np.isnan(profiles_of_bin[1])), "rho_alpha at the current range seem to be malformed (got nan values)"
                    assert not np.any(np.isnan(profiles_of_bin[2])), "rho_heat at the current range seem to be malformed (got nan values)"

                    radial_grid = profiles.r_grid_cell[:] / (1 + zgrid)  # pMpc/h
                    kwargs = {
                        "halo_catalog": halo_catalog.at_indices(halo_indices),
                        "z": zgrid,
                        # profiles related quantities
                        "radial_grid": radial_grid,
                        "r_lyal": profiles.r_lyal[:],
                        "profiles_of_bin": profiles_of_bin,
                        # shared memory buffers
                        "buffer_lyal": buffer_xal,
                        "buffer_temp": buffer_Temp,
                        "buffer_xHII": buffer_xHII
                    }

                    if self.parameters.simulation.cores > 1:
                        # use the multiprocessing approach and submit the task to the executor
                        f = executor.submit(
                            self.paint_single_mass_bin,
                            **kwargs
                        )
                        futures.append(f)
                    else:
                        # use the single process approach and call the function directly
                        self.paint_single_mass_bin(**kwargs)

            # wait for all futures to complete
            completed, uncompleted = wait(futures)
            assert len(uncompleted) == 0, "Not all painting subprocesses completed successfully"
            assert total_halos == halo_catalog.size, f"Number of painted halos ({total_halos}) does not match the halo catalog size ({halo_catalog.size})."

        # clean up the shared memory buffers - but keep the data that was in the buffers
        if buffer_xHII:
            array = np.ndarray(zero_grid.shape, dtype=np.float64, buffer=buffer_xHII.buf)
            Grid_xHII = array.copy()
            buffer_xHII.close()
            buffer_xHII.unlink()
        else:
            Grid_xHII = zero_grid
        if buffer_Temp:
            array = np.ndarray(zero_grid.shape, dtype=np.float64, buffer=buffer_Temp.buf)
            Grid_Temp = array.copy()
            buffer_Temp.close()
            buffer_Temp.unlink()
        else:
            Grid_Temp = zero_grid
        if buffer_xal:
            array = np.ndarray(zero_grid.shape, dtype=np.float64, buffer=buffer_xal.buf)
            Grid_xal = array.copy()
            buffer_xal.close()
            buffer_xal.unlink()
        else:
            Grid_xal = zero_grid

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

        return grid_data


    def paint_single_mass_bin(
        self,
        halo_catalog: HaloCatalog,
        # profile related quantities - we don't want to pass the whole radiation_profiles object
        z: float,
        radial_grid: np.ndarray,
        r_lyal: np.ndarray,
        profiles_of_bin: tuple[np.ndarray, np.ndarray, np.ndarray],
        buffer_lyal: np.ndarray = None,
        buffer_temp: np.ndarray = None,
        buffer_xHII: np.ndarray = None,
    ):
        """Paint all halos in a single mass/alpha bin into shared buffers.

        This lower-level method is invoked either directly (single-core)
        or in worker processes. It computes per-halo contributions for
        ionization, Lyman-alpha and temperature and writes them into the
        provided shared-memory buffers.

        Args:
            halo_catalog (HaloCatalog): Subset of halos to paint.
            z (float): Snapshot redshift.
            radial_grid (np.ndarray): Radial coordinate grid for profiles.
            r_lyal (np.ndarray): Radial grid for Lyman-alpha profiles.
            profiles_of_bin (tuple): Tuple ``(R_bubble, rho_alpha, Temp_profile)``.
            buffer_lyal, buffer_temp, buffer_xHII: Optional shared-memory buffers.

        Returns:
            None: Shared buffers are modified in-place.
        """
        nGrid = self.parameters.simulation.Ncell
        output_shape = (nGrid, nGrid, nGrid)
        LBox = self.parameters.simulation.Lbox
        # TODO
        # truncate = self.parameters.simulation.truncate_radius
        truncate = False

        R_bubble, rho_alpha_, Temp_profile = profiles_of_bin

        # place the halos on the grid so that they can be used in a convolution
        halo_grid = halo_catalog.to_mesh()
        # precompute the FFT of the halo grid once; all three paint functions reuse it
        fft_halo_grid = precompute_fft(halo_grid)

        # Every halo in the mass bin i is assumed to have the mass M_bin[i].
        if buffer_xHII:
            # initialize the output grid over the shared memory buffer
            output_grid_xHII = np.ndarray(output_shape, dtype=np.float64, buffer=buffer_xHII.buf)
            x_HII_profile = np.zeros((len(radial_grid)))
            x_HII_profile[np.where(radial_grid < R_bubble / (1 + z))] = 1

            # modify Grid_xHII in place
            paint_ionization_profile(
                output_grid_xHII, radial_grid, x_HII_profile, nGrid, LBox, z, halo_grid,
                fft_halo_grid=fft_halo_grid,
            )

        if buffer_lyal:
            # initialize the output grid over the shared memory buffer
            output_grid_lyal = np.ndarray(output_shape, dtype=np.float64, buffer=buffer_lyal.buf)
            x_alpha_prof = 1.81e11 * (rho_alpha_) / (1 + z)
            # We add up S_alpha(z, T_extrap, 1 - xHII_extrap) later, a the map level.

            # TODO - document how r_lyal is the physical distance for lyal profile. Never goes further away than 100 pMpc/h (checked)
            # modify Grid_xal in place
            paint_alpha_profile(
                output_grid_lyal, r_lyal, x_alpha_prof, nGrid, LBox, self.parameters.simulation.minimum_grid_size_lyal, z, truncate, halo_grid,
                fft_halo_grid=fft_halo_grid,
            )

        if buffer_temp:
            # initialize the output grid over the shared memory buffer
            output_grid_temp = np.ndarray(output_shape, dtype=np.float64, buffer=buffer_temp.buf)
            # modify Grid_Temp in place
            paint_temperature_profile(
                output_grid_temp, radial_grid, Temp_profile, nGrid, LBox, self.parameters.simulation.minimum_grid_size_heat, z, truncate, halo_grid,
                fft_halo_grid=fft_halo_grid,
            )
