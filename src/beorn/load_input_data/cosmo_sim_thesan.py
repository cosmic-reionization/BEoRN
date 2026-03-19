import h5py
import numpy as np
from typing import Literal
from .base import BaseLoader
from .lookback import get_lookback_range
from ..structs import HaloCatalog
from ..particle_mapping import pylians
from .alpha_fitting import vectorized_alpha_fit

# following: https://thesan-project.com/thesan/thesan.html

INVALID_FALLBACK: float | Literal["mean", "median"] = 0.79

class ThesanLoader(BaseLoader):
    """
    Loader for the THESAN data format.
    """

    def __init__(self, *args, **kwargs):
        is_high_res = kwargs.pop("is_high_res", False)

        super().__init__(*args, **kwargs)
        self.thesan_root = self.parameters.simulation.file_root
        self.tree_root = self.thesan_root / "postprocessing"/ "trees"/ "LHaloTree"
        self.snapshot_path_root = self.thesan_root / "output"
        self.offset_path_root = self.thesan_root / "postprocessing" / "offsets"
        self.cached_tree = self.tree_root / "tree_cache.hdf5"

        if is_high_res:
            # this is thesan dark 1
            tree_name = "trees_sf1_190.10.hdf5"
            self.particle_count = 2100**3
        else:
            # this is thesan dark 2
            tree_name = "trees_sf1_167.10.hdf5"
            self.particle_count = 1050**3


        self.logger.info(f"Initialized THESAN data loader - reading files from {self.tree_root}, {self.snapshot_path_root}, {self.offset_path_root}")

        # get the available redshifts
        with h5py.File(self.tree_root / tree_name, "r") as f:
            header = f["Header"]
            redshifts = header["Redshifts"][:]
            self.logger.debug(f"Redshifts: {redshifts.size}")


        catalogs = list(self.snapshot_path_root.glob("groups_*"))
        density_directories = list(self.snapshot_path_root.glob("snapdir_*"))
        offset_files = list(self.offset_path_root.glob("offsets_*"))
        # since they are named with their index, we can sort them
        catalogs.sort()
        density_directories.sort()
        offset_files.sort()

        # now restrict to the redshift range
        indices = np.where((redshifts >= self.parameters.solver.redshifts[0]) & (redshifts <= self.parameters.solver.redshifts[-1]))[0]

        self._redshifts = redshifts[indices]
        self.catalogs = [catalogs[i] for i in indices]
        self.density_directories = [density_directories[i] for i in indices]
        self.offset_files = [offset_files[i] for i in indices]

        self.logger.info(f"THESAN redshifts: {self.redshifts.size} ({self.redshifts[0]} - {self.redshifts[-1]})")

        first_snapshot = next(self.density_directories[0].glob("snap_*.hdf5"))

        with h5py.File(first_snapshot, "r") as f:
            self.thesan_h = f["Header"].attrs["HubbleParam"]
            self.logger.debug(f"Using h={self.thesan_h} from the first snapshot header")


    @property
    def redshifts(self):
        return self._redshifts


    def load_tree_cache(self):
        # generation of the cache file is done in the extract_simplified_tree.ipynb notbebook
        with h5py.File(self.cached_tree, "r") as f:
            tree_halo_ids = f["tree_halo_ids"][:]
            tree_snap_num = f["tree_snap_num"][:]
            tree_mass = f["tree_mass"][:]
            tree_main_progenitor = f["tree_main_progenitor"][:]
        self.logger.debug(f"Loaded pre-cached THESAN merger tree from: {self.cached_tree}")
        return tree_halo_ids, tree_snap_num, tree_mass, tree_main_progenitor


    def get_halo_accretion_rate_from_tree(self, current_index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the (incomplete) accretion rate by performing a fit to the mass history of the halos in the merger tree.
        Since some are not represented in the tree, this function returns the indices and alpha values for the halos that are present in the tree.
        """
        redshift_range = get_lookback_range(self.parameters, redshifts = self.redshifts[:current_index + 1])
        self.logger.debug(f"Merger tree lookback range is [{redshift_range[0]:.2f} - {redshift_range[-1]:.2f}] ({redshift_range.size} snapshots)")

        tree_halo_ids, tree_snap_num, tree_mass, tree_main_progenitor = self.load_tree_cache()

        current_halo_indices = (tree_snap_num == current_index) & (tree_mass > 0)
        current_halo_ids = tree_halo_ids[current_halo_indices]
        current_halo_count = current_halo_indices.sum()

        # if the redshift range is shorter than the range requested in the parameters, this means that the snapshot is early and has too few predecessors
        if redshift_range.size < self.parameters.source.mass_accretion_lookback:
            # in this case a fit is likely to be very unstable, so we just return a constant baseline value
            # since at that point halo masses are low, this is not too problematic
            self.logger.warning(f"Available redshifts for lookback ({redshift_range.size}) are too few compared to the requested ({self.parameters.source.mass_accretion_lookback}), returning constant fallback value for mass accretion rate")
            return current_halo_ids, np.full(current_halo_count, 0.04)
            # TODO - should the value of this be a parameter?

        halo_mass_history = np.ndarray((current_halo_count, redshift_range.size))

        for i in range(redshift_range.size):
            # find the progenitors of the current halos
            progenitor_indices = tree_main_progenitor[current_halo_indices]

            # find the mass of these progenitors
            halo_mass_history[:, i] = tree_mass[progenitor_indices]
            # halo_mass_history[progenitor_indices < 0, i] = np.nan  # set negative indices to NaN

            # now we can find the next progenitors
            current_halo_indices = progenitor_indices

        # at this point halo_mass_history has the same shape as redshift_range, and the sorting is current_redshift -> past redshifts
        # remove invalid values, but don't set them to 0 because the fitting is in log space
        halo_mass_history[halo_mass_history <= 0] = 1
        self.logger.debug(f"Found {np.sum(np.isnan(halo_mass_history))} haloes with nan masses")
        self.logger.debug(f"Found {np.sum(np.any(halo_mass_history == 1, axis=1))} trees that stopped early (invalid or missing mass)")

        halo_alphas = vectorized_alpha_fit(redshift_range, halo_mass_history)
        self.logger.debug(f"Fitting gave {np.sum(np.isnan(halo_alphas))} NaN values and {np.sum(np.isinf(halo_alphas))} inf values.")

        return current_halo_ids, halo_alphas


    def get_halo_information_from_catalog(self, redshift_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        offset_file = self.offset_files[redshift_index]

        path = self.catalogs[redshift_index]
        current_snapshots = path.rglob("*.hdf5")
        current_snapshots = list(current_snapshots)

        # sort the list according to the number in the filename: 1, 2, 3, ..., 10, 11, ...
        current_snapshots.sort(key=lambda x: int(x.stem.split(".")[1]))

        with h5py.File(offset_file, "r") as f:
            group_offset_indices = f["FileOffsets"]["Group"][:]
            subhalo_start_index = f["FileOffsets"]["Subhalo"][:]

        # the THESAN documentation says that each chunk files shows how many halos/groups the whole snapshot will yield:
        # snapshot_halo_count = f["Header"]["Ngroups_Total"]
        # in practice this key does not exist, so we need the offset file to estimate the number of halos
        snapshot_group_count = int(group_offset_indices[-1]  * 1.5)
        snapshot_subhalo_count = int(subhalo_start_index[-1] * 1.5)

        current_halo_positions = np.zeros((snapshot_group_count, 3))
        current_halo_masses = np.zeros((snapshot_group_count))
        current_subhalo_to_group_mappings = np.zeros((snapshot_group_count), dtype=int)

        group_start_index = 0
        subhalo_start_index = 0
        for i, f in enumerate(current_snapshots):
            with h5py.File(f, "r") as snap:
                # logger.debug(f"Reading group catalog {f.name}...")
                if "GroupPos" not in snap["Group"]:
                    # the file is empty because all halos were already loaded
                    continue
                group_positions = snap["Group"]["GroupPos"][:]
                group_end_index = group_start_index + group_positions.shape[0]

                current_halo_positions[group_start_index:group_end_index, :] = group_positions
                # NB: other keys related to mass exist - make sure you understand the requirements of your simulation
                # Group_M_Crit200, Group_M_Crit500
                current_halo_masses[group_start_index:group_end_index] = snap["Group"]["GroupMass"][:]

                group_start_index = group_end_index

                subhalo_mappings = snap["Subhalo"]["SubhaloGrNr"][:]
                subhalo_end_index = subhalo_start_index + subhalo_mappings.shape[0]
                current_subhalo_to_group_mappings[subhalo_start_index:subhalo_end_index] = subhalo_mappings # no + offset needed apparently
                subhalo_start_index = subhalo_end_index

                # the ids are just the indices in the file plus the offset

        # just keep the ones that were really loaded
        current_halo_positions = current_halo_positions[:group_start_index, :]
        current_halo_masses = current_halo_masses[:group_start_index]
        current_subhalo_to_group_mappings = current_subhalo_to_group_mappings[:subhalo_start_index]

        return current_halo_positions, current_halo_masses, current_subhalo_to_group_mappings


    def load_halo_catalog(self, redshift_index: int) -> HaloCatalog:
        # using the tree fitting only gives an estimate for halos that actually appear in a tree
        current_halo_ids, halo_alphas = self.get_halo_accretion_rate_from_tree(redshift_index)
        alpha_fallback = self.fallback_alpha(halo_alphas)

        # the information for the current snapshot is not from the tree but from the group catalog
        current_halo_positions, current_halo_masses, current_subhalo_to_group_mappings = self.get_halo_information_from_catalog(redshift_index)
        snapshot_group_count = current_halo_masses.size

        ## join both informations
        # assuming that the indices from the group catalog are strictly monotonic we can force the same sorting on the halo ids obtained from the tree
        sorting = np.argsort(current_halo_ids)
        tree_ids = current_halo_ids[sorting]
        sorted_halo_alphas = halo_alphas[sorting]

        # fill the baseline value and replace it afterwards for the entries where we have a value
        full_alphas = np.full(snapshot_group_count, alpha_fallback)

        # since we use the main progenitor branch, each id from there corresponds to a subhalo of the actual group - additional mapping is needed
        group_ids = current_subhalo_to_group_mappings[tree_ids]
        # fill in the alphas for the halos that are in the tree
        full_alphas[group_ids] = sorted_halo_alphas

        # the alpha fitting returned np.inf for halos that have "too short" mass histories and nan for halos that have no history at all
        # for now we just set them to the baseline value as well, but the treatment could be refined
        # or in working form:
        full_alphas[~np.isfinite(full_alphas)] = alpha_fallback

        # full_alphas[:] = full_alphas.mean()  # TEMPORARY OVERRIDE FOR TESTING PURPOSES
        # self.logger.info(f"Overriding all alphas to the mean value of {full_alphas.mean():.2f} for testing purposes")

        # finally - clip the alphas to the range that is allowed by the parameters
        # - negative accretion has no meaning in beorn
        # - all values should lie in the "paintable" range, i.e. not be too high
        # note that the upper limit is above the paintable range, so we use the -2 index to make sure that the halos are "paintable"

        alpha_range = self.parameters.simulation.halo_mass_accretion_alpha
        below = full_alphas < alpha_range[0]
        above = full_alphas > alpha_range[-2]
        full_alphas[below] = alpha_range[0]
        full_alphas[above] = alpha_range[-2]
        self.logger.debug(f"Corrected {np.sum(below)} alphas below {alpha_range[0]:.2f} and {np.sum(above)} alphas above {alpha_range[-2]:.2f}")

        assert full_alphas.shape == current_halo_masses.shape, "The alphas and masses must have the same shape"

        catalog = HaloCatalog(
            positions = current_halo_positions * 1e-3 / self.thesan_h, # convert from kpc/h to Mpc/h to Mpc
            masses = current_halo_masses * 1e10 / self.thesan_h, # convert to Msun/h to Msun
            alphas = full_alphas,
            parameters = self.parameters
        )

        self.logger.debug(f"Catalog alphas: min={catalog.alphas.min(initial=0):.2f}, max={catalog.alphas.max(initial=0):.2f}, mean={catalog.alphas.mean():.2f}, std={catalog.alphas.std():.2f}")
        return catalog


    def load_density_field(self, redshift_index: int) -> np.ndarray:

        # the current redshift index only gives a folder containing the Thesan snapshot files
        snapshot_path = self.density_directories[redshift_index]
        # the files have a particular format, cf. https://thesan-project.com/thesan/snapshots.html
        snapshots = snapshot_path.glob("snap_*.hdf5")

        particle_positions = np.zeros((self.particle_count, 3), dtype=np.float32)

        # load all particles which are spread across multiple file chunks
        start_index = 0
        for snapshot in snapshots:
            # logger.debug(f"Reading snapshot {snapshot.name}...")
            with h5py.File(snapshot, "r") as f:
                positions = f["PartType1"]["Coordinates"][:]
                end_index = start_index + positions.shape[0]
                particle_positions[start_index:end_index, :] = positions
                start_index = end_index

        # map them to a mesh that is LBox x LBox x LBox
        # Create a density mesh from the particle positions
        mesh_size = self.parameters.simulation.Ncell
        mesh = np.zeros((mesh_size, mesh_size, mesh_size), dtype=np.float32)

        # convert the coordinates to Mpc/h
        particle_positions *= 1e-3 / self.thesan_h
        # logger.debug(f"Particle information, ended at {start_index=} => {np.sum(particle_positions == 0)} empty fields, {particle_positions[:, 0].min():.2f} - {particle_positions[:, 0].max():.2f} in the first dimension, {particle_positions[:, 1].min():.2f} - {particle_positions[:, 1].max():.2f} in the second dimension, {particle_positions[:, 2].min():.2f} - {particle_positions[:, 2].max():.2f} in the third dimension")
        physical_size = particle_positions.max()

        mass_assignment = self.parameters.simulation.halo_catalogs_thesan_mass_assignment
        pylians.map_particles_to_mesh(mesh, physical_size, particle_positions, mass_assignment=mass_assignment)

        # Normalize the mesh to get the density field
        delta_b = mesh / np.mean(mesh, dtype=np.float64) - 1
        return delta_b


    def load_rsd_fields(self, redshift_index: int):
        # the current redshift index only gives a folder containing the Thesan snapshot files
        snapshot_path = self.density_directories[redshift_index]
        # the files have a particular format, cf. https://thesan-project.com/thesan/snapshots.html
        snapshots = snapshot_path.glob("snap_*.hdf5")

        particle_velocities = np.zeros((self.particle_count, 3), dtype=np.float32)
        particle_positions = np.zeros((self.particle_count, 3), dtype=np.float32)

        # load all particles which are spread across multiple file chunks
        start_index = 0
        for snapshot in snapshots:
            # logger.debug(f"Reading snapshot {snapshot.name}...")
            with h5py.File(snapshot, "r") as f:
                positions = f["PartType1"]["Coordinates"][:]
                velocities = f["PartType1"]["Velocities"][:]
                end_index = start_index + velocities.shape[0]
                particle_velocities[start_index:end_index, :] = velocities
                particle_positions[start_index:end_index, :] = positions

                start_index = end_index

        mesh_size = self.parameters.simulation.Ncell
        mesh_x = np.zeros((mesh_size, mesh_size, mesh_size), dtype=np.float32)
        mesh_y = mesh_x.copy()
        mesh_z = mesh_x.copy()

        # convert the coordinates to Mpc/h and the velocities to km/s
        particle_positions *= 1e-3 / self.thesan_h
        scale_factor = 1 / (1 + self.redshifts[redshift_index])
        particle_velocities *= 1 / np.sqrt(scale_factor)

        mass_assignment = self.parameters.simulation.halo_catalogs_thesan_mass_assignment
        pylians.map_particles_to_mesh(mesh_x, self.parameters.simulation.Lbox, particle_positions, mass_assignment=mass_assignment, weights = particle_velocities[:, 0])
        pylians.map_particles_to_mesh(mesh_y, self.parameters.simulation.Lbox, particle_positions, mass_assignment=mass_assignment, weights = particle_velocities[:, 1])
        pylians.map_particles_to_mesh(mesh_z, self.parameters.simulation.Lbox, particle_positions, mass_assignment=mass_assignment, weights = particle_velocities[:, 2])

        return mesh_x, mesh_y, mesh_z


    def fallback_alpha(self, halo_alpha: np.ndarray) -> float:
        if isinstance(INVALID_FALLBACK, float):
            return INVALID_FALLBACK
        elif INVALID_FALLBACK == "mean":
            return np.mean(halo_alpha)
        elif INVALID_FALLBACK == "median":
            return np.median(halo_alpha)
        else:
            raise ValueError(f"Invalid fallback method: {INVALID_FALLBACK}")
