"""Loader for THESAN dark-matter-only simulation data.

Inherits merger tree walking and alpha fitting from
:class:`~beorn.load_input_data.merger_tree_base.MergerTreeLoader`.
Only THESAN-specific I/O is implemented here:

- Group/FoF catalogs (``groups_*/``), offset files, and LHaloTree-format
  merger trees.
- Dark-matter particle snapshots for the density field.

The merger tree must be preprocessed once into a flat HDF5 cache using the
``extract_simplified_tree.ipynb`` notebook.  The cache schema is documented
in :meth:`MergerTreeLoader.load_tree_cache`.

See https://thesan-project.com for data format documentation.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .merger_tree_base import MergerTreeLoader
from ..particle_mapping import map_particles_to_mesh


class ThesanLoader(MergerTreeLoader):
    """Loader for THESAN-DARK simulation data with LHaloTree merger trees.

    This is the production-ready loader shipped with BEoRN.  If you are
    writing a custom loader for a different simulation see
    :class:`~beorn.load_input_data.merger_tree_base.MergerTreeLoader` for
    the minimal interface you need to implement.

    Args:
        parameters: BEoRN parameter container.  ``cosmo_sim.file_root``
            must point to the THESAN data root (the directory that contains
            ``output/``, ``postprocessing/``, etc.).
        cache_file (Path | str): Path to the flat HDF5 tree cache produced
            by the cache extraction cell in the notebook.  Snapshot redshifts
            and other provenance metadata are read from its root attributes
            and the ``snapshot_redshifts`` dataset.
        is_high_res (bool): ``True`` for THESAN-DARK 1 (2100³ particles),
            ``False`` (default) for THESAN-DARK 2 (1050³ particles).
    """

    simulation_code = "THESAN"

    @staticmethod
    def _snapshot_index_from_name(path: Path) -> int:
        """Extract the integer snapshot index from names like groups_039 or offsets_039.hdf5."""
        suffix = path.stem if path.is_file() else path.name
        return int(suffix.split("_")[1])

    def __init__(self, parameters, cache_file: Path | str, *, is_high_res: bool = False):
        super().__init__(parameters)

        self.thesan_root      = Path(self.parameters.cosmo_sim.file_root)
        self.tree_root        = self.thesan_root / "postprocessing" / "trees" / "LHaloTree"
        self.snapshot_path_root = self.thesan_root / "output"
        self.offset_path_root = self.thesan_root / "postprocessing" / "offsets"
        self.cached_tree      = Path(cache_file)
        self.particle_count   = 2100 ** 3 if is_high_res else 1050 ** 3

        self.logger.info(f"Initialized ThesanLoader — root: {self.thesan_root}")

        if not self.cached_tree.exists():
            raise FileNotFoundError(
                f"Tree cache not found: {self.cached_tree}. "
                "Run the cache extraction cell in the notebook first."
            )

        # Read snapshot redshifts from the HDF5 cache (stored as a dataset
        # alongside the provenance attributes — no separate YAML needed).
        with h5py.File(self.cached_tree, "r") as f:
            redshifts = f["snapshot_redshifts"][:]
        self.logger.info(f"Loaded {redshifts.size} snapshot redshifts from {self.cached_tree}")

        # Index available directories by snapshot number (parsed from the name,
        # e.g. "groups_100" → 100).  Using a dict means only the snapshots you
        # actually downloaded need to be present — a subset is fine.
        catalogs     = {self._snapshot_index_from_name(p): p
                        for p in self.snapshot_path_root.glob("groups_*")}
        density_dirs = {self._snapshot_index_from_name(p): p
                        for p in self.snapshot_path_root.glob("snapdir_*")}
        offset_files = {self._snapshot_index_from_name(p): p
                        for p in self.offset_path_root.glob("offsets_*")}

        # Restrict to the solver redshift range
        z_min = min(self.parameters.solver.redshifts)
        z_max = max(self.parameters.solver.redshifts)
        indices = np.where((redshifts >= z_min) & (redshifts <= z_max))[0]
        available_snapshot_indices = sorted(set(catalogs) & set(density_dirs) & set(offset_files))
        self._snap_indices = np.array(
            [int(i) for i in indices if int(i) in available_snapshot_indices],
            dtype=int,
        )

        if self._snap_indices.size == 0:
            raise FileNotFoundError(
                "No THESAN snapshots within the requested solver.redshifts range have a complete "
                "set of groups_*, snapdir_*, and offsets_* data."
            )

        self._redshifts = redshifts[self._snap_indices]
        self._catalog_directories = [catalogs[i] for i in self._snap_indices]
        self._density_directories = [density_dirs.get(i) for i in self._snap_indices]
        self._offset_files = [offset_files[i] for i in self._snap_indices]

        self.logger.info(
            f"THESAN snapshots: {self._redshifts.size} "
            f"(z={self._redshifts.max():.2f} → {self._redshifts.min():.2f})"
        )

        # Read h from first snapshot header
        first_snap = next(self._density_directories[0].glob("snap_*.hdf5"))
        with h5py.File(first_snap, "r") as f:
            self.thesan_h = f["Header"].attrs["HubbleParam"]

    # ── BaseLoader interface ───────────────────────────────────────────────

    @property
    def redshifts(self) -> np.ndarray:
        return self._redshifts

    def _particle_mapping_config(self) -> tuple[str, str]:
        """Return the particle-to-mesh mapping scheme and backend.

        ``main`` stores the THESAN-specific assignment kernel under
        ``cosmo_sim.halo_catalogs_thesan_mass_assignment``. Older branches used
        ``cosmo_sim.particle_mass_assignment`` instead, so we keep a fallback
        here while preferring the current schema.
        """
        cosmo_sim = self.parameters.cosmo_sim
        mass_assignment = getattr(
            cosmo_sim,
            "halo_catalogs_thesan_mass_assignment",
            getattr(cosmo_sim, "particle_mass_assignment", "CIC"),
        )
        backend = self.parameters.simulation.backend.resolve('mass_assignment')
        return mass_assignment, backend

    # ── MergerTreeLoader interface ─────────────────────────────────────────

    def load_tree_cache(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load the simplified merger tree cache.

        The cache HDF5 file is produced by ``extract_simplified_tree.ipynb``.
        It must contain four datasets: ``tree_halo_ids``, ``tree_snap_num``,
        ``tree_mass``, ``tree_main_progenitor``.
        """
        with h5py.File(self.cached_tree, "r") as f:
            tree_halo_ids       = f["tree_halo_ids"][:]
            tree_snap_num       = f["tree_snap_num"][:]
            tree_mass           = f["tree_mass"][:]
            tree_main_progenitor = f["tree_main_progenitor"][:]
        self.logger.debug(f"Loaded tree cache: {self.cached_tree} ({tree_halo_ids.size:,} entries)")
        return tree_halo_ids, tree_snap_num, tree_mass, tree_main_progenitor

    def get_halo_information_from_catalog(
        self, redshift_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read group positions, masses, and subhalo→group mapping from THESAN group catalogs."""
        offset_file = self._offset_files[redshift_index]
        catalog_dir = self._catalog_directories[redshift_index]

        snap_files = sorted(
            catalog_dir.rglob("*.hdf5"),
            key=lambda p: int(p.stem.split(".")[1]),
        )

        with h5py.File(offset_file, "r") as f:
            group_offsets    = f["FileOffsets"]["Group"][:]
            subhalo_offsets  = f["FileOffsets"]["Subhalo"][:]

        # Pre-allocate with generous upper bounds (THESAN docs lack Ngroups_Total)
        n_groups_approx   = int(group_offsets[-1]  * 1.5)
        n_subhalos_approx = int(subhalo_offsets[-1] * 1.5)

        positions              = np.zeros((n_groups_approx, 3), dtype=np.float32)
        masses                 = np.zeros(n_groups_approx, dtype=np.float64)
        subhalo_to_group_map   = np.zeros(n_subhalos_approx, dtype=np.int64)

        g_ptr, s_ptr = 0, 0
        for snap_file in snap_files:
            with h5py.File(snap_file, "r") as f:
                if "GroupPos" not in f["Group"]:
                    continue
                gpos   = f["Group"]["GroupPos"][:]
                gmass  = f["Group"]["GroupMass"][:]
                g_end  = g_ptr + gpos.shape[0]
                positions[g_ptr:g_end] = gpos
                masses[g_ptr:g_end]    = gmass
                g_ptr = g_end

                smap  = f["Subhalo"]["SubhaloGrNr"][:]
                s_end = s_ptr + smap.shape[0]
                subhalo_to_group_map[s_ptr:s_end] = smap
                s_ptr = s_end

        positions            = positions[:g_ptr]
        masses               = masses[:g_ptr]
        subhalo_to_group_map = subhalo_to_group_map[:s_ptr]

        # Unit conversions: kpc/h → Mpc/h, 10^10 M☉/h → M☉
        positions /= (1e3 * self.thesan_h)   # kpc/h → Mpc
        masses    *= 1e10 / self.thesan_h    # 10^10 M☉/h → M☉

        return positions, masses, subhalo_to_group_map

    # ── Density and velocity fields ────────────────────────────────────────

    def load_density_field(self, redshift_index: int) -> np.ndarray:
        """Paint DM particles onto a mesh and return the overdensity field δ = ρ/⟨ρ⟩ − 1."""
        snapshot_path = self._density_directories[redshift_index]
        snapshots = list(snapshot_path.glob("snap_*.hdf5"))

        particle_positions = np.zeros((self.particle_count, 3), dtype=np.float32)
        ptr = 0
        for snap in snapshots:
            with h5py.File(snap, "r") as f:
                pos = f["PartType1"]["Coordinates"][:]
                particle_positions[ptr: ptr + pos.shape[0]] = pos
                ptr += pos.shape[0]

        mesh_size = self.parameters.simulation.Ncell
        mesh = np.zeros((mesh_size, mesh_size, mesh_size), dtype=np.float32)

        particle_positions *= 1e-3 / self.thesan_h  # kpc/h → Mpc/h
        physical_size = particle_positions.max()

        mass_assignment, backend = self._particle_mapping_config()
        map_particles_to_mesh(
            mesh, physical_size, particle_positions,
            mass_assignment=mass_assignment, backend=backend,
        )
        return mesh / np.mean(mesh, dtype=np.float64) - 1

    def load_rsd_fields(self, redshift_index: int):
        """Return per-axis velocity meshes for redshift-space distortions."""
        snapshot_path = self._density_directories[redshift_index]
        snapshots = list(snapshot_path.glob("snap_*.hdf5"))

        particle_positions  = np.zeros((self.particle_count, 3), dtype=np.float32)
        particle_velocities = np.zeros((self.particle_count, 3), dtype=np.float32)
        ptr = 0
        for snap in snapshots:
            with h5py.File(snap, "r") as f:
                pos = f["PartType1"]["Coordinates"][:]
                vel = f["PartType1"]["Velocities"][:]
                particle_positions [ptr: ptr + pos.shape[0]] = pos
                particle_velocities[ptr: ptr + vel.shape[0]] = vel
                ptr += pos.shape[0]

        mesh_size = self.parameters.simulation.Ncell
        mesh_x = np.zeros((mesh_size, mesh_size, mesh_size), dtype=np.float32)
        mesh_y = mesh_x.copy()
        mesh_z = mesh_x.copy()

        particle_positions *= 1e-3 / self.thesan_h
        scale_factor = 1 / (1 + self.redshifts[redshift_index])
        particle_velocities /= np.sqrt(scale_factor)  # peculiar km/s

        Lbox = self.parameters.Lbox_hunits
        mass_assignment, backend = self._particle_mapping_config()
        map_particles_to_mesh(
            mesh_x, Lbox, particle_positions,
            mass_assignment=mass_assignment, backend=backend, weights=particle_velocities[:, 0],
        )
        map_particles_to_mesh(
            mesh_y, Lbox, particle_positions,
            mass_assignment=mass_assignment, backend=backend, weights=particle_velocities[:, 1],
        )
        map_particles_to_mesh(
            mesh_z, Lbox, particle_positions,
            mass_assignment=mass_assignment, backend=backend, weights=particle_velocities[:, 2],
        )
        return mesh_x, mesh_y, mesh_z
