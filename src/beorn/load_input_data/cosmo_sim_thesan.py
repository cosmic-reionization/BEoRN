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

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .merger_tree_base import MergerTreeLoader
from ..particle_mapping import map_particles_to_mesh


@dataclass
class SubboxConfig:
    """Defines a cubic subregion of the periodic simulation box to paint.

    The painted volume is the *target* subbox expanded by a buffer on every
    side.  Including the buffer halos in the FFT convolution prevents the
    periodic wrap artefact from contaminating the edges of the target region:
    sources just outside the target still illuminate its boundary correctly.
    Discard the outer ``buffer`` Mpc/h of each face from the output grids to
    recover the target subbox.

    All quantities are in comoving Mpc/h, matching the units used inside
    :class:`ThesanLoader` after unit conversion.

    Args:
        origin: Lower corner of the *target* subbox (before buffer), shape (3,).
        size:   Side length of the cubic target subbox [Mpc/h].
        buffer: Width of the buffer region added to every face [Mpc/h].
        lbox_full: Side length of the parent periodic box [Mpc/h].
    """
    origin: np.ndarray   # (3,) lower corner of target, Mpc/h
    size: float          # target side length, Mpc/h
    buffer: float        # buffer width on each face, Mpc/h
    lbox_full: float     # parent box side length, Mpc/h

    @property
    def lbox_eff(self) -> float:
        """Side length of the painted region including buffer on both sides."""
        return self.size + 2.0 * self.buffer

    @property
    def lo_corner(self) -> np.ndarray:
        """Lower corner of the buffered region (may be negative; wraps periodically)."""
        return self.origin - self.buffer


def _subbox_filter(
    positions: np.ndarray,
    subbox: SubboxConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a boolean mask and remapped coordinates for the buffered subbox.

    Positions are shifted so that the buffered lower corner maps to the origin,
    with periodic wrapping from the parent box applied first.  Only positions
    that land inside ``[0, lbox_eff)`` on every axis are kept.

    The mask is built one axis at a time so that only one column-sized temporary
    (~N floats) is live at a time, rather than a full N×3 shifted copy.  This
    matters for the 1050³ particle case where a naïve vectorised shift would
    create an extra 13 GB intermediate and cause OOM on per-rank memory budgets.

    Args:
        positions: (N, 3) array of comoving positions in Mpc/h.
        subbox:    Subbox configuration.

    Returns:
        inside:       Boolean mask of shape (N,), True for kept entries.
        pos_remapped: (M, 3) positions shifted to ``[0, lbox_eff)`` for the
                      M kept entries (already filtered; no further indexing needed).
    """
    lo = subbox.lo_corner
    # Build mask column-by-column; each axis temp is freed before the next.
    inside = np.ones(positions.shape[0], dtype=bool)
    for i in range(3):
        col = (positions[:, i] - lo[i]) % subbox.lbox_full
        inside &= col < subbox.lbox_eff
        del col
    # Remap only the kept entries to [0, lbox_eff).
    n_kept = int(inside.sum())
    pos_remapped = np.empty((n_kept, 3), dtype=positions.dtype)
    for i in range(3):
        pos_remapped[:, i] = (positions[inside, i] - lo[i]) % subbox.lbox_full
    return inside, pos_remapped


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

    def __init__(
        self,
        parameters,
        cache_file: Path | str,
        *,
        is_high_res: bool = False,
        subbox: SubboxConfig | None = None,
    ):
        super().__init__(parameters)

        self.thesan_root      = Path(self.parameters.cosmo_sim.file_root)
        self.tree_root        = self.thesan_root / "postprocessing" / "trees" / "LHaloTree"
        self.snapshot_path_root = self.thesan_root / "output"
        self.offset_path_root = self.thesan_root / "postprocessing" / "offsets"
        self.cached_tree      = Path(cache_file)
        self.particle_count   = 2100 ** 3 if is_high_res else 1050 ** 3
        # When set, all spatial I/O (halos and density particles) is restricted
        # to the buffered subregion and coordinates are remapped to [0, lbox_eff).
        self.subbox           = subbox

        self.logger.info(f"Initialized ThesanLoader — root: {self.thesan_root}")
        if subbox is not None:
            self.logger.info(
                f"Subbox enabled — origin={subbox.origin} Mpc/h, "
                f"size={subbox.size} Mpc/h, buffer={subbox.buffer} Mpc/h, "
                f"effective box={subbox.lbox_eff:.2f} Mpc/h"
            )

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
        backend = getattr(cosmo_sim, "particle_mapping_backend", "numpy")
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
        positions /= (1e3 * self.thesan_h)   # kpc/h → Mpc/h
        masses    *= 1e10 / self.thesan_h    # 10^10 M☉/h → M☉

        if self.subbox is not None:
            # Keep only groups whose centre falls inside the buffered subbox and
            # remap their coordinates to [0, lbox_eff).  The subhalo→group map
            # uses group indices into the full catalog, so we must remap it to
            # the compressed group array produced by the spatial mask.
            # _subbox_filter returns already-remapped positions for kept entries.
            inside, positions = _subbox_filter(positions, self.subbox)
            # Build a compact group array and update the subhalo→group index map.
            old_to_new = np.full(masses.size, -1, dtype=np.int64)
            old_to_new[inside] = np.arange(inside.sum(), dtype=np.int64)
            masses               = masses[inside]
            # positions is already the filtered+remapped subset from _subbox_filter.
            # Remap subhalo→group indices; subhalos whose group was removed get -1.
            valid_sh = (subhalo_to_group_map >= 0) & (subhalo_to_group_map < old_to_new.size)
            new_map = np.full(subhalo_to_group_map.size, -1, dtype=np.int64)
            new_map[valid_sh] = old_to_new[subhalo_to_group_map[valid_sh]]
            subhalo_to_group_map = new_map
            self.logger.debug(
                f"Subbox halo filter: kept {inside.sum():,} / {inside.size:,} groups"
            )

        return positions, masses, subhalo_to_group_map

    # ── Density and velocity fields ────────────────────────────────────────

    def load_density_field(self, redshift_index: int) -> np.ndarray:
        """Paint DM particles onto a mesh and return the overdensity field δ = ρ/⟨ρ⟩ − 1.

        Particles are processed one HDF5 file at a time and painted directly into
        the output mesh.  The full particle_count × 3 array (~14 GB for 1050³) is
        never allocated; peak RAM per iteration is O(chunk_size), typically ~100 MB.
        map_particles_to_mesh accumulates into the mesh, so calling it once per
        chunk is equivalent to calling it once on the concatenated array.
        """
        snapshot_path = self._density_directories[redshift_index]
        snapshots = sorted(snapshot_path.glob("snap_*.hdf5"))

        mesh_size = self.parameters.simulation.Ncell
        mesh = np.zeros((mesh_size, mesh_size, mesh_size), dtype=np.float32)
        mass_assignment, backend = self._particle_mapping_config()

        if self.subbox is not None:
            lo = self.subbox.lo_corner
            physical_size = self.subbox.lbox_eff
            for snap in snapshots:
                with h5py.File(snap, "r") as f:
                    pos = f["PartType1"]["Coordinates"][:].astype(np.float32)
                pos *= 1e-3 / self.thesan_h              # kpc/h → Mpc/h, in-place
                # Build mask axis-by-axis; free each column temp immediately.
                inside = np.ones(pos.shape[0], dtype=bool)
                for i in range(3):
                    col = (pos[:, i] - lo[i]) % self.subbox.lbox_full
                    inside &= col < self.subbox.lbox_eff
                    del col
                if inside.any():
                    kept = pos[inside]
                    # Remap kept positions to [0, lbox_eff) in-place.
                    for i in range(3):
                        kept[:, i] = (kept[:, i] - lo[i]) % self.subbox.lbox_full
                    map_particles_to_mesh(
                        mesh, physical_size, kept,
                        mass_assignment=mass_assignment, backend=backend,
                    )
                del pos, inside
        else:
            physical_size = self.parameters.simulation.Lbox
            for snap in snapshots:
                with h5py.File(snap, "r") as f:
                    pos = f["PartType1"]["Coordinates"][:].astype(np.float32)
                pos *= 1e-3 / self.thesan_h              # kpc/h → Mpc/h, in-place
                map_particles_to_mesh(
                    mesh, physical_size, pos,
                    mass_assignment=mass_assignment, backend=backend,
                )
                del pos

        return mesh / np.mean(mesh, dtype=np.float64) - 1

    def load_rsd_fields(self, redshift_index: int):
        """Return per-axis velocity meshes for redshift-space distortions.

        Applies the same subbox spatial filter as :meth:`load_density_field`
        when a subbox is active, so velocity grids are consistent with the
        density and halo grids.
        """
        snapshot_path = self._density_directories[redshift_index]
        snapshots = list(snapshot_path.glob("snap_*.hdf5"))

        mesh_size = self.parameters.simulation.Ncell
        mesh_x = np.zeros((mesh_size, mesh_size, mesh_size), dtype=np.float32)
        mesh_y = mesh_x.copy()
        mesh_z = mesh_x.copy()

        scale_factor = 1 / (1 + self.redshifts[redshift_index])
        vel_scale = 1.0 / (self.thesan_h * np.sqrt(scale_factor))  # combined unit factor
        mass_assignment, backend = self._particle_mapping_config()

        if self.subbox is not None:
            lo = self.subbox.lo_corner
            Lbox = self.subbox.lbox_eff
            for snap in snapshots:
                with h5py.File(snap, "r") as f:
                    pos = f["PartType1"]["Coordinates"][:].astype(np.float32)
                    vel = f["PartType1"]["Velocities"][:].astype(np.float32)
                pos *= 1e-3 / self.thesan_h
                vel *= vel_scale                          # peculiar km/s, in-place
                inside = np.ones(pos.shape[0], dtype=bool)
                for i in range(3):
                    col = (pos[:, i] - lo[i]) % self.subbox.lbox_full
                    inside &= col < self.subbox.lbox_eff
                    del col
                if inside.any():
                    kept_pos = pos[inside]
                    kept_vel = vel[inside]
                    for i in range(3):
                        kept_pos[:, i] = (kept_pos[:, i] - lo[i]) % self.subbox.lbox_full
                    map_particles_to_mesh(mesh_x, Lbox, kept_pos, mass_assignment=mass_assignment, backend=backend, weights=kept_vel[:, 0])
                    map_particles_to_mesh(mesh_y, Lbox, kept_pos, mass_assignment=mass_assignment, backend=backend, weights=kept_vel[:, 1])
                    map_particles_to_mesh(mesh_z, Lbox, kept_pos, mass_assignment=mass_assignment, backend=backend, weights=kept_vel[:, 2])
                del pos, vel, inside
        else:
            Lbox = self.parameters.simulation.Lbox
            for snap in snapshots:
                with h5py.File(snap, "r") as f:
                    pos = f["PartType1"]["Coordinates"][:].astype(np.float32)
                    vel = f["PartType1"]["Velocities"][:].astype(np.float32)
                pos *= 1e-3 / self.thesan_h
                vel *= vel_scale
                map_particles_to_mesh(mesh_x, Lbox, pos, mass_assignment=mass_assignment, backend=backend, weights=vel[:, 0])
                map_particles_to_mesh(mesh_y, Lbox, pos, mass_assignment=mass_assignment, backend=backend, weights=vel[:, 1])
                map_particles_to_mesh(mesh_z, Lbox, pos, mass_assignment=mass_assignment, backend=backend, weights=vel[:, 2])
                del pos, vel

        return mesh_x, mesh_y, mesh_z
