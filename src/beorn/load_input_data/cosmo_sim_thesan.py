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

import os
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
        cache_file (Path | str | None): Path to the flat HDF5 tree cache
            produced by the cache extraction cell in the notebook.  Snapshot
            redshifts and other provenance metadata are read from its root
            attributes and the ``snapshot_redshifts`` dataset.

            Pass ``None`` for a **density/velocity-only loader**: redshifts are
            then read directly from the ``snapdir_XXX`` snapshot headers, and
            the available-snapshot set is every ``snapdir_XXX`` on disk rather
            than the intersection of ``groups_*``/``snapdir_*``/``offsets_*``
            (painting a density or velocity mesh needs neither the merger trees
            nor the group catalogs).  In this mode the tree- and catalog-backed
            methods — :meth:`load_merger_tree_data` and
            :meth:`get_halo_information_from_catalog` — are unavailable and
            raise; use it for grid preprocessing, not for a full solver run.
        is_high_res (bool): ``True`` for THESAN-DARK 1 (2100³ particles),
            ``False`` (default) for THESAN-DARK 2 (1050³ particles).
        density_cache_dir (Path | str | None): When set, painted density
            meshes are cached as ``.npy`` files in this directory, keyed by
            snapshot, mesh geometry, mass-assignment scheme, and subbox.
            Painting a mesh requires streaming the full particle set
            (~minutes per snapshot); the cached mesh loads in milliseconds.
            Point multiple runs at the same directory to share the cache —
            the mesh is independent of all source/painting parameters.
        velocity_cache_dir (Path | str | None): Same idea as
            ``density_cache_dir`` but for the three RSD velocity meshes
            returned by :meth:`load_rsd_fields`, cached together as one
            ``.npz`` file per snapshot. Can point at the same directory as
            ``density_cache_dir``; the file name prefix keeps them apart.
    """

    simulation_code = "THESAN"

    @staticmethod
    def _snapshot_index_from_name(path: Path) -> int:
        """Extract the integer snapshot index from names like groups_039 or offsets_039.hdf5."""
        suffix = path.stem if path.is_file() else path.name
        return int(suffix.split("_")[1])

    @staticmethod
    def _redshifts_from_snapshot_headers(density_dirs: dict[int, Path]) -> np.ndarray:
        """Redshift per snapshot number, read from the ``snapdir_XXX`` headers.

        Returns an array indexed by *snapshot number* (so it can be indexed the
        same way as the tree cache's ``snapshot_redshifts`` dataset).  Snapshot
        numbers with no ``snapdir`` on disk are filled with NaN, which compares
        False against any redshift range and so drops out of the selection.
        """
        if not density_dirs:
            raise FileNotFoundError(
                "No snapdir_* directories found; cannot determine snapshot redshifts "
                "without a tree cache."
            )
        redshifts = np.full(max(density_dirs) + 1, np.nan, dtype=np.float64)
        for snap_index, snap_dir in density_dirs.items():
            try:
                snap_file = next(snap_dir.glob("snap_*.hdf5"))
            except StopIteration:
                continue
            with h5py.File(snap_file, "r") as f:
                redshifts[snap_index] = float(f["Header"].attrs["Redshift"])
        return redshifts

    def __init__(
        self,
        parameters,
        cache_file: Path | str | None = None,
        *,
        is_high_res: bool = False,
        subbox: SubboxConfig | None = None,
        density_cache_dir: Path | str | None = None,
        velocity_cache_dir: Path | str | None = None,
    ):
        super().__init__(parameters)

        self.thesan_root      = Path(self.parameters.cosmo_sim.file_root)
        self.tree_root        = self.thesan_root / "postprocessing" / "trees" / "LHaloTree"
        self.snapshot_path_root = self.thesan_root / "output"
        self.offset_path_root = self.thesan_root / "postprocessing" / "offsets"
        self.cached_tree      = Path(cache_file) if cache_file is not None else None
        self.particle_count   = 2100 ** 3 if is_high_res else 1050 ** 3
        # When set, all spatial I/O (halos and density particles) is restricted
        # to the buffered subregion and coordinates are remapped to [0, lbox_eff).
        self.subbox           = subbox
        self.density_cache_dir = Path(density_cache_dir) if density_cache_dir is not None else None
        self.velocity_cache_dir = Path(velocity_cache_dir) if velocity_cache_dir is not None else None

        self.logger.info(f"Initialized ThesanLoader — root: {self.thesan_root}")
        if subbox is not None:
            self.logger.info(
                f"Subbox enabled — origin={subbox.origin} Mpc/h, "
                f"size={subbox.size} Mpc/h, buffer={subbox.buffer} Mpc/h, "
                f"effective box={subbox.lbox_eff:.2f} Mpc/h"
            )

        if self.cached_tree is not None and not self.cached_tree.exists():
            raise FileNotFoundError(
                f"Tree cache not found: {self.cached_tree}. "
                "Run the cache extraction cell in the notebook first."
            )

        # Index available directories by snapshot number (parsed from the name,
        # e.g. "groups_100" → 100).  Using a dict means only the snapshots you
        # actually downloaded need to be present — a subset is fine.
        catalogs     = {self._snapshot_index_from_name(p): p
                        for p in self.snapshot_path_root.glob("groups_*")}
        density_dirs = {self._snapshot_index_from_name(p): p
                        for p in self.snapshot_path_root.glob("snapdir_*")}
        offset_files = {self._snapshot_index_from_name(p): p
                        for p in self.offset_path_root.glob("offsets_*")}

        if self.density_cache_dir is not None:
            # A snapdir is ~2 TB, so they are routinely deleted once the density
            # meshes have been cached.  Painting only needs the cached mesh, so
            # accept those snapshots: the snapdir path is still constructed (the
            # cache key is derived from its name), and load_density_field raises a
            # clear error if a cache miss ever reaches a snapdir that is gone.
            cache_only = []
            for snap_index in sorted(catalogs):
                if snap_index in density_dirs:
                    continue
                snapdir_name = f"snapdir_{snap_index:03d}"
                if self._density_cache_path_for_snapdir_name(snapdir_name).exists():
                    density_dirs[snap_index] = self.snapshot_path_root / snapdir_name
                    cache_only.append(snap_index)
            if cache_only:
                self.logger.info(
                    f"{len(cache_only)} snapshot(s) have no snapdir on disk but do have a "
                    f"cached density mesh — accepted (snapshots {cache_only[0]}–{cache_only[-1]})"
                )

        if self.cached_tree is not None:
            # Read snapshot redshifts from the HDF5 cache (stored as a dataset
            # alongside the provenance attributes — no separate YAML needed).
            with h5py.File(self.cached_tree, "r") as f:
                redshifts = f["snapshot_redshifts"][:]
            self.logger.info(
                f"Loaded {redshifts.size} snapshot redshifts from {self.cached_tree}"
            )
            # Trees and catalogs are in play, so every product must be present.
            available_snapshot_indices = set(catalogs) & set(density_dirs) & set(offset_files)
            missing_data_message = (
                "No THESAN snapshots within the requested solver.redshifts range have a complete "
                "set of groups_*, snapdir_*, and offsets_* data."
            )
        else:
            # Density/velocity-only mode: the snapshot headers are the source of
            # truth for redshift, and a snapdir is the only product required.
            redshifts = self._redshifts_from_snapshot_headers(density_dirs)
            self.logger.info(
                f"Loaded {len(density_dirs)} snapshot redshifts from snapdir headers "
                "(no tree cache)"
            )
            available_snapshot_indices = set(density_dirs)
            missing_data_message = (
                "No THESAN snapdir_* directories fall within the requested "
                "solver.redshifts range."
            )

        # Restrict to the solver redshift range
        z_min = min(self.parameters.solver.redshifts)
        z_max = max(self.parameters.solver.redshifts)
        indices = np.where((redshifts >= z_min) & (redshifts <= z_max))[0]
        self._snap_indices = np.array(
            [int(i) for i in indices if int(i) in available_snapshot_indices],
            dtype=int,
        )

        if self._snap_indices.size == 0:
            raise FileNotFoundError(missing_data_message)

        self._redshifts = redshifts[self._snap_indices]
        self._catalog_directories = [catalogs.get(i) for i in self._snap_indices]
        self._density_directories = [density_dirs.get(i) for i in self._snap_indices]
        self._offset_files = [offset_files.get(i) for i in self._snap_indices]

        self.logger.info(
            f"THESAN snapshots: {self._redshifts.size} "
            f"(z={self._redshifts.max():.2f} → {self._redshifts.min():.2f})"
        )

        # Read h from first snapshot header, when a raw snapdir is actually on disk.
        # Falls back to parameters.cosmology.h0 for snapdir-less runs (see the
        # cache-only handling above) — verified to match the on-disk THESAN-1 header
        # value (0.6774) via the surviving group-catalog files.
        try:
            first_snap = next(self._density_directories[0].glob("snap_*.hdf5"))
        except StopIteration:
            self.thesan_h = self.parameters.cosmology.h0
            self.logger.info(
                f"No snap_*.hdf5 found for snapshot {self._snap_indices[0]} to read "
                f"HubbleParam from; falling back to parameters.cosmology.h0 = {self.thesan_h}"
            )
        else:
            with h5py.File(first_snap, "r") as f:
                self.thesan_h = f["Header"].attrs["HubbleParam"]

    # ── BaseLoader interface ───────────────────────────────────────────────

    @property
    def redshifts(self) -> np.ndarray:
        return self._redshifts

    @property
    def snapshot_numbers(self) -> np.ndarray:
        """THESAN snapshot indices (e.g. the ``050`` in ``groups_050``) available.

        Parallel to :attr:`redshifts` — ``snapshot_numbers[i]`` is the raw
        snapshot number backing ``redshifts[i]``. Useful for scripts that
        need to select a specific ``redshift_index`` by its on-disk snapshot
        number rather than by position in the (possibly range-filtered)
        ``redshifts`` array.
        """
        return self._snap_indices

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

        The arrays are memoized on first use: the file holds ~3.6e8 entries (~6 GB,
        ~30 s to read) and one alpha fit is performed per painted snapshot, so
        re-reading it every time dominates the run.  Peak memory is unchanged — the
        arrays were already materialised on every call.
        """
        if getattr(self, "_tree_cache_arrays", None) is not None:
            return self._tree_cache_arrays
        if self.cached_tree is None:
            raise RuntimeError(
                "This ThesanLoader was built without a tree cache "
                "(cache_file=None), so merger-tree data is unavailable. "
                "Construct it with cache_file=<tree_cache_v2.hdf5> for a full run."
            )
        with h5py.File(self.cached_tree, "r") as f:
            tree_halo_ids       = f["tree_halo_ids"][:]
            tree_snap_num       = f["tree_snap_num"][:]
            tree_mass           = f["tree_mass"][:]
            tree_main_progenitor = f["tree_main_progenitor"][:]
        self.logger.debug(f"Loaded tree cache: {self.cached_tree} ({tree_halo_ids.size:,} entries)")
        self._tree_cache_arrays = (tree_halo_ids, tree_snap_num, tree_mass, tree_main_progenitor)
        return self._tree_cache_arrays

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

    def _density_cache_path(self, redshift_index: int) -> Path | None:
        """Cache file for the painted density mesh of one snapshot, or None.

        The key covers everything the mesh depends on: the snapshot, the mesh
        geometry (Ncell/Lbox — already overridden to the effective values in
        subbox mode), the mass-assignment scheme, and the subbox region.  The
        mesh is independent of all source/painting parameters, so the cache can
        be shared between runs with different astrophysics.
        """
        if self.density_cache_dir is None:
            return None
        return self._density_cache_path_for_snapdir_name(
            self._density_directories[redshift_index].name
        )

    def _density_cache_path_for_snapdir_name(self, snapdir_name: str) -> Path:
        """Cache file for a snapdir identified by name (e.g. ``"snapdir_042"``).

        Split out of :meth:`_density_cache_path` so ``__init__`` can test whether a
        snapshot has a cached mesh before ``self._density_directories`` exists.
        Requires ``density_cache_dir`` to be set.
        """
        mass_assignment, _ = self._particle_mapping_config()
        sim = self.parameters.simulation
        tag = f"{snapdir_name}_N{sim.Ncell}_L{sim.Lbox:.6g}_{mass_assignment}"
        if self.subbox is not None:
            o = self.subbox.origin
            tag += (
                f"_subbox_o{o[0]:.6g}-{o[1]:.6g}-{o[2]:.6g}"
                f"_s{self.subbox.size:.6g}_b{self.subbox.buffer:.6g}"
            )
        return self.density_cache_dir / f"delta_{tag}.npy"

    def _velocity_cache_path(self, redshift_index: int) -> Path | None:
        """Cache file for the painted velocity meshes of one snapshot, or None.

        Same key as :meth:`_density_cache_path` (the velocity meshes depend
        on exactly the same mesh geometry, mass-assignment scheme, and
        subbox region); only the file prefix differs so the two caches can
        share a directory.
        """
        if self.velocity_cache_dir is None:
            return None
        snapshot_path = self._density_directories[redshift_index]
        mass_assignment, _ = self._particle_mapping_config()
        sim = self.parameters.simulation
        tag = f"{snapshot_path.name}_N{sim.Ncell}_L{sim.Lbox:.6g}_{mass_assignment}"
        if self.subbox is not None:
            o = self.subbox.origin
            tag += (
                f"_subbox_o{o[0]:.6g}-{o[1]:.6g}-{o[2]:.6g}"
                f"_s{self.subbox.size:.6g}_b{self.subbox.buffer:.6g}"
            )
        return self.velocity_cache_dir / f"vel_{tag}.npz"

    def load_density_field(self, redshift_index: int) -> np.ndarray:
        """Paint DM particles onto a mesh and return the overdensity field δ = ρ/⟨ρ⟩ − 1.

        Particles are processed one HDF5 file at a time and painted directly into
        the output mesh.  The full particle_count × 3 array (~14 GB for 1050³) is
        never allocated; peak RAM per iteration is O(chunk_size), typically ~100 MB.
        map_particles_to_mesh accumulates into the mesh, so calling it once per
        chunk is equivalent to calling it once on the concatenated array.

        When ``density_cache_dir`` is set, the resulting mesh is cached on disk
        and reloaded instead of re-streaming the particle snapshot.
        """
        cache_path = self._density_cache_path(redshift_index)
        if cache_path is not None and cache_path.exists():
            self.logger.info(f"Loading cached density mesh from {cache_path}")
            return np.load(cache_path)

        snapshot_path = self._density_directories[redshift_index]
        if not snapshot_path.exists():
            # Accepted in __init__ on the strength of a cached mesh that has since
            # gone missing (or a changed Ncell/Lbox/mass-assignment cache key).
            # Painting an empty mesh would yield delta = nan everywhere, so stop here.
            raise FileNotFoundError(
                f"Snapshot directory {snapshot_path} does not exist and the density-mesh "
                f"cache entry {cache_path} is missing, so the mesh cannot be (re)computed. "
                "Re-run thesan_snapshot_preprocess.py for this snapshot, or restore the snapdir."
            )
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

        delta = mesh / np.mean(mesh, dtype=np.float64) - 1

        if cache_path is not None:
            # Atomic write (tmp + rename): concurrent MPI ranks computing the
            # same snapshot race harmlessly — both produce identical content.
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(f".tmp{os.getpid()}.npy")
            np.save(tmp_path, delta)
            os.replace(tmp_path, cache_path)
            self.logger.info(f"Cached density mesh to {cache_path}")

        return delta

    def load_rsd_fields(self, redshift_index: int):
        """Return per-axis velocity meshes for redshift-space distortions.

        Applies the same subbox spatial filter as :meth:`load_density_field`
        when a subbox is active, so velocity grids are consistent with the
        density and halo grids.

        When ``velocity_cache_dir`` is set, the resulting meshes are cached
        on disk (as one ``.npz`` per snapshot) and reloaded instead of
        re-streaming the particle snapshot.
        """
        cache_path = self._velocity_cache_path(redshift_index)
        if cache_path is not None and cache_path.exists():
            self.logger.info(f"Loading cached velocity meshes from {cache_path}")
            with np.load(cache_path) as cached:
                return cached["vx"], cached["vy"], cached["vz"]

        snapshot_path = self._density_directories[redshift_index]
        snapshots = list(snapshot_path.glob("snap_*.hdf5"))

        mesh_size = self.parameters.simulation.Ncell
        mesh_x = np.zeros((mesh_size, mesh_size, mesh_size), dtype=np.float32)
        mesh_y = mesh_x.copy()
        mesh_z = mesh_x.copy()

        scale_factor = 1 / (1 + self.redshifts[redshift_index])
        # THESAN stores DM velocities as km*sqrt(a)/s; peculiar velocity in km/s
        # is sqrt(a) * v_raw (a_scaling=0.5, no h-scaling — unlike Coordinates,
        # which are ckpc/h). See thesan-project.com/thesan/snapshots.html.
        vel_scale = np.sqrt(scale_factor)
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

        if cache_path is not None:
            # Atomic write (tmp + rename), same rationale as load_density_field.
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(f".tmp{os.getpid()}.npz")
            np.savez(tmp_path, vx=mesh_x, vy=mesh_y, vz=mesh_z)
            os.replace(tmp_path, cache_path)
            self.logger.info(f"Cached velocity meshes to {cache_path}")

        return mesh_x, mesh_y, mesh_z
