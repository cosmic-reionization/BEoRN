"""Generic base loader for N-body simulation data.

Provides :class:`NBodyLoader`, an abstract mid-level class that sits between
:class:`~beorn.load_input_data.base.BaseLoader` and concrete format-specific
loaders (e.g. :class:`~beorn.load_input_data.cosmo_sim_pkdgrav.PKDGravLoader`).

Responsibilities handled here so subclasses do not have to repeat them:

- **Separate density / halo directories** — density grids and halo catalogs
  often live in different folders and/or come from different codes (PKDGrav3
  built-in FOF, AHF, Rockstar, …).
- **Optional YML snapshot catalog** — when ``catalog_yml`` is provided, BEoRN
  writes the snapshot list to that path on the first run and reads it back on
  subsequent runs, skipping the directory scan.  Pass the same path each run
  to cache results.  Omit it to discover snapshots fresh every time.
- **Manual snapshot list** — pass ``snapshots=`` to bypass
  :meth:`discover_snapshots` entirely and supply the list of
  ``{redshift, density_file, halo_file}`` dicts directly.
- **Partial datasets** — snapshots without a halo catalog return an empty
  :class:`~beorn.structs.HaloCatalog` so painting can continue without error.
- **Duplicate removal** — logs sometimes contain duplicate redshift entries
  (restarts); these are filtered automatically.
"""
from __future__ import annotations

import hashlib
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseLoader
from ..structs import Parameters, HaloCatalog

_FORMAT_VERSION = 1
logger = logging.getLogger(__name__)


class NBodyLoader(BaseLoader):
    """Abstract base loader for N-body simulation data.

    Subclass this and implement :meth:`_read_halo_file` and
    :meth:`_read_density_file`.  Override :meth:`discover_snapshots` when you
    want automatic directory scanning; otherwise pass the snapshot list via
    the ``snapshots`` argument.

    Args:
        parameters (Parameters): BEoRN parameter container.  Must have
            ``cosmo_sim.file_root`` set to the root directory of the N-body
            data.
        density_dir (Path | str | None): Directory containing density-grid
            files.  Defaults to ``file_root``.
        halo_dir (Path | str | None): Directory containing halo-catalog files.
            Defaults to ``file_root``.
        catalog_yml (Path | str | None): Path to a snapshot catalog YML file.

            - If the file **exists**, BEoRN loads from it (no scan).
            - If the path **does not exist**, BEoRN runs :meth:`discover_snapshots`
              (or uses ``snapshots=``) and **writes** the result to that path.
            - If ``None`` (default), snapshots are discovered or taken from
              ``snapshots=`` and **no file is written**.

        snapshots (list[dict] | None): Supply the snapshot inventory directly
            instead of calling :meth:`discover_snapshots`.  Each dict must
            have keys ``"redshift"`` (float), ``"density_file"`` (str, relative
            to ``file_root``), and ``"halo_file"`` (str | None).  A YML is
            written when ``catalog_yml`` is also given.
        halo_finder (str): Name of the halo-finder code (e.g. ``"FOF"``,
            ``"AHF"``).  Stored in the YML metadata.  Default: ``"unknown"``.
        n_particles (int | None): Total simulation particle count.  Stored in
            YML metadata.  ``None`` → ``"unknown"``.
        degrade_resolution: Read from ``parameters.simulation.degrade_resolution``
            (default 1 = no degradation).  Values > 1 block-average each density
            grid by that factor before returning it, e.g. ``degrade_resolution=4``
            turns a 256³ grid into 64³.  Set ``parameters.simulation.Ncell`` to
            the *degraded* grid size so that the rest of BEoRN uses the correct
            resolution.

    Class attributes (override in subclasses):
        simulation_code (str): Name of the N-body code, e.g. ``"PKDGrav3"``.
    """

    simulation_code: str = "unknown"

    def __init__(
        self,
        parameters: Parameters,
        density_dir: Optional[Path] = None,
        halo_dir: Optional[Path] = None,
        catalog_yml: Optional[Path] = None,
        snapshots: Optional[list] = None,
        halo_finder: str = "unknown",
        n_particles: Optional[int] = None,
    ):
        super().__init__(parameters)
        self.file_root = Path(self.parameters.cosmo_sim.file_root)
        self.density_dir = Path(density_dir) if density_dir is not None else self.file_root
        self.halo_dir = Path(halo_dir) if halo_dir is not None else self.file_root
        self.halo_finder = halo_finder
        self.n_particles = n_particles

        yml_path = Path(catalog_yml) if catalog_yml is not None else None

        # ── Resolve snapshot list ─────────────────────────────────────────────
        if snapshots is not None:
            # Manual override — use provided list directly.
            all_snapshots = list(snapshots)
            if yml_path is not None:
                self._write_catalog_yml(yml_path, all_snapshots)
            else:
                self._log_creation_summary(None, all_snapshots, mass_stats=None)

        elif yml_path is not None and yml_path.exists():
            # YML exists → load from it, skip discovery.
            all_snapshots = self._load_catalog_yml(yml_path, user_supplied=True)

        elif yml_path is not None:
            # YML path given but file missing → discover and write.
            all_snapshots = self.discover_snapshots()
            self._write_catalog_yml(yml_path, all_snapshots)

        else:
            # No YML, no manual list → discover without caching.
            all_snapshots = self.discover_snapshots()
            logger.info(
                "Snapshots discovered from disk (no catalog_yml provided — "
                "results are not cached). Pass catalog_yml='path/to/catalog.yml' "
                "to save the snapshot list for future runs."
            )

        self._build_snapshot_lists(all_snapshots)
        self.remove_duplicates()

        # ── Auto-set Ncell from the first density file ────────────────────────
        if self.density_paths:
            native_n = self._peek_grid_size(self.density_paths[0])
            self._apply_degrade_resolution(native_n)
            if self._degrade_factor <= 1:
                logger.info(f"Auto-set Ncell={self.parameters.simulation.Ncell} from density file.")

    # ── Abstract interface for subclasses ─────────────────────────────────────

    def discover_snapshots(self) -> list[dict]:
        """Scan the data directory and return the full snapshot inventory.

        Override this in your subclass to implement automatic discovery.  It is
        only called when no ``snapshots=`` list is provided and no existing YML
        is loaded.

        Returns:
            list[dict]: One dict per snapshot, with keys:

            - ``"redshift"`` (float) — snapshot redshift.
            - ``"density_file"`` (str) — path to the density-grid file,
              **relative to** ``self.file_root``.
            - ``"halo_file"`` (str | None) — relative path to the halo
              catalog, or ``None`` if absent.

        Include **all** discovered snapshots (no z-range filtering);
        filtering is applied in :meth:`_build_snapshot_lists`.

        Raises:
            NotImplementedError: If not overridden and no ``snapshots=`` list
                was supplied at construction time.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} did not override discover_snapshots(). "
            "Either implement it in your subclass or pass a snapshot list via "
            "the snapshots= argument."
        )

    @abstractmethod
    def _read_halo_file(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Read a single halo catalog file.

        Args:
            path (Path): Absolute path to the file.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - ``masses`` — 1D array of halo masses in :math:`M_\\odot`.
                - ``positions`` — (N, 3) array of positions in comoving Mpc/h.
        """

    @abstractmethod
    def _read_density_file(self, path: Path) -> np.ndarray:
        """Read a single density-grid file.

        Args:
            path (Path): Absolute path to the file.

        Returns:
            numpy.ndarray: 3D overdensity field
            :math:`\\delta_b = \\rho_b / \\langle\\rho_b\\rangle - 1`
            with shape ``(Ncell, Ncell, Ncell)``.
        """

    def _peek_grid_size(self, path: Path) -> int:
        """Return the native cubic grid dimension without fully reading the file.

        The default implementation assumes a flat binary file of 32-bit floats
        (N³ elements → N = cbrt(file_size / 4)).  Override this in subclasses
        for other formats (HDF5, netCDF, FITS, …).

        Args:
            path (Path): Absolute path to a density file.

        Returns:
            int: Native grid dimension N (file stores an N×N×N cube).
        """
        n_elements = path.stat().st_size // 4  # float32 = 4 bytes
        return round(n_elements ** (1 / 3))

    # ── Concrete BaseLoader implementations ───────────────────────────────────

    @property
    def redshifts(self) -> np.ndarray:
        """Redshifts of snapshots selected by ``solver.redshifts``."""
        return self._redshifts

    def load_halo_catalog(self, redshift_index: int) -> HaloCatalog:
        """Load the halo catalog for a given snapshot.

        Returns an empty catalog for snapshots that have no halo file so that
        painting can continue without error.

        Args:
            redshift_index (int): Index into :attr:`redshifts`.

        Returns:
            HaloCatalog: Catalog with ``masses`` (M☉) and ``positions`` (Mpc/h).
        """
        halo_path = self.catalogs[redshift_index]
        z = float(self._redshifts[redshift_index])

        if halo_path is None:
            return HaloCatalog(
                masses=np.array([]),
                positions=np.zeros((0, 3)),
                parameters=self.parameters,
                redshift_index=redshift_index,
                redshift=z,
            )

        masses, positions = self._read_halo_file(halo_path)
        return HaloCatalog(
            masses=masses,
            positions=positions,
            parameters=self.parameters,
            redshift_index=redshift_index,
            redshift=z,
        )

    def load_density_field(self, redshift_index: int) -> np.ndarray:
        """Load the baryonic density field for a given snapshot.

        If ``degrade_resolution`` was set at construction time the field is
        block-averaged by that factor before being returned.

        Args:
            redshift_index (int): Index into :attr:`redshifts`.

        Returns:
            numpy.ndarray: 3D overdensity field, optionally at reduced resolution.
        """
        delta = self._read_density_file(self.density_paths[redshift_index])
        if self._degrade_factor > 1:
            delta = self._coarsen_density(delta, self._degrade_factor)
        return delta

    def _coarsen_density(self, delta: np.ndarray, factor: int) -> np.ndarray:
        """Block-average a 3D density field by an integer factor.

        Each ``factor³`` block of cells is averaged into one output cell.
        This conserves mean density: ⟨δ_coarse⟩ = ⟨δ_fine⟩.

        Thin wrapper around :func:`beorn.particle_mapping.coarsen_field`
        (shared with the fine-paint-then-downsample path, issue #48).

        Args:
            delta (np.ndarray): Input 3D overdensity array of shape (N, N, N).
            factor (int): Downsampling factor.  N must be divisible by factor.

        Returns:
            np.ndarray: Coarsened array of shape (N//factor, N//factor, N//factor).

        Raises:
            ValueError: If any axis of ``delta`` is not divisible by ``factor``.
        """
        from ..particle_mapping import coarsen_field
        return coarsen_field(delta, factor)

    def load_rsd_fields(self, redshift_index: int):
        """Not implemented for generic N-body loaders.

        Override in a subclass if velocity/RSD fields are available.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide RSD fields. "
            "Override load_rsd_fields() in a subclass if velocity data is available."
        )

    @property
    def input_tag(self) -> str:
        """Short identifier for this dataset, used to namespace output files."""
        sim = self.parameters.simulation
        h = hashlib.md5(str(self.file_root.resolve()).encode()).hexdigest()[:8]
        return f"nbody_N{sim.Ncell}_L{int(sim.Lbox)}_{h}"

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_snapshot_lists(self, all_snapshots: list[dict]) -> None:
        """Filter ``all_snapshots`` to the requested z range and populate lists."""
        z_values = self.parameters.solver.redshifts
        z_min = min(z_values[0], z_values[-1])
        z_max = max(z_values[0], z_values[-1])

        redshifts_list, density_list, catalog_list = [], [], []
        for snap in all_snapshots:
            z = snap["redshift"]
            if z < z_min or z > z_max:
                continue
            density_file = snap.get("density_file")
            if density_file is None:
                continue
            redshifts_list.append(z)
            density_list.append(self.file_root / density_file)
            halo_file = snap.get("halo_file")
            catalog_list.append(self.file_root / halo_file if halo_file else None)

        self._redshifts = np.array(redshifts_list)
        self.density_paths = density_list
        self.catalogs = catalog_list

    def remove_duplicates(self) -> None:
        """Remove duplicate redshift entries (e.g. from simulation restarts)."""
        z_prev = np.inf
        zz, dens, cats = [], [], []
        for z, d, c in zip(self._redshifts, self.density_paths, self.catalogs):
            if z < z_prev:
                zz.append(z)
                dens.append(d)
                cats.append(c)
                z_prev = z
            else:
                logger.debug(f"Skipping duplicate redshift z={z:.3f}")
        self._redshifts = np.array(zz)
        self.density_paths = dens
        self.catalogs = cats

    def _halo_mass_stats(self, all_snapshots: list[dict]) -> dict:
        """Read the halo catalog at the lowest available redshift and return mass stats."""
        candidates = [s for s in all_snapshots if s.get("halo_file") is not None]
        if not candidates:
            return {
                "z": "unknown",
                "min_mass_Msol": "unknown",
                "max_mass_Msol": "unknown",
                "n_halos": "unknown",
            }

        lowest = min(candidates, key=lambda s: s["redshift"])
        path = self.file_root / lowest["halo_file"]
        try:
            masses, _ = self._read_halo_file(path)
            if len(masses) == 0:
                return {
                    "z": float(lowest["redshift"]),
                    "min_mass_Msol": "unknown",
                    "max_mass_Msol": "unknown",
                    "n_halos": 0,
                }
            return {
                "z": float(lowest["redshift"]),
                "min_mass_Msol": float(masses.min()),
                "max_mass_Msol": float(masses.max()),
                "n_halos": int(len(masses)),
            }
        except Exception as exc:
            logger.warning(f"Could not read halo mass stats from {path}: {exc}")
            return {
                "z": float(lowest["redshift"]),
                "min_mass_Msol": "unknown",
                "max_mass_Msol": "unknown",
                "n_halos": "unknown",
            }

    # ── YML catalog I/O ───────────────────────────────────────────────────────

    def _write_catalog_yml(self, path: Path, all_snapshots: list[dict]) -> None:
        """Write a snapshot catalog YML to ``path``."""
        try:
            import yaml
        except ImportError:
            logger.warning("pyyaml not installed — skipping catalog YML write.")
            self._log_creation_summary(None, all_snapshots, mass_stats=None)
            return

        sim = self.parameters.simulation
        cosmo = self.parameters.cosmology

        logger.info("Reading halo mass statistics from the lowest-redshift catalog…")
        mass_stats = self._halo_mass_stats(all_snapshots)

        doc = {
            "format_version": _FORMAT_VERSION,
            "generated_by": "beorn",
            "file_root": str(self.file_root.resolve()),
            "simulation": {
                "code": self.simulation_code,
                "n_particles": self.n_particles if self.n_particles is not None else "unknown",
                "Ncell": sim.Ncell,
                "Lbox_Mpc_per_h": float(sim.Lbox),
            },
            "halo_finder": self.halo_finder,
            "cosmology": {
                "Om": cosmo.Om,
                "Ob": cosmo.Ob,
                "Ol": cosmo.Ol,
                "h0": cosmo.h0,
            },
            "halo_mass_stats": mass_stats,
            "snapshots": all_snapshots,
        }

        try:
            with open(path, "w") as f:
                yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
        except PermissionError:
            logger.warning(
                f"Cannot write catalog YML to {path} (read-only filesystem). "
                "Continuing without caching."
            )
            path = None

        self._log_creation_summary(path, all_snapshots, mass_stats)

    def _load_catalog_yml(self, path: Path, user_supplied: bool) -> list[dict]:
        """Load a snapshot catalog from a YML file and log a summary."""
        try:
            import yaml
        except ImportError:
            logger.warning("pyyaml not installed — rescanning data directory instead.")
            return self.discover_snapshots()

        with open(path) as f:
            doc = yaml.safe_load(f)

        version = doc.get("format_version", 0)
        if version != _FORMAT_VERSION:
            logger.warning(
                f"Catalog YML format version {version} != expected {_FORMAT_VERSION}. "
                "Rescanning data directory."
            )
            return self.discover_snapshots()

        snapshots = doc.get("snapshots", [])
        self._log_load_summary(path, doc, snapshots, user_supplied)
        return snapshots

    # ── Logging helpers ───────────────────────────────────────────────────────

    def _log_creation_summary(
        self,
        yml_path: Optional[Path],
        all_snapshots: list[dict],
        mass_stats: Optional[dict],
    ) -> None:
        sim = self.parameters.simulation
        n_density = sum(1 for s in all_snapshots if s.get("density_file"))
        n_halo = sum(1 for s in all_snapshots if s.get("halo_file"))
        z_vals = [s["redshift"] for s in all_snapshots if s.get("density_file")]
        z_range = (
            f"z={max(z_vals):.2f} → {min(z_vals):.2f}"
            if z_vals else "no density files found"
        )

        ms = mass_stats or {}
        if ms.get("n_halos") not in (None, "unknown", 0):
            mass_line = (
                f"  Halo mass range :  {ms['min_mass_Msol']:.2e} — "
                f"{ms['max_mass_Msol']:.2e} M☉  "
                f"({ms['n_halos']:,} halos at z={ms['z']:.2f})"
            )
        else:
            mass_line = "  Halo mass range :  unknown (no catalogs found)"

        if yml_path is not None:
            yml_line = f"  Catalog written  :  {yml_path}"
        else:
            yml_line = "  Catalog          :  not written (no catalog_yml provided)"

        logger.info(
            "\n" + "=" * 62 + "\n"
            "  BEoRN N-body snapshot inventory\n"
            + "=" * 62 + "\n"
            f"  Simulation code  :  {self.simulation_code}\n"
            f"  Halo finder      :  {self.halo_finder}\n"
            f"  N particles      :  {self.n_particles or 'unknown'}\n"
            f"  Grid             :  {sim.Ncell}³ cells,  {sim.Lbox} Mpc/h\n"
            f"  Density files    :  {n_density} snapshots  ({z_range})\n"
            f"  Halo catalogs    :  {n_halo} snapshots\n"
            + mass_line + "\n"
            + yml_line + "\n"
            + "=" * 62
        )

    def _log_load_summary(
        self,
        yml_path: Path,
        doc: dict,
        all_snapshots: list[dict],
        user_supplied: bool,
    ) -> None:
        sim_meta = doc.get("simulation", {})
        mass_stats = doc.get("halo_mass_stats", {})
        source = "user-supplied" if user_supplied else "auto-found"

        n_density = sum(1 for s in all_snapshots if s.get("density_file"))
        n_halo = sum(1 for s in all_snapshots if s.get("halo_file"))
        z_vals = [s["redshift"] for s in all_snapshots if s.get("density_file")]
        z_range = (
            f"z={max(z_vals):.2f} → {min(z_vals):.2f}"
            if z_vals else "none"
        )

        ms = mass_stats
        if ms.get("n_halos") not in (None, "unknown", 0):
            mass_line = (
                f"  Halo mass range  :  {ms['min_mass_Msol']:.2e} — "
                f"{ms['max_mass_Msol']:.2e} M☉  "
                f"({ms['n_halos']:,} halos at z={ms['z']:.2f})"
            )
        else:
            mass_line = "  Halo mass range  :  unknown"

        logger.info(
            "\n" + "=" * 62 + "\n"
            f"  Reading BEoRN snapshot catalog  [{source}]\n"
            + "=" * 62 + "\n"
            f"  File             :  {yml_path}\n"
            f"  Simulation code  :  {sim_meta.get('code', 'unknown')}\n"
            f"  Halo finder      :  {doc.get('halo_finder', 'unknown')}\n"
            f"  N particles      :  {sim_meta.get('n_particles', 'unknown')}\n"
            f"  Grid             :  {sim_meta.get('Ncell', '?')}³ cells,  "
            f"{sim_meta.get('Lbox_Mpc_per_h', '?')} Mpc/h\n"
            f"  Density files    :  {n_density} total  ({z_range})\n"
            f"  Halo catalogs    :  {n_halo} total\n"
            + mass_line + "\n"
            + "=" * 62
        )
