"""PKDGrav3 N-body data loader for BEoRN.

Provides :class:`PKDGravLoader`, a concrete implementation of
:class:`~beorn.load_input_data.nbody_base.NBodyLoader` for data produced by
the `PKDGrav3 <https://github.com/pkdgrav3/pkdgrav3>`_ N-body code.

**Expected data layout** (all paths relative to ``cosmo_sim.file_root``):

- ``<log_filename>`` — PKDGrav3 log file; column 1 is the redshift at each
  step (row index = step number − 1).
- ``<density_subpath>/<prefix>.<step>.den.256.0`` — binary float32 density
  grids on a 256³ mesh (or another resolution; controlled by
  ``simulation.Ncell``).
- ``<halo_subpath>/<prefix>.<step>.<halo_suffix>`` — halo catalog files
  produced by PKDGrav3's built-in FOF finder or an external code such as
  AHF or Rockstar.  Only snapshots that have a catalog file can contribute
  halos during painting; snapshots without one are treated as having zero
  halos.

The step number embedded in each filename is used to look up the
corresponding redshift in the log, so **partial datasets** (not every step
has every file type) are handled correctly.

Example — default PKDGrav3 FOF output::

    loader = PKDGravLoader(parameters)

Example — AHF catalogs in a separate directory, with user-supplied catalog::

    loader = PKDGravLoader(
        parameters,
        halo_subpath="AHF_output",
        halo_glob="*.AHF_halos",
        catalog_yml="my_catalog.yml",
    )
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

from .nbody_base import NBodyLoader
from ..structs import Parameters


def _guess_n_particles(file_root: Path, log_filename: str = "") -> Optional[int]:
    """Try to infer the particle count from the directory name or log filename.

    PKDGrav3 runs are commonly named ``CDM_<L>Mpc_<N>`` where ``N`` is the
    number of particles per side.  For example ``CDM_200Mpc_2048`` →
    2048³ = 8,589,934,592 particles.  This function checks the directory
    name first, then falls back to the log filename stem.

    Returns ``None`` if the pattern is not recognised.
    """
    # pattern: anything ending in _<digits>Mpc_<digits> or _<digits>
    pattern = r'_(\d+)(?:Mpc)?_(\d+)'
    for name in (file_root.name, Path(log_filename).stem):
        match = re.search(pattern, name)
        if match:
            n_side = int(match.group(2))
            return n_side ** 3
    return None


class PKDGravLoader(NBodyLoader):
    """Loader for PKDGrav3 N-body simulation data.

    Reads the PKDGrav3 log file to map step numbers to redshifts, then
    discovers density and halo files by globbing the configured directories.
    The halo-file glob pattern can be changed to support different halo
    finders (PKDGrav3 built-in FOF, AHF, Rockstar, …).

    All snapshot-catalog caching, partial-dataset handling, and informative
    logging is provided by the parent
    :class:`~beorn.load_input_data.nbody_base.NBodyLoader`.

    Args:
        parameters (Parameters): BEoRN parameter container.  Must have
            ``cosmo_sim.file_root`` pointing at the root of the N-body data.
        density_subpath (str): Subdirectory (relative to ``file_root``)
            containing the binary density-grid files.  Default: ``"nc256"``.
        halo_subpath (str): Subdirectory (relative to ``file_root``) containing
            halo catalog files.  Default: ``"haloes"``.
        halo_glob (str): Glob pattern used to discover halo catalog files
            inside ``halo_subpath``.  Change this to match the output format
            of AHF (``"*.AHF_halos"``), Rockstar (``"halos_*.ascii"``), etc.
            Default: ``"*.fof.txt"`` (PKDGrav3 built-in FOF finder).
        density_glob (str): Glob pattern for density files.
            Default: ``"*.den.256.0"``.
        log_filename (str): Name of the PKDGrav3 log file.
            Default: ``"CDM_200Mpc_2048.log"``.
        catalog_yml (Path | str | None): Path to an existing snapshot catalog
            YML file.  If supplied, BEoRN reads that file directly and skips
            the directory scan.  See :class:`~beorn.load_input_data.nbody_base.NBodyLoader`
            for details.  Default: ``None`` (auto-find or auto-create).
        halo_finder (str | None): Override the halo-finder name stored in the
            YML.  If ``None`` (default), the name is inferred from
            ``halo_glob``.
    """

    simulation_code = "PKDGrav3"

    #: Map halo-file glob patterns to human-readable finder names.
    _HALO_FINDER_NAMES: dict[str, str] = {
        "*.fof.txt":      "PKDGrav3 FOF (built-in)",
        "*.AHF_halos":    "AHF",
        "*.rockstar":     "Rockstar",
        "halos_*.ascii":  "Rockstar (ascii)",
    }

    def __init__(
        self,
        parameters: Parameters,
        density_subpath: str = "nc256",
        halo_subpath: str = "haloes",
        halo_glob: str = "*.fof.txt",
        density_glob: str = "*.den.256.0",
        log_filename: str = "CDM_200Mpc_2048.log",
        catalog_yml: Optional[Path] = None,
        halo_finder: Optional[str] = None,
    ):
        self._halo_glob = halo_glob
        self._density_glob = density_glob
        self._log_filename = log_filename

        file_root = Path(parameters.cosmo_sim.file_root)

        # Infer halo finder name from the glob pattern if not supplied.
        resolved_halo_finder = (
            halo_finder
            if halo_finder is not None
            else self._HALO_FINDER_NAMES.get(halo_glob, f"unknown (glob: {halo_glob})")
        )

        # Try to infer particle count from the directory or log filename,
        # e.g. "CDM_200Mpc_2048.log" → 2048³ particles.
        n_particles = _guess_n_particles(file_root, log_filename)

        super().__init__(
            parameters,
            density_dir=file_root / density_subpath,
            halo_dir=file_root / halo_subpath,
            catalog_yml=catalog_yml,
            halo_finder=resolved_halo_finder,
            n_particles=n_particles,
        )

    # ── NBodyLoader abstract interface ────────────────────────────────────────

    def discover_snapshots(self) -> list[dict]:
        """Scan the PKDGrav3 data directories and return the snapshot inventory.

        Reads the PKDGrav3 log file (column 1 = redshift at each step, with
        ``row_index = step_number − 1``) and matches density and halo files by
        their embedded step number.

        Returns:
            list[dict]: One entry per density file found, with keys
            ``"redshift"``, ``"density_file"``, and ``"halo_file"`` (the
            latter is ``None`` when no matching catalog exists for that step).
            File paths are relative to ``self.file_root``.
        """
        log_path = self.file_root / self._log_filename
        logs = np.loadtxt(log_path)
        log_z = logs[:, 1]  # redshift at step N is log_z[N - 1]

        def _step(path: Path) -> int:
            """Parse the zero-padded step number from a PKDGrav3 filename."""
            return int(path.name.split('.')[1])

        density_by_step = {
            _step(p): p for p in self.density_dir.glob(self._density_glob)
        }
        catalog_by_step = {
            _step(p): p for p in self.halo_dir.glob(self._halo_glob)
        }

        snapshots = []
        for step in sorted(density_by_step):
            row = step - 1
            if row < 0 or row >= len(log_z):
                self.logger.debug(
                    f"Step {step} has no corresponding row in {self._log_filename} — skipping."
                )
                continue

            density_rel = density_by_step[step].relative_to(self.file_root)
            halo_path = catalog_by_step.get(step)
            halo_rel = str(halo_path.relative_to(self.file_root)) if halo_path else None

            snapshots.append({
                "redshift": float(log_z[row]),
                "density_file": str(density_rel),
                "halo_file": halo_rel,
            })

        return snapshots

    def _read_halo_file(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Read a PKDGrav3 FOF catalog (or compatible text format).

        The expected file format has four whitespace-separated columns::

            mass  x  y  z

        where mass is in :math:`M_\\odot / h` and positions are in Mpc/h.

        Args:
            path (Path): Absolute path to the ``.fof.txt`` file.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - ``masses`` — halo masses in :math:`M_\\odot`.
                - ``positions`` — (N, 3) positions in comoving Mpc/h,
                  shifted so the box centre is at the origin.
        """
        catalog_array = np.loadtxt(path)
        if catalog_array.ndim == 1:
            if catalog_array.size == 0:
                return np.array([]), np.zeros((0, 3))
            catalog_array = catalog_array[np.newaxis, :]

        masses = catalog_array[:, 0] * self.parameters.cosmology.h0
        positions = catalog_array[:, 1:] + self.parameters.Lbox_hunits / 2
        return masses, positions

    def _read_density_file(self, path: Path) -> np.ndarray:
        """Read a PKDGrav3 binary density-grid file.

        The file contains a flat array of ``float32`` values representing the
        dark-matter particle count per cell on an
        ``Ncell × Ncell × Ncell`` grid stored in Fortran (column-major) order.

        Args:
            path (Path): Absolute path to the ``.den.256.0`` file.

        Returns:
            numpy.ndarray: 3D baryonic overdensity field
            :math:`\\delta_b = \\rho / \\langle\\rho\\rangle - 1`.
        """
        # Always reshape to the native file size — the base class applies
        # degrade_resolution block-averaging afterwards if requested.
        density_array = np.fromfile(path, dtype=np.float32)
        n = round(density_array.size ** (1 / 3))
        density = density_array.reshape(n, n, n).T
        return density / np.mean(density, dtype=np.float64) - 1

    def load_rsd_fields(self, redshift_index: int):
        """RSD fields are not available for PKDGrav3 in this implementation.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "RSD fields are not implemented for PKDGravLoader. "
            "Override this method in a subclass if velocity data is available."
        )
