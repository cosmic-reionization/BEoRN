"""Global description of the 3d data computed over multiple redshifts."""
from dataclasses import dataclass
from typing import ClassVar
from pathlib import Path
import h5py
import numpy as np
import logging
import re
from tqdm.auto import tqdm

from .base_struct import BaseStruct
from .coeval_cube import CoevalCube
from .base_quantities import GridBasePropertiesMixin
from .derived_quantities import GridDerivedPropertiesMixin
from .parameters import Parameters

logger = logging.getLogger(__name__)


class LazyFieldArray:
    """Lazy 4D array backed by a list of per-redshift HDF5 files.

    Each file contributes one z-slice ``(Ncell, Ncell, Ncell)``.
    Slices are loaded from disk on first access and cached in memory so that
    repeated access to the same redshift is free.

    Supports standard numpy-style indexing:

    - ``arr[i]``          — load and return slice *i* (3D array).
    - ``arr[i, ...]``     — same, with trailing index forwarded.
    - ``arr[i:j]``        — load slices *i* through *j-1* and stack them.
    - ``arr[:]``          — load all slices.
    - ``np.asarray(arr)`` — materialise all slices into a 4D numpy array.

    Derived-quantity properties (e.g. ``Grid_dTb``) use ``__array__`` via
    numpy broadcasting, which transparently loads all slices. For
    memory-efficient global statistics iterate with ``arr[i]`` instead,
    or use :meth:`TemporalCube.global_mean`.
    """

    def __init__(self, z_files: list, field_name: str):
        self._z_files = list(z_files)
        self._field_name = field_name
        self._cache: dict[int, np.ndarray] = {}

        # Read slice shape from first file — metadata only, no data transferred.
        with h5py.File(self._z_files[0], 'r') as f:
            ds = f[field_name]
            self._slice_shape = ds.shape
            self._dtype = ds.dtype

    # ------------------------------------------------------------------ #
    # Array-like interface                                                 #
    # ------------------------------------------------------------------ #

    @property
    def shape(self) -> tuple:
        return (len(self._z_files), *self._slice_shape)

    @property
    def ndim(self) -> int:
        return 1 + len(self._slice_shape)

    @property
    def dtype(self):
        return self._dtype

    def _load(self, z_index: int) -> np.ndarray:
        idx = int(z_index) % len(self._z_files)
        if idx not in self._cache:
            with h5py.File(self._z_files[idx], 'r') as f:
                self._cache[idx] = f[self._field_name][:]
        return self._cache[idx]

    def __len__(self) -> int:
        return len(self._z_files)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._load(key)

        if isinstance(key, tuple):
            z_key, rest = key[0], key[1:]
            if isinstance(z_key, (int, np.integer)):
                data = self._load(z_key)
                return data[rest] if rest else data
            # Fall through for slice / array z-keys.

        if isinstance(key, slice):
            indices = range(*key.indices(len(self._z_files)))
            return np.stack([self._load(i) for i in indices])

        # General fallback: materialise everything and re-index.
        return self.__array__()[key]

    def __array__(self, dtype=None):
        data = np.stack([self._load(i) for i in range(len(self._z_files))])
        return data if dtype is None else data.astype(dtype)

    def __repr__(self) -> str:
        return (
            f"LazyFieldArray('{self._field_name}', shape={self.shape}, "
            f"cached {len(self._cache)}/{len(self._z_files)} z-slices)"
        )


@dataclass
class TemporalCube(BaseStruct, GridBasePropertiesMixin, GridDerivedPropertiesMixin):
    """Collection of grid data over multiple redshifts.

    Unlike other :class:`BaseStruct` subclasses, a :class:`TemporalCube` is
    stored as a **directory** of per-redshift HDF5 files rather than a single
    monolithic file:

    .. code-block:: text

        igm_data_{input_tag}_{beorn_hash}/
            CoevalCube_z7.000.h5
            CoevalCube_z7.500.h5
            ...
            CoevalCube_z15.000.h5

    Each ``CoevalCube_z{z:.3f}.h5`` file is a standard :class:`CoevalCube` HDF5 file
    containing ``delta_b``, ``Grid_Temp``, ``Grid_xHII``, and ``Grid_xal``
    for that redshift.  This layout means:

    - Snapshots can be added incrementally without touching existing files.
    - The directory doubles as a resume checkpoint — painting skips redshifts
      whose file already exists.
    - There is no separate ``CoevalCube_*_z_index=N.h5`` cache file.
    """

    z_snapshots: np.ndarray = None
    """Sorted array of redshifts for which snapshots are available on disk."""

    @property
    def z(self) -> np.ndarray:
        """Alias for :attr:`z_snapshots` — kept for backward compatibility."""
        return self.z_snapshots

    # ------------------------------------------------------------------ #
    # File-path helpers                                                    #
    # ------------------------------------------------------------------ #

    @classmethod
    def get_file_path(cls, directory: Path, parameters: Parameters, input_tag: str = None, **kwargs) -> Path:
        """Return the output **directory** path ``igm_data_{input_tag}_{beorn_hash}``.

        Args:
            input_tag (str, optional): Human-readable identifier for the upstream
                input data (e.g. ``'py21cmfast_N128_D384_L100_seed12345_9b1f5a85'``).
        """
        beorn_hash = parameters.beorn_hash()
        dir_name = f"igm_data_{input_tag}_{beorn_hash}" if input_tag else f"igm_data_{beorn_hash}"
        return directory / dir_name

    def snapshot_path(self, z: float) -> Path:
        """Return the path for the per-redshift file ``CoevalCube_z{z:.3f}.h5``."""
        return self._file_path / f"CoevalCube_z{z:.3f}.h5"

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sanitize_component(value: str) -> str:
        """Return a path-safe identifier fragment."""
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
        return sanitized.strip("._-") or "simulation"

    @classmethod
    def simulation_name(cls, parameters: Parameters) -> str:
        """Infer a stable simulation name for snapshot file names."""
        # cosmo_sim.file_root is the canonical location; simulation.file_root is
        # a legacy fallback used by some run scripts that set it dynamically.
        for section in ("cosmo_sim", "simulation"):
            file_root = getattr(getattr(parameters, section, None), "file_root", None)
            if isinstance(file_root, Path):
                candidate = file_root.name or file_root.stem
                if candidate:
                    return cls._sanitize_component(candidate)
        return "simulation"

    @classmethod
    def snapshot_directory(cls, cube_path: Path) -> Path:
        """Return the directory containing per-snapshot field files."""
        return cube_path.parent / f"{cube_path.stem}_snapshots"

    @classmethod
    def snapshot_file_name(cls, parameters: Parameters, field: str, snapshot_index: int) -> str:
        """Build the per-snapshot field file name."""
        simulation_name = cls.simulation_name(parameters)
        ncell = int(parameters.simulation.Ncell)
        field_name = cls._sanitize_component(field)
        return f"{simulation_name}_snapshot_{snapshot_index:03d}_{field_name}_N{ncell}.h5"

    @classmethod
    def grid_field_names(cls, parameters: Parameters) -> list[str]:
        """Return all grid-like fields stored in the temporal cube manifest."""
        fields = ["delta_b", "Grid_Temp", "Grid_xHII", "Grid_xal"]
        for field_name in parameters.simulation.store_grids:
            if field_name not in fields:
                fields.append(field_name)
        return fields

    @classmethod
    def create_empty(cls, parameters: Parameters, directory: Path, snapshot_number: int = None, **kwargs) -> "TemporalCube":
        """Create an empty :class:`TemporalCube` output directory.

        Args:
            parameters (Parameters): Simulation parameters used to build the directory name.
            directory (Path): Parent directory under which the output folder is created.
            snapshot_number: Ignored — kept for API compatibility. Per-z files are
                written on demand so no pre-allocation is needed.
            **kwargs: Forwarded to :meth:`get_file_path` (e.g. ``input_tag``).

        Returns:
            TemporalCube: New instance whose ``_file_path`` points to the created directory.
        """
        path = cls.get_file_path(directory, parameters, **kwargs)
        path.mkdir(parents=True, exist_ok=True)
        if parameters is not None:
            # Redshifts are redundant here: profile redshifts live in the
            # RadiationProfiles cache; snapshot redshifts are inferrable from
            # the CoevalCube_z*.h5 filenames on disk.
            parameters.to_yaml(
                path / "igm_params.yaml",
                exclude_keys={
                    # Redshifts are redundant: profile z lives in the RadiationProfiles
                    # cache; snapshot z is inferrable from CoevalCube_z*.h5 filenames.
                    "solver.redshifts",
                    "cosmo_sim.snapshot_redshifts",
                    # Simulator-specific fields that don't belong in a generic IGM output:
                    # py21cmfast internals (seed is in the dir name)
                    "cosmo_sim.IC_seed",
                },
            )
        ret = cls(z_snapshots=None, parameters=parameters, delta_b=None, Grid_Temp=None, Grid_xHII=None, Grid_xal=None)
        ret._file_path = path
        return ret

    # ------------------------------------------------------------------ #
    # Per-snapshot I/O                                                     #
    # ------------------------------------------------------------------ #

    def append(self, grid_snapshot: CoevalCube, index: int) -> None:
        """Write a :class:`CoevalCube` snapshot as ``z{z:.3f}.h5`` inside the directory.

        Args:
            grid_snapshot (CoevalCube): Snapshot to write.
            index (int): Unused — kept for API compatibility. The file name is
                derived from ``grid_snapshot.z``.
        """
        if not isinstance(grid_snapshot, CoevalCube):
            raise TypeError("grid_snapshot must be an instance of CoevalCube")
        if self._file_path is None:
            raise ValueError("Output directory is not set. Call create_empty() first.")
        path = self.snapshot_path(grid_snapshot.z)
        original_file_path = grid_snapshot._file_path
        try:
            # The same CoevalCube may already have been written to a cache
            # location. Clear the bound path temporarily so the temporal output
            # write targets this snapshot file instead of reusing the cache path.
            grid_snapshot._file_path = None
            grid_snapshot.write(file_path=path)
        finally:
            grid_snapshot._file_path = original_file_path

    # ------------------------------------------------------------------ #
    # Loading                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def _resolve_path(cls, file_path, directory, parameters, kwargs):
        if file_path is not None and (directory or kwargs):
            raise ValueError("Provide either file_path or directory+parameters, not both.")
        if file_path is None:
            file_path = cls.get_file_path(directory, parameters, **kwargs)
        return Path(file_path)

    @classmethod
    def read(cls, file_path: "Path | str" = None, directory: Path = None, parameters: Parameters = None, **kwargs):
        """Open a :class:`TemporalCube` directory without loading grid data.

        Scans for ``CoevalCube_z*.h5`` files and records the available
        redshifts from their filenames.  Grid arrays (``Grid_xHII``,
        ``Grid_Temp``, etc.) are **not** loaded into memory; call
        :meth:`load_grids` when you need them.

        The simplest way to reopen a completed run from its output folder::

            cube = TemporalCube.read("igm_data_<tag>_<hash>/")

        Parameters are recovered automatically from the embedded HDF5 metadata
        (or from ``parameters.yaml`` if present), so no :class:`Parameters`
        object is required.

        Args:
            file_path (Path | str): Direct path to the output directory.
            directory / parameters / kwargs: Alternatively, derive the path via
                :meth:`get_file_path`.
        """
        file_path = cls._resolve_path(file_path, directory, parameters, kwargs)

        z_files = sorted(file_path.glob("CoevalCube_z*.h5"), key=lambda p: float(p.stem[len("CoevalCube_z"):]))
        if not z_files:
            logger.warning(f"No per-redshift files found in {file_path}. Returning empty TemporalCube.")
            ret = cls(z_snapshots=None, parameters=parameters, delta_b=None, Grid_Temp=None, Grid_xHII=None, Grid_xal=None)
            ret._file_path = file_path
            return ret

        # Extract z values from filenames only — no HDF5 reads needed.
        z_values = [float(p.stem[len("CoevalCube_z"):]) for p in z_files]

        # Identify which grid fields are present and load parameters — one file open.
        grid_fields = ['delta_b', 'Grid_Temp', 'Grid_xHII', 'Grid_xal']
        loaded_parameters = parameters
        with h5py.File(z_files[0], 'r') as f:
            available_fields = [name for name in grid_fields if name in f]
            if loaded_parameters is None and 'parameters' in f:
                loaded_parameters = Parameters.from_group(f['parameters'])

        # Ensure igm_params.yaml exists in the folder so the directory is self-contained.
        yaml_path = file_path / "igm_params.yaml"
        if not yaml_path.exists() and loaded_parameters is not None:
            loaded_parameters.to_yaml(yaml_path)
            logger.info(
                "igm_params.yaml not found in output folder — recreated from embedded HDF5 parameters "
                f"and written to {yaml_path}."
            )

        # Attach a LazyFieldArray for each available field — no data loaded yet.
        lazy = {name: LazyFieldArray(z_files, name) for name in available_fields}

        ret = cls(
            z_snapshots=np.array(z_values),
            parameters=loaded_parameters,
            delta_b=lazy.get('delta_b'),
            Grid_Temp=lazy.get('Grid_Temp'),
            Grid_xHII=lazy.get('Grid_xHII'),
            Grid_xal=lazy.get('Grid_xal'),
        )
        ret._file_path = file_path
        logger.info(f"Opened {len(z_files)} snapshots from {file_path} (grid data is lazy — slices load on demand).")
        return ret

    def load_grids(self):
        """Materialise all lazy grid fields into 4D numpy arrays.

        After this call every grid field (``Grid_xHII``, ``Grid_Temp``, etc.)
        is a plain ``(n_z, Ncell, Ncell, Ncell)`` numpy array held entirely
        in RAM. Useful when you need repeated random access across all
        redshifts or want to pass data to code that expects a numpy array.

        If a field is already a numpy array it is left untouched.

        Returns:
            self: For convenient chaining, e.g. ``cube = TemporalCube.read(...).load_grids()``.
        """
        grid_fields = ['delta_b', 'Grid_Temp', 'Grid_xHII', 'Grid_xal']
        for name in tqdm(grid_fields, desc="Loading grid fields", unit="field"):
            val = getattr(self, name, None)
            if isinstance(val, LazyFieldArray):
                setattr(self, name, val.__array__())

        logger.info("All grid fields materialised into numpy arrays.")
        return self

    _FIELD_ALIASES: ClassVar[dict] = {
        'dTb':  'Grid_dTb',
        'Temp': 'Grid_Temp',
        'xHII': 'Grid_xHII',
        'xal':  'Grid_xal',
        'delta_b': 'delta_b',
    }

    def global_mean(self, field: str) -> np.ndarray:
        """Compute the spatial mean of *field* at each redshift without loading all z into RAM.

        Iterates over snapshots one at a time, so peak memory usage is one
        ``(Ncell, Ncell, Ncell)`` array regardless of how many redshifts exist.

        Short aliases are accepted: ``'dTb'``, ``'Temp'``, ``'xHII'``, ``'xal'``.

        Args:
            field (str): Attribute name or short alias, e.g. ``'xHII'`` or ``'Grid_xHII'``.

        Returns:
            np.ndarray: 1D array of shape ``(n_z,)`` with the spatial mean per redshift.
        """
        field = self._FIELD_ALIASES.get(field, field)
        arr = getattr(self, field, None)
        if arr is None:
            raise ValueError(f"Field '{field}' is not available on this TemporalCube.")
        means = np.empty(len(self.z_snapshots))
        for i in range(len(self.z_snapshots)):
            means[i] = np.mean(arr[i])
        return means

    def snapshot(self, z: "int | float") -> CoevalCube:
        """Load a single redshift snapshot as a :class:`CoevalCube`.

        Only the HDF5 file for that redshift is opened, so memory usage is
        one 3D snapshot.  Derived quantities (e.g. ``Grid_dTb``) are computed
        from 3D arrays instead of the full 4D multi-z arrays.

        Args:
            z (int | float): Redshift index (int) **or** redshift value (float).
                When a float is given the nearest available snapshot is returned.

        Returns:
            CoevalCube: Snapshot for the requested redshift.
        """
        if isinstance(z, (int, np.integer)):
            z_val = float(self.z_snapshots[int(z)])
        else:
            z_val = float(self.z_snapshots[np.argmin(np.abs(self.z_snapshots - z))])
        path = self.snapshot_path(z_val)
        if not path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {path}")
        return CoevalCube.read(file_path=path, parameters=self.parameters)

    def __post_init__(self):
        # Directory-backed TemporalCubes are loaded explicitly via read().
        # Skip BaseStruct.__post_init__() which expects an HDF5 file.
        pass

    def write(self, **kwargs):
        # TemporalCube has no monolithic file to write; snapshots are written
        # individually via append().  This is a deliberate no-op.
        pass

    # ------------------------------------------------------------------ #
    # Analysis                                                             #
    # ------------------------------------------------------------------ #

    def power_spectrum(
        self,
        quantity: np.ndarray,
        parameters: Parameters,
        mass_assignment: str | None = None,
        deconvolve: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D power spectra for a given grid quantity over all z.

        Args:
            quantity (np.ndarray): Array shaped (z, nx, ny, nz) to analyse.
            parameters (Parameters): Simulation parameters providing `kbins` and `Lbox`.
            mass_assignment: ``'NGP'``, ``'CIC'``, ``'TSC'``, or ``'PCS'`` — the
                scheme ``quantity`` was painted with. Required if ``deconvolve=True``.
            deconvolve: If ``True``, undo the mass-assignment window (issue #48)
                before binning — fixes the high-k suppression it introduces as a
                pure numerical artefact. See
                :func:`beorn.particle_mapping.deconvolve_mas` for why this is a
                P(k)-only fix (not safe to apply to the field itself).

        Returns:
            tuple: ``(power_spectrum, bins)`` where ``power_spectrum`` has shape
            (n_z, n_k) and ``bins`` are the k-bin edges.
        """
        from ..power_spectrum import power_spectrum_1d

        bin_number = parameters.simulation.kbins.size
        box_dims = parameters.Lbox_hunits
        power_spectrum = np.zeros((self.z_snapshots.size, bin_number))

        delta_quantity = quantity[:] / np.mean(quantity, axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis] - 1

        kny = None
        for i, z in enumerate(tqdm(self.z_snapshots, desc='Power spectrum', unit='snapshot')):
            power_spectrum[i, ...], bins, kny = power_spectrum_1d(
                delta_quantity[i, ...], box_dims=box_dims, kbins=bin_number,
                mass_assignment=mass_assignment, deconvolve=deconvolve,
            )
        if kny is not None:
            logger.info(f"TemporalCube.power_spectrum: k_Nyquist = {kny:.4g}")

        return power_spectrum, bins


    def redshift_of_reionization(self, ionization_fraction: float = 0.5) -> int:
        """Return the redshift index where the volume-averaged ionization crosses a threshold.

        Args:
            ionization_fraction (float): Threshold volume-averaged ionization fraction. Default 0.5.

        Returns:
            int: Index in the time/redshift array corresponding to the crossing.
        """
        if self.Grid_xHII is None:
            raise ValueError("Grid_xHII is not available.")

        xHII_mean = self.global_mean('Grid_xHII')
        reionization_index = np.argmin(np.abs(xHII_mean - ionization_fraction))
        return reionization_index
