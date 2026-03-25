"""Global description of the 3d data computed over multiple redshifts."""
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np
import logging
import re
import tools21cm as t2c
logger = logging.getLogger(__name__)

from .base_struct import BaseStruct
from .coeval_cube import CoevalCube
from .base_quantities import GridBasePropertiesMixin
from .derived_quantities import GridDerivedPropertiesMixin
from .parameters import Parameters


@dataclass
class TemporalCube(BaseStruct, GridBasePropertiesMixin, GridDerivedPropertiesMixin):
    """
    Collection of grid data over multiple redshifts. This is implemented such that an additional z dimension is added to each field of the similar 'CoevalCube' class.
    Appending a new redshift to this data automatically appends to the underlying hdf5 file.
    As such, this class reuses all the grid data properties (which are implemented as base properties and derived properties in mixin classes). Only the z dimension is added here.
    """

    z: np.ndarray = None
    """Array of redshifts for which the grid data is available."""

    @staticmethod
    def _sanitize_component(value: str) -> str:
        """Return a path-safe identifier fragment."""
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
        return sanitized.strip("._-") or "simulation"

    @classmethod
    def simulation_name(cls, parameters: Parameters) -> str:
        """Infer a stable simulation name for snapshot file names."""
        file_root = getattr(parameters.simulation, "file_root", None)
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
        """Create an empty :class:`TemporalCube` and corresponding HDF5 file.

        Args:
            parameters (Parameters): Simulation parameters used to build file name.
            directory (Path): Directory where the HDF5 file will be created.
            snapshot_number (int|None): If provided, preallocate space for
                ``snapshot_number`` redshift slices (required for parallel runs).
            **kwargs: Additional name components used when generating the file path.

        Returns:
            TemporalCube: New instance with an on-disk HDF5 file created.
        """
        if snapshot_number is None:
            raise ValueError("snapshot_number is required when creating a TemporalCube manifest.")

        path = cls.get_file_path(directory, parameters, **kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_dir = cls.snapshot_directory(path)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        grid_shape = (
            snapshot_number,
            parameters.simulation.Ncell,
            parameters.simulation.Ncell,
            parameters.simulation.Ncell,
        )
        single_snapshot_shape = grid_shape[1:]

        helper = cls(
            z=None,
            parameters=parameters,
            delta_b=None,
            Grid_Temp=None,
            Grid_xHII=None,
            Grid_xal=None,
        )

        with h5py.File(path, "w", libver="latest") as hdf5_file:
            helper._to_h5_field(hdf5_file, "parameters", parameters)
            hdf5_file.create_dataset("z", data=np.full((snapshot_number,), np.nan, dtype=np.float64))

            for field_name in cls.grid_field_names(parameters):
                layout = h5py.VirtualLayout(shape=grid_shape, dtype=np.float64)
                for snapshot_index in range(snapshot_number):
                    snapshot_path = snapshot_dir / cls.snapshot_file_name(
                        parameters,
                        field_name,
                        snapshot_index,
                    )
                    source = h5py.VirtualSource(
                        str(snapshot_path),
                        field_name,
                        shape=single_snapshot_shape,
                    )
                    layout[snapshot_index, ...] = source
                hdf5_file.create_virtual_dataset(field_name, layout, fillvalue=0.0)

            hdf5_file.attrs["snapshot_storage"] = "per_snapshot_field_files"
            hdf5_file.attrs["snapshot_directory"] = str(snapshot_dir)
            hdf5_file.attrs["simulation_name"] = cls.simulation_name(parameters)

        return cls.read(file_path=path)


    def append(self, grid_snapshot: CoevalCube, index: int) -> None:
        """Append a :class:`CoevalCube` snapshot into the HDF5-backed collection.

        The method writes arrays from ``grid_snapshot`` into the HDF5
        datasets at position ``index``.
        Args:
            grid_snapshot (CoevalCube): Snapshot to append.
            index (int): Index/slot in the temporal dataset where the snapshot will be written.
        """
        if not isinstance(grid_snapshot, CoevalCube):
            raise TypeError("grid_snapshot must be an instance of GridData")

        if self._file_path is None:
            raise ValueError("File path is not set. Cannot append data.")

        snapshot_dir = self.snapshot_directory(self._file_path)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # NB: this could in theory have been made mpi-compatible: the h5py context can handle calls from different mpi ranks
        # but: this requires h5py to compiled against an mpi-compatible h5 backend
        # Instead, we use the precompiled h5py and simply assign a "master" process that handles the writing part centrally. No special care needed.
        with h5py.File(self._file_path, 'a') as hdf5_file:
            hdf5_file["z"][index] = float(grid_snapshot.z)

        for f in grid_snapshot._writable_fields():
            if f in ("z", "parameters"):
                continue

            value = getattr(grid_snapshot, f)

            if isinstance(value, h5py.Dataset):
                value = value[:]
            elif isinstance(value, (float, int, list)):
                value = np.array(value)

            if not isinstance(value, np.ndarray):
                logger.debug(f"Not appending {f} to {self._file_path.name} because type {type(value)} is not appendable.")
                continue

            snapshot_path = snapshot_dir / self.snapshot_file_name(self.parameters, f, index)
            with h5py.File(snapshot_path, "w") as snapshot_file:
                snapshot_file.create_dataset(f, data=value)
                snapshot_file.attrs["field"] = f
                snapshot_file.attrs["snapshot_index"] = int(index)
                snapshot_file.attrs["redshift"] = float(grid_snapshot.z)
                snapshot_file.attrs["grid_size"] = int(self.parameters.simulation.Ncell)
                snapshot_file.attrs["simulation_name"] = self.simulation_name(self.parameters)


    def power_spectrum(self, quantity: np.ndarray, parameters: Parameters) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D power spectra for a given grid quantity over all z.

        Args:
            quantity (np.ndarray): Array shaped (z, nx, ny, nz) to analyse.
            parameters (Parameters): Simulation parameters providing `kbins` and `Lbox`.

        Returns:
            tuple: ``(power_spectrum, bins)`` where ``power_spectrum`` has shape
            (n_z, n_k) and ``bins`` are the k-bin edges.
        """
        bin_number = parameters.simulation.kbins.size
        box_dims = parameters.simulation.Lbox
        power_spectrum = np.zeros((self.z.size, bin_number))

        delta_quantity = quantity[:] / np.mean(quantity, axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis] - 1

        for i, z in enumerate(self.z):
            power_spectrum[i, ...], bins = t2c.power_spectrum.power_spectrum_1d(delta_quantity[i, ...], box_dims=box_dims, kbins=bin_number)

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

        xHII_mean = np.mean(self.Grid_xHII, axis=(1, 2, 3))
        reionization_index = np.argmin(np.abs(xHII_mean - ionization_fraction))
        return reionization_index
