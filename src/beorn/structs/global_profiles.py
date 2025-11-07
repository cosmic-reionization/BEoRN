"""Global description of the 3d data computed over multiple redshifts."""
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np
import logging
import tools21cm as t2c
logger = logging.getLogger(__name__)

from .base_struct import BaseStruct
from .snapshot_profiles import GridData
from .base_quantities import GridBasePropertiesMixin
from .derived_quantities import GridDerivedPropertiesMixin
from .parameters import Parameters


@dataclass
class GridDataMultiZ(BaseStruct, GridBasePropertiesMixin, GridDerivedPropertiesMixin):
    """
    Collection of grid data over multiple redshifts. This is implemented such that an additional z dimension is added to each field of the GridData class.
    Appending a new redshift to this data automatically appends to the underlying hdf5 file.
    As such, this class reuses all the grid data properties (which are implemented as base properties and derived properties in mixin classes). Only the z dimension is added here.
    """

    z: np.ndarray = None
    """Array of redshifts for which the grid data is available."""

    @classmethod
    def create_empty(cls, parameters: Parameters, directory: Path, **kwargs) -> "GridDataMultiZ":
        """
        Creates an empty HDF5 file with the given file path. If the file already exists, it is not overwritten.
        """
        path = cls.get_file_path(directory, parameters, **kwargs)
        ret = cls(
            z = None,
            parameters = parameters,
            delta_b = None,
            Grid_Temp = None,
            Grid_xHII = None,
            Grid_xal = None,
        )
        # set after initialization to avoid reading from that file on construction
        ret._file_path = path
        path.touch(exist_ok=True)
        return ret


    def append(self, grid_data: GridData) -> None:
        """
        Append a new GridData (for another redshift snapshot) to the collection of grid data.
        """
        # TODO make compliant with a MPI implementation
        if not isinstance(grid_data, GridData):
            raise TypeError("grid_data must be an instance of GridData")

        if self._file_path is None:
            raise ValueError("File path is not set. Cannot append data.")

        with h5py.File(self._file_path, 'a') as hdf5_file:
            for f in grid_data._writable_fields():
                value = getattr(grid_data, f)

                if isinstance(value, (float, int, list)):
                    # Convert float to numpy array so that they can still be appended
                    value = np.array(value)

                elif isinstance(value, h5py.Dataset):
                    # If the value is already a h5py.Dataset, we can directly append it
                    value = value[:]

                elif isinstance(value, Parameters):
                    if f not in hdf5_file.keys():
                        self._to_h5_field(hdf5_file, f, value)
                    else:
                        logger.debug(f"Not overriding {f} in {self._file_path.name}")
                    continue

                elif not isinstance(value, np.ndarray):
                    logger.debug(f"Not appending {f} to {self._file_path.name} because type {type(value)} is not appendable.")
                    continue

                if f not in hdf5_file:
                    # Create a new dataset if it doesn't exist
                    hdf5_file.create_dataset(
                        f,
                        data = value[np.newaxis, ...],
                        maxshape = (None, *value.shape)
                        # explicitly set the maxshape to allow for appending
                        )
                else:
                    # Append to the existing dataset
                    dataset = hdf5_file[f]
                    dataset.resize((dataset.shape[0] + 1, *dataset.shape[1:]))
                    dataset[-1] = value


    def power_spectrum(self, quantity: np.ndarray, parameters: Parameters) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the power spectrum of the given quantity over all redshifts.
        """
        bin_number = parameters.simulation.kbins.size
        box_dims = parameters.simulation.Lbox
        power_spectrum = np.zeros((self.z.size, bin_number))

        delta_quantity = quantity[:] / np.mean(quantity, axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis] - 1

        for i, z in enumerate(self.z):
            power_spectrum[i, ...], bins = t2c.power_spectrum.power_spectrum_1d(delta_quantity[i, ...], box_dims=box_dims, kbins=bin_number)

        return power_spectrum, bins


    def redshift_of_reionization(self, ionization_fraction: float = 0.5) -> int:
        """
        Compute the redshift of reionization, defined as the redshift at which the volume-averaged ionization fraction crosses the given threshold.
        Parameters
        ----------
        ionization_fraction : float
            The ionization fraction threshold to define the redshift of reionization. Default is 0.5.
        Returns
        -------
        int
            The index of the redshift at which the volume-averaged ionization fraction crosses the threshold.
        """
        if self.Grid_xHII is None:
            raise ValueError("Grid_xHII is not available.")

        xHII_mean = np.mean(self.Grid_xHII, axis=(1, 2, 3))
        reionization_index = np.argmin(np.abs(xHII_mean - ionization_fraction))
        return reionization_index
