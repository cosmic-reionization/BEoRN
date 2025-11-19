from pathlib import Path
import h5py
import numpy as np

from .base import BaseLoader
from ..structs import HaloCatalog



class TwentyOneCmFastLoader(BaseLoader):
    """
    Loader for the 21cmfast data format. This
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_root = self.parameters.simulation.file_root


    def load_halo_catalog(self, redshift_index: int) -> HaloCatalog:
        redshift = self.redshifts[redshift_index]
        path = self.file_root / f'haloes_z{redshift}.h5'

        with h5py.File(path, 'r') as f:
            haloes = f['PerturbHaloField']
            # convert to numpy array as an intermediate step
            m, positions = haloes['halo_masses'], haloes['halo_coords']

            scaling = float(self.parameters.simulation.Lbox / self.parameters.simulation.Ncell)

            # 21cmfast quantities need to be rescaled
            return HaloCatalog(
                masses = np.asarray(m) * self.parameters.cosmology.h,
                positions = np.asarray(positions) * scaling, # + self.parameters.simulation.Lbox / 2 # is zero-centered needed?
                parameters = self.parameters,
            )


    def load_density_field(self, redshift_index: int) -> np.ndarray:
        redshift = self.redshifts[redshift_index]
        path = self.file_root / f'densities_z{redshift}.h5'

        with h5py.File(path, 'r') as f:
            field = f['PerturbedField']
            dens = field['density']

            dens_array = dens[:]
            return dens_array
