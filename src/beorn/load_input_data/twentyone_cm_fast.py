from pathlib import Path
import h5py
import numpy as np

from ..structs.halo_catalog import HaloCatalog
from ..structs.parameters import Parameters


def load_halo_catalog(path: Path, parameters: Parameters) -> HaloCatalog:
    with h5py.File(path, 'r') as f:
        haloes = f['PerturbHaloField']
        # convert to numpy array as an intermediate step
        m, positions = haloes['halo_masses'], haloes['halo_coords']

        scaling = float(parameters.simulation.Lbox / parameters.simulation.Ncell)

        # 21cmfast quantities need to be rescaled
        return HaloCatalog(
            masses = np.asarray(m) * parameters.cosmology.h,
            positions = np.asarray(positions) * scaling, # + parameters.simulation.Lbox / 2
            parameters = parameters,
        )


def load_density_field(file: Path, LBox):
    with h5py.File(file, 'r') as f:
        field = f['PerturbedField']
        dens = field['density']

        dens_array = dens[:]
        return dens_array
