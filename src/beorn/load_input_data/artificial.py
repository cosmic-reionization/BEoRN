import numpy as np

from .base import BaseLoader
from ..structs import HaloCatalog


def exponential_mass_accretion(z, M0, z0, alpha):
    return M0 * np.exp(-alpha * (z - z0))


class ArtificialHaloLoader(BaseLoader):
    """
    Loader for your reference and quick testing. It maps <halo_count> halos onto a grid and simulates exponential growth.
    """
    def __init__(self, parameters, halo_count: int, seed: int = 12345, final_mass: float = 1e12, alpha: float = 0.79):
        super().__init__(parameters)
        rng = np.random.default_rng(seed)
        self.X = rng.random(halo_count) * self.parameters.simulation.Lbox
        self.Y = rng.random(halo_count) * self.parameters.simulation.Lbox
        self.Z = rng.random(halo_count) * self.parameters.simulation.Lbox

        self.mass_at_z6 = np.full(halo_count, final_mass)
        self.alpha = alpha

    @property
    def redshifts(self):
        return np.flip(np.arange(6, 25, 0.5))

    def load_density_field(self, redshift_index):
        n_cell = self.parameters.simulation.Ncell
        return np.zeros((n_cell, n_cell, n_cell))

    def load_rsd_fields(self, redshift_index):
        raise NotImplementedError("RSD fields are not implemented for FakeHaloLoader.")

    def load_halo_catalog(self, redshift_index):
        z = self.redshifts[redshift_index]
        masses = exponential_mass_accretion(z, M0 = self.mass_at_z6, z0=self.redshifts.min(), alpha=self.alpha)
        positions = np.stack([self.X, self.Y, self.Z], axis=-1)
        return HaloCatalog(
            positions, masses, self.parameters, redshift_index
        )
