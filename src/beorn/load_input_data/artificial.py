import numpy as np

from .base import BaseLoader
from ..structs import HaloCatalog
from ..structs import Parameters

def exponential_mass_accretion(z, M0, z0, alpha):
    """Exponential mass accretion model used by the artificial loader.

    Args:
        z (float|array): Redshift(s) at which to evaluate the model.
        M0 (float|array): Reference mass at redshift ``z0``.
        z0 (float): Reference redshift.
        alpha (float): Exponential growth parameter.

    Returns:
        float|array: Mass(es) at redshift ``z``.
    """
    return M0 * np.exp(-alpha * (z - z0))


class ArtificialHaloLoader(BaseLoader):
    """Simple synthetic loader for quick testing.

    The loader places a fixed number of halos at random positions in
    the box and returns masses that follow a simple exponential mass
    evolution. Intended for tests and examples.
    """

    def __init__(self, parameters: Parameters, halo_count: int, seed: int = 12345, final_mass: float = 1e12, alpha: float = 0.79):
        """Create an artificial loader.

        Args:
            parameters (Parameters): Simulation parameters.
            halo_count (int): Number of synthetic halos to create.
            seed (int, optional): RNG seed for reproducible positions.
            final_mass (float, optional): Final mass at the reference redshift used as M0 in the accretion model.
            alpha (float, optional): Exponential accretion parameter.
        """
        super().__init__(parameters)
        rng = np.random.default_rng(seed)
        self.X = rng.random(halo_count) * self.parameters.simulation.Lbox
        self.Y = rng.random(halo_count) * self.parameters.simulation.Lbox
        self.Z = rng.random(halo_count) * self.parameters.simulation.Lbox

        self.mass_at_z0 = np.full(halo_count, final_mass)
        self.alpha = alpha

    @property
    def redshifts(self):
        """Return the redshift grid used by this loader.

        The grid is returned in ascending order (current -> past).
        """
        return np.flip(np.arange(6, 25, 0.5))

    def load_density_field(self, redshift_index):
        """Return a zero density field (placeholder for tests).

        Returns an array of zeros with shape ``(Ncell, Ncell, Ncell)``.
        """
        n_cell = self.parameters.simulation.Ncell
        return np.zeros((n_cell, n_cell, n_cell))

    def load_rsd_fields(self, redshift_index):
        """Not implemented for the artificial loader.

        Raises:
            NotImplementedError: RSD fields are not provided.
        """
        raise NotImplementedError("RSD fields are not implemented for FakeHaloLoader.")

    def load_halo_catalog(self, redshift_index):
        """Return a :class:`HaloCatalog` with synthetic positions and masses.

        Args:
            redshift_index (int): Index of the snapshot to return.

        Returns:
            HaloCatalog: Catalog containing positions and masses.
        """
        z = self.redshifts[redshift_index]
        masses = exponential_mass_accretion(z, M0 = self.mass_at_z0, z0=self.redshifts.min(), alpha=self.alpha)
        positions = np.stack([self.X, self.Y, self.Z], axis=-1)
        return HaloCatalog(
            positions, masses, self.parameters, redshift_index
        )
