"""Halo catalog data structure."""
from dataclasses import dataclass
import numpy as np
import logging
logger = logging.getLogger(__name__)

from .parameters import Parameters
from ..particle_mapping.pylians import map_particles_to_mesh


@dataclass
class HaloCatalog:
    """Container for halo positions, masses and associated properties.

    Instances represent all halos at a single given redshift and provide
    filtering and utility methods used by the painting stage.
    """

    positions: np.ndarray
    """Halo positions in 3D space (X, Y, Z coordinates) in units of cMpc => shape=(N, 3)"""

    masses: np.ndarray
    """Halo masses in units of Msun => shape=(N,)"""

    parameters: Parameters
    """The parameters of the simulation, which are used to filter the halo catalog."""

    redshift_index: int = 0
    """The index of the redshift snapshot that this catalog corresponds to. This is used to look up accretion history"""

    alphas: np.ndarray = None
    """
    Halo mass accretion rate, calculated from the mass history of the halo. If not available, it is set to a default value of 0.79.
    Inputs that provide a mass history will override this value to have the same shape as the masses array.
    """

    def __post_init__(self):
        self.masses = np.asarray(self.masses)
        assert self.positions.shape[0] == self.masses.size, "Halo catalog arrays must have the same length."

        if self.alphas is None:
            logger.info("No alpha values provided, using default value of 0.79 for all halos.")
            self.alphas = np.ones(self.masses.size) * 0.79

        # filter out haloes that are considerd non-star forming
        condition_min = self.masses > self.parameters.source.halo_mass_min
        condition_max = self.masses < self.parameters.source.halo_mass_max
        # logger.debug(f"Removing {np.sum(~(condition_min & condition_max))} halos that are outside the parameter mass range")
        self.positions = self.positions[condition_min & condition_max, :]
        self.masses = self.masses[condition_min & condition_max]
        self.alphas = self.alphas[condition_min & condition_max]


    @property
    def size(self) -> int:
        """
        Returns the number of halos in the catalog.
        """
        return self.masses.size


    def get_halo_indices(self, alpha_range: list[float], mass_range: list[float]) -> np.ndarray:
        """Return indices of halos within provided alpha and mass ranges.

        Args:
            alpha_range (list[float]): [alpha_min, alpha_max).
            mass_range (list[float]): [mass_min, mass_max).

        Returns:
            numpy.ndarray: Integer indices of matching halos (may be empty).
        """
        if self.masses.size == 0:
            return []

        alpha_inf, alpha_sup = alpha_range
        mass_inf, mass_sup = mass_range

        # Get the indices of the halos that are within the mass and alpha range
        indices_match = np.where(
            (self.masses >= mass_inf) & (self.masses < mass_sup) &
            (self.alphas >= alpha_inf) & (self.alphas < alpha_sup)
        )[0]
        # in this case where returns two arrays, we only want the first one

        # if indices_match.size != 0:
        #     logger.debug(f"{alpha_range=} and {mass_range=} resulted in matches: {indices_match.size}")
        return indices_match


    def to_mesh(self) -> np.ndarray:
        """Rasterize halo positions into a 3D number-count mesh.

        The mesh uses the nearest-neighbor mass-assignment scheme. The returned array represents halo counts
        (number density), not mass.

        Returns:
            numpy.ndarray: 3D float32 array with shape (Ncell, Ncell, Ncell).
        """
        physical_size = self.parameters.simulation.Lbox
        grid_size = self.parameters.simulation.Ncell
        mesh = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        map_particles_to_mesh(mesh, physical_size, self.positions.astype(np.float32), "NGP")
        return mesh



    ### methods that create new halo catalogs
    def at_indices(self, indices) -> "HaloCatalog":
        """Return a new :class:`HaloCatalog` restricted to the given indices.

        Args:
            indices (array-like): Indices selecting a subset of halos.

        Returns:
            HaloCatalog: New instance containing only the selected halos.
        """
        if indices.size == 0:
            indices = []

        return HaloCatalog(
            positions = self.positions[indices, :],
            masses = self.masses[indices],
            parameters = self.parameters,
            redshift_index = self.redshift_index,
            # at that point self.alphas is guaranteed to exist since __post_init__ was called
            alphas = self.alphas[indices]
        )


    def halo_mass_function(self, bin_count: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the halo mass function (dn/dlnM) and Poisson errors.

        Args:
            bin_count (int|None): Number of mass bins. If ``None`` a
                default value proportional to the mass dynamic range is used.

        Returns:
            tuple: ``(bin_edges, hmf, error)`` where ``hmf`` is in
            units of (Mpc/h)^-3 and ``error`` is the Poisson uncertainty.
        """
        Lbox = self.parameters.simulation.Lbox

        if bin_count is None:
            bin_count = int(10 * np.log10(self.masses.max() / self.masses.min()))
        # increase the range a bit to capture all halos
        bin_edges = np.logspace(np.log(self.masses.min() * 0.9), np.log(self.masses.max() * 1.1), bin_count + 1, base=np.e)
        log_spacing = np.log(bin_edges[1]) - np.log(bin_edges[0])
        # digitize the masses into the bins
        indices = np.digitize(self.masses, bin_edges) - 1
        # -1 because digitize returns the index of the bin to the right of the value
        indices, count = np.unique(indices, return_counts=True)
        # now we have the counts in each bin, but only for the bins that have at least one halo
        full_count = np.zeros(bin_count, dtype=int)
        full_count[indices] = count

        # for the halo mass function we want dn / d ln(M) and then the normalization by the box volume
        hmf = full_count / Lbox**3 / log_spacing

        # for the error we assume Poisson statistics
        error = hmf / np.sqrt(full_count)
        assert hmf.size == bin_count, f"{hmf.size=} does not match expected {bin_count=}"
        assert error.size == bin_count, f"{error.size=} does not match expected {bin_count=}"
        return bin_edges, hmf, error
