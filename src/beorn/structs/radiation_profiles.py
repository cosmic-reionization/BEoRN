from dataclasses import dataclass
import numpy as np

from .base_struct import BaseStruct


@dataclass(slots = True)
class RadiationProfiles(BaseStruct):
    """
    Flux profiles around star forming halos, computed for a range of halo masses and accretion rates (alpha). The central assumption is that each halo with given key properties (mass, alpha) produces the same radiation profile, meaning these profiles can be reused for multiple halos in the painting step of the simulation.
    """

    z_history: np.ndarray
    """redshift range for which the profiles have been computed. Corresponds to the parameters.solver.redshifts parameter"""

    halo_mass_bins: np.ndarray
    """bin edges of the halo masses that the profiles have been computed for. The radiation profile at index i corresponds to the halo mass range [halo_mass_bins[i], halo_mass_bins[i+1]]"""

    # the core profiles:
    rho_xray: np.ndarray
    """X-ray profile"""

    rho_heat: np.ndarray
    """heating profile, derived from the X-ray profile"""

    rho_alpha: np.ndarray
    """rho alpha profile"""

    R_bubble: np.ndarray
    """radius of the ionized bubble around the star forming halo"""
    r_lyal: np.ndarray
    """radius of the Lyman alpha halo"""

    # radial component of the profiles
    r_grid_cell: np.ndarray
    """radial grid of the profiles"""


    def profiles_of_halo_bin(self, z_index: int, alpha_index: slice, mass_index:slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the three core profiles for a halo bin.

        Args:
            z_index (int): Redshift index.
            alpha_index (int or slice): Alpha (accretion) index or slice.
            mass_index (int or slice): Mass bin index or slice.

        Returns:
            tuple: ``(R_bubble, rho_alpha, rho_heat)`` arrays copied from the
            stored profiles for the requested bin.
        """
        return(
            self.R_bubble[mass_index, alpha_index, z_index].copy(),
            self.rho_alpha[:, mass_index, alpha_index, z_index].copy(),
            self.rho_heat[:, mass_index, alpha_index, z_index].copy(),
        )


    def __post_init__(self):
        BaseStruct.__post_init__(self)
        assert self.z_history.ndim == 1, "z_history must be a 1D array"
        assert self.r_grid_cell.ndim == 1, "r_grid_cell must be a 1D array"

        # nan value in any of the profiles indicates a miscalculation
        assert np.all(np.isfinite(self.rho_xray)), "rho_xray contains invalid values"
        assert np.all(np.isfinite(self.rho_heat)), "rho_heat contains invalid values"
        assert np.all(np.isfinite(self.rho_alpha)), "rho_alpha contains invalid values"
        assert np.all(np.isfinite(self.R_bubble)), "R_bubble contains invalid values"
        assert np.all(np.isfinite(self.r_lyal)), "r_lyal contains invalid values"
