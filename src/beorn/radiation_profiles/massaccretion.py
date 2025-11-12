"""
Mass Accretion Model
"""
import numpy as np
import logging
logger = logging.getLogger(__name__)

from ..cosmo import Hubble
from ..structs.parameters import Parameters


def mass_accretion(parameters: Parameters, z_bins: np.ndarray, m_bins: np.ndarray, alpha_bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the halo mass and its derivative with respect to time using an exponential mass accretion model. A range of initial halo masses and alpha values is computed.
    Args:
        param : dictionary containing all the input parameters
        z_bins  : arr of redshifts

    Returns:
        halo_mass and halo_mass_derivative
        These are two 3d arrays of shape (m_bins, alpha_bins, z_bins)
        halo_mass_derivative is in [Msol/h/yr]
    """
    z_initial = z_bins.min()
    logger.debug(f"Computing mass accretion for a parameter space consisting of: {m_bins.size=}, {alpha_bins.size=} and {z_bins.size=}")

    halo_mass = m_bins[:, None, None] * np.exp(alpha_bins[None, :, None] * (z_initial - z_bins[None, None, :]))
    halo_mass_derivative = mass_accretion_derivative(parameters, halo_mass, z_bins, m_bins, alpha_bins)

    return halo_mass, halo_mass_derivative


def mass_accretion_derivative(parameters: Parameters, halo_mass: np.ndarray, z_bins: np.ndarray, m_bins: np.ndarray, alpha_bins: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    param : dictionary containing all the input parameters
    Mh : arr. Halo masss
    z  : arr of redshifts, shape(Mh).

    Returns
    ----------
    Halo mass accretion rate, i.e. time derivative of halo mass (dMh/dt in [Msol/h/yr])
    """
    # by construction halo_mass has an alpha dependence and an initial mass dependence
    # using the function from above we can formulate an analytical expression for the derivative:
    # dMh/dt = Mh * alpha * H(z) * (z+1)
    return halo_mass * alpha_bins[None, :, None] * ((1 + z_bins) * Hubble(z_bins, parameters))[None, None, :]
