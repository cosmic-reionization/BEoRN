"""
Mass Accretion Model
"""
import numpy as np
import logging
logger = logging.getLogger(__name__)

from ..cosmo import Hubble
from ..structs.parameters import Parameters


def mass_accretion(parameters: Parameters, z_bins: np.ndarray, m_bins: np.ndarray, alpha_bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute halo mass evolution and its time derivative.

    Uses an exponential accretion model where the mass grows as
    M(z) = M0 * exp(alpha * (z_initial - z)). The function returns
    both the halo mass evolution and the mass accretion rate dM/dt for
    the provided grid of initial masses and alpha values.

    Args:
        parameters (Parameters): Simulation parameters used for cosmology.
        z_bins (np.ndarray): Redshift grid (decreasing order preferred).
        m_bins (np.ndarray): Initial halo masses (bin edges or centers).
        alpha_bins (np.ndarray): Exponential growth parameters for each alpha bin.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: ``(halo_mass, halo_mass_derivative)``
        where both arrays have shape ``(m_bins, alpha_bins, len(z_bins))``.
        The derivative is given in physical units consistent with the
        rest of the code (approximately M_sun/h per time unit).
    """
    z_initial = z_bins.min()
    logger.debug(f"Computing mass accretion for a parameter space consisting of: {m_bins.size=}, {alpha_bins.size=} and {z_bins.size=}")

    halo_mass = m_bins[:, None, None] * np.exp(alpha_bins[None, :, None] * (z_initial - z_bins[None, None, :]))
    halo_mass_derivative = mass_accretion_derivative(parameters, halo_mass, z_bins, m_bins, alpha_bins)

    return halo_mass, halo_mass_derivative


def mass_accretion_derivative(parameters: Parameters, halo_mass: np.ndarray, z_bins: np.ndarray, m_bins: np.ndarray, alpha_bins: np.ndarray) -> np.ndarray:
    """Compute the time derivative dM/dt of the halo mass evolution.

    The derivative follows analytically from the exponential model
    used in :func:`mass_accretion` and includes the Hubble-factor
    conversion via :func:`Hubble`.

    Args:
        parameters (Parameters): Simulation cosmology parameters.
        halo_mass (numpy.ndarray): Halo mass array as returned by :func:`mass_accretion`.
        z_bins (numpy.ndarray): Redshift grid.
        m_bins (numpy.ndarray): Initial mass grid (not directly used but provided for clarity).
        alpha_bins (numpy.ndarray): Alpha growth parameters.

    Returns:
        numpy.ndarray: dM/dt array with same shape as ``halo_mass``.
    """
    # by construction halo_mass has an alpha dependence and an initial mass dependence
    # using the function from above we can formulate an analytical expression for the derivative:
    # dMh/dt = Mh * alpha * H(z) * (z+1)
    return halo_mass * alpha_bins[None, :, None] * ((1 + z_bins) * Hubble(z_bins, parameters))[None, None, :]
