"""
Here we write all the functions needed for the solver describing interactions, such as cross sections, cooling and collisional coupling coefficients etc...
"""

import numpy as np
from .constants import *


def sigma_HI(E):
    """
    Parameters
    ----------
    E : Photon energy, in [eV]

    Returns
    ----------
    Photo-ionization cross section for Hydrogen,  [cm ** 2]
    Taken from  Verner et al. arXiv:astro-ph/9601009
    """
    sigma_0 = 5.475 * 10 ** 4 * 10 ** -18 ## cm**2 1Mbarn = 1e-18 cm**2
    E_01 = 4.298 * 10 ** -1
    y_a = 3.288 * 10 ** 1
    P = 2.963
    y_w = y_0 = y_1 = 0
    x = E / E_01 - y_0
    y = np.sqrt(x ** 2 + y_1 ** 2)
    F = ((x - 1) ** 2 + y_w ** 2) * y ** (0.5 * P - 5.5) * (1 + np.sqrt(y / y_a)) ** -P
    sigma = sigma_0 * F
    return sigma

def sigma_HeI(E):
    """
    Parameters
    ----------
    E : Photon energy, in [eV]

    Returns
    ----------
    Photo-ionization cross section for Helium HeI,  [cm ** 2]
    Taken from  Verner et al. arXiv:astro-ph/9601009
    """
    sigma_0 = 9.492 * 10 ** 2 * 10 ** -18
    E_01 = 1.361 * 10 ** 1
    y_a = 1.469
    P = 3.188
    y_w = 2.039
    y_0 = 4.434 * 10 ** -1
    y_1 = 2.136
    x = E / E_01 - y_0
    y = np.sqrt(x ** 2 + y_1 ** 2)
    F = ((x - 1) ** 2 + y_w ** 2) * y ** (0.5 * P - 5.5) * (1 + np.sqrt(y / y_a)) ** -P
    sigma = sigma_0 * F
    return sigma

def sigma_HeII(E):
    """
    Parameters
    ----------
    E : Photon energy, in [eV]

    Returns
    ----------
    Photo-ionization cross section for singly ionized Helium HeII,  [cm ** 2]
    Taken from  Verner et al. arXiv:astro-ph/9601009
    """
    sigma_0 = 1.369 * 10 ** 4 * 10 ** -18  # cm**2
    E_01 = 1.72
    y_a = 3.288 * 10 ** 1
    P = 2.963
    y_w = 0
    y_0 = 0
    y_1 = 0
    x = E / E_01 - y_0
    y = np.sqrt(x ** 2 + y_1 ** 2)
    F = ((x - 1) ** 2 + y_w ** 2) * y ** (0.5 * P - 5.5) * (1 + np.sqrt(y / y_a)) ** -P
    sigma = sigma_0 * F
    return sigma


def alpha_HII(T):
    """
    Parameters
    ----------
    T : temperature in K

    Returns
    ----------
    Case B recombination coefficient for Hydrogen :  [cm3.s-1]
    Taken from Fukugita and Kawasaki 1994
    """
    return 2.6 * 10 ** -13 * (T / 10 ** 4) ** -0.85
