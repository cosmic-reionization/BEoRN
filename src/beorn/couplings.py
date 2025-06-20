"""
Here we compute the Lyman_alpha and collisional coupling coefficient (x_al and x_coll).
"""

import numpy as np
import importlib
from pathlib import Path

from scipy.interpolate import splrep,splev
from .constants import *
from .cosmo import T_cmb
from .parameters import Parameters


def kappa_coll():
    """
    Eq.10 in arXiv:1109.6012
    Used only in x_coll.
    Reads in tables for the scattering rates between H-H and H-e.

    Parameters
    ----------
    None

    Returns
    ----------
    Rate coefficient for spin de-excitation in collisions with H and e [cm^3/s]
    """

    names = 'T, kappa'
    path_to_file = Path(importlib.util.find_spec('beorn').origin).parent / 'input_data' / 'kappa_eH.dat'
    eH = np.genfromtxt(path_to_file, usecols=(0, 1), comments='#', dtype=float, names=names)

    names = 'T, kappa'
    path_to_file = Path(importlib.util.find_spec('beorn').origin).parent / 'input_data' / 'kappa_HH.dat'
    HH = np.genfromtxt(path_to_file, usecols=(0, 1), comments='#', dtype=float, names=names)

    return HH, eH


def x_coll(z, Tk, xHI, rho_b):
    """
    Parameters
    ----------
    z     : redshift
    Tk    : Gas kinetic temperature profile [K]
    xHI   : Hydrogen neutral fraction
    rho_b : baryon density in nbr of [H atoms /pcm**3] (physical cm)

    Returns
    ----------
    Collisional coupling coefficient. [Dimensionless]
    """
    # nH and e- densities
    n_HI  = rho_b * xHI
    n_HII = rho_b * (1-xHI) # [1/cm^3]

    # prefac (Eq.10 in arXiv:1109.6012)
    Tcmb = T_cmb(z)
    prefac = Tstar / A10 / Tcmb  # [s]

    HH, eH = kappa_coll()
    kappa_eH_tck = splrep(eH['T'], eH['kappa'])
    kappa_eH = splev(Tk, kappa_eH_tck, ext=3)  # [cm^3/s]
    kappa_HH_tck = splrep(HH['T'], HH['kappa'])
    kappa_HH = splev(Tk, kappa_HH_tck, ext=3)

    x_HH = prefac * kappa_HH * n_HI
    x_eH = prefac * kappa_eH * n_HII
    return x_HH  + x_eH

def x_coll_coef(z,param):
    """
    Coefficient to turn rho/rho_mean into a baryon density in nbr of H atoms per physical cm**3 [1/pcm**3]
    """
    Om, Ob, h0 = param.cosmo.Om, param.cosmo.Ob, param.cosmo.h
    coef = rhoc0 * h0 ** 2 * Ob * (1 + z) ** 3 * M_sun / cm_per_Mpc ** 3 / m_H
    return coef


def S_alpha(z, Tk, xHI):
    """
    Parameters
    ----------
    z     : redshift
    Tk    : Gas kinetic temperature [K]
    xHI   : Hydrogen neutral fraction

    Returns
    ----------
    Suppression factor S_alpha. [dDmensionless]
    Following method in astro-ph/0608032
    """
    # Eq.43
    tau_GP = 3.0e5 * xHI * ((1 + z) / 7.0) ** 1.5
    # gamma = 1 / tau_GP

    # Eq. 55
    S_al = np.exp(-0.803 * Tk ** (-2 / 3) * (1e-6 * tau_GP) ** (1 / 3))

    return S_al


def eps_lyal(nu, parameters: Parameters):
    """
    Lyman-a spectral energy distribution (power-law). See eq.8 in BEoRN paper.

    Parameters
    ----------
    nu : Frequency [Hz].
    param : BEoRN dictionnary containing model parameters.

    Returns
    -------
    float. [photons.yr-1.Hz-1.SFR-1], SFR being the Star Formation Rate in Msol/h/yr
    """

    h0    = parameters.cosmology.h
    N_al  = parameters.source.n_lyman_alpha_photons  #9690 number of lya photons per protons (baryons) in stars
    alS = parameters.source.lyman_alpha_power_law

    nu_min_norm  = nu_al
    nu_max_norm  = nu_LL

    Anorm = (1-alS)/(nu_max_norm**(1-alS) - nu_min_norm**(1-alS))
    Inu   = lambda nu: Anorm * nu**(-alS)

    eps_alpha = Inu(nu)*N_al/(m_p_in_Msun * h0)

    return eps_alpha






####### Not used below this line

def phi_alpha(x,E):
    """
    Fraction of the absorbed photon energy that goes into excitation. [Dimensionless]
    From Dijkstra, Haiman, Loeb. Apj 2004.

    Parameters
    ----------
    x : ionized hydrogen fraction at location
    E : energy in eV

    Returns
    -------
    float
    """
    return 0.39*(1-x**(0.4092*a_alpha(x,E)))**1.7592

def a_alpha(x,E):
    """
    Used in phi_alpha.
    """
    return 2/np.pi * np.arctan(E/120 * (0.03/x**1.5 + 1)**0.25)








