"""beorn.astro
=================

Utilities related to astrophysical source modelling used by BEoRN.

This module provides simple, vectorised functions for star-formation
efficiencies, escape fractions, X-ray spectral shapes and related
convenience helpers.
"""
import numpy as np
from .structs import Parameters
from .constants import *


def S_fct(Mh, Mt, g3, g4):
    """Small-scale modifier of the stellar-to-halo efficiency.

    Args:
        Mh (float or array): Halo mass in Msol/h.
        Mt (float): Cutoff mass in Msol/h.
        g3 (float): Control parameter for the small-mass power law.
        g4 (float): Exponent controlling the strength of the modifier.

    Returns:
        float or array: Modifier factor applied to the stellar-to-halo
            efficiency. Typical choices of ``(g3,g4)`` produce a boost,
            power-law behaviour or a cutoff at small scales (see
            arXiv:2305.15466, Eq. 6).
    """

    return (1 + (Mt / Mh) ** g3) ** g4


def f_star_Halo(parameters: Parameters, Mh):
    """Compute the stellar fraction for haloes of a given mass.

    The returned quantity corresponds to the star-formation efficiency
    (fraction of accreted baryons turned into stars) used in the
    BEoRN model. The function applies a double power-law in mass with
    an additional small-scale modifier provided by :func:`S_fct`.

    Args:
        parameters (Parameters): Model parameters container.
        Mh (float or array): Halo mass in Msol/h.

    Returns:
        float or array: Stellar fraction in the range [0, 1].

    Notes:
        See Eq. (5) in arXiv:2305.15466 for the full definition.
    """

    f_st = parameters.source.f_st
    Mp = parameters.source.Mp
    g1 = parameters.source.g1
    g2 = parameters.source.g2
    Mt = parameters.source.Mt
    g3 = parameters.source.g3
    g4 = parameters.source.g4
    fstar = np.minimum(2 * f_st / ((Mh / Mp) ** g1 + (Mh / Mp) ** g2) * S_fct(Mh, Mt, g3, g4),1)
    fstar[np.where(Mh < parameters.source.halo_mass_min)] = 0
    return fstar


def f_esc(parameters: Parameters, Mh):
    """Escape fraction of ionising photons as a function of halo mass.

    Args:
        parameters (Parameters): Model parameters container.
        Mh (float or array): Halo mass in Msol/h.

    Returns:
        float or array: Escape fraction (clipped to a maximum of 1).
    """

    f0  = parameters.source.f0_esc
    Mp  = parameters.source.Mp_esc
    pl  = parameters.source.pl_esc
    fesc = f0 * (Mp / Mh) ** pl
    return np.minimum(fesc,1)


def f_Xh(x_e):
    """Fraction of X-ray energy deposited as heat in the IGM.

    Args:
        x_e (float): Free-electron fraction in the neutral medium.

    Returns:
        float: Fraction of X-ray energy deposited as heat (unitless).
    """
    # Schull 1985 fit.
    # C,a,b = 0.9971, 0.2663, 1.3163
    # fXh = C * (1-(1-x_e**a)**b)

    fXh = x_e ** 0.225
    return fXh


def eps_xray(nu_, parameters: Parameters):
    """Spectral distribution of X-ray emission per unit star-formation.

    The spectral energy distribution is assumed to be a power law in
    frequency with a normalisation set by the parameter container. The
    returned units follow the convention used in BEoRN (photons / Hz / s / SFR).

    Args:
        nu_ (float or array): Photon frequency in Hz.
        parameters (Parameters): Model parameters container.

    Returns:
        float or array: Spectral photon emission rate per unit SFR.

    Notes:
        The implementation follows Eq. 2 of arXiv:1406.4120.
    """

    # param.source.cX  is in [erg / s /SFR]

    sed_xray = parameters.source.alS_xray
    norm_xray = (1 - sed_xray) / ((parameters.source.energy_max_sed_xray / h_eV_sec) ** (1 - sed_xray) - (parameters.source.energy_min_sed_xray / h_eV_sec) ** ( 1 - sed_xray)) ## [Hz**al-1]
    # param.source.cX * eV_per_erg * norm_xray * nu_ ** (-sed_xray) * Hz_per_eV   # [eV/eV/s/SFR]

    return parameters.source.xray_normalisation / parameters.cosmology.h * eV_per_erg * norm_xray * nu_ ** (-sed_xray) /(nu_*h_eV_sec)   # [photons/Hz/s/SFR]
