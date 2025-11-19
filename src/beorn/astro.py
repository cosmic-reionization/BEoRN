"""
Contains various functions related to astrophysical sources.
"""
import numpy as np
from .structs import Parameters
from .constants import *



def f_star_Halo(parameters: Parameters, Mh):
    """
    Parameters
    ----------
    Mh : float. Halo mass in [Msol/h]
    param : Bunch

    Returns
    ----------
    fstar * Mh_dot * Ob/Om = Mstar_dot.
    fstar is therefore the conversion from baryon accretion rate  to star formation rate.
    See eq.(5) in arXiv:2305.15466.
    Double power law + either boost or cutoff at small scales (S_fct)
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
    """
    Parameters
    ----------
    Mh : float. Halo mass in [Msol/h]
    param : Bunch

    Returns
    ----------
    Escape fraction of ionising photons
    """

    f0  = parameters.source.f0_esc
    Mp  = parameters.source.Mp_esc
    pl  = parameters.source.pl_esc
    fesc = f0 * (Mp / Mh) ** pl
    return np.minimum(fesc,1)


def f_Xh(x_e):
    """
     Parameters
     ----------
     x_e : Free electron fraction in the neutral medium

     Returns
     ----------
     Fraction of X-ray energy deposited as heat in the IGM.
     Dimensionless. Various fitting functions exist in the literature
    """
    # Schull 1985 fit.
    # C,a,b = 0.9971, 0.2663, 1.3163
    # fXh = C * (1-(1-x_e**a)**b)

    fXh = x_e ** 0.225
    return fXh


def eps_xray(nu_, parameters: Parameters):
    """
    Parameters
    ----------
    nu_ : float. Photon frequency in [Hz]
    TODO: rename
    param : Bunch

    Returns
    ----------
    Spectral distribution function of x-ray emission in [1/s/Hz*(yr*h/Msun)]
    See Eq.2 in arXiv:1406.4120
    Note : fX is included in cX in this code.
    """

    # param.source.cX  is in [erg / s /SFR]

    sed_xray = parameters.source.alS_xray
    norm_xray = (1 - sed_xray) / ((parameters.source.energy_max_sed_xray / h_eV_sec) ** (1 - sed_xray) - (parameters.source.energy_min_sed_xray / h_eV_sec) ** ( 1 - sed_xray)) ## [Hz**al-1]
    # param.source.cX * eV_per_erg * norm_xray * nu_ ** (-sed_xray) * Hz_per_eV   # [eV/eV/s/SFR]

    return parameters.source.xray_normalisation / parameters.cosmology.h * eV_per_erg * norm_xray * nu_ ** (-sed_xray) /(nu_*h_eV_sec)   # [photons/Hz/s/SFR]
