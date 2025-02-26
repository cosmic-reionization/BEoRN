"""
Contains various functions related to astrophysical sources.
"""

import numpy as np
from .constants import *
import pickle
from .cosmo import Hubble
from .functions import *



def BB_Planck(nu, T):
    """
    Parameters
    ----------
    nu : float. Photon frequency in [Hz]
    T : float. Black-Body temperature in [K]

    Returns
    ----------
    Black-Body spectrum (Planck's law) in [J.s-1.m−2.Hz−1]
    """

    a_ = 2.0 * h__ * nu**3 / c__**2
    intensity = 4 * np.pi * a_ / ( np.exp(h__*nu/(k__*T)) - 1.0)
    return intensity



def S_fct(Mh, Mt, g3, g4):
    """
    Parameters
    ----------
    Mh : float. Halo mass in [Msol/h]
    Mt : float. Cutoff mass in [Msol/h]
    g3,g4 : floats. Control the power-law behavior of the fct.

    Returns
    ----------
    Small-scale part of the stellar-to-halo function f_star. See eq.6 in arXiv:2305.15466.
    (g3,g4) = (1,1),(0,0),(4,-4) gives a boost, power-law, cutoff of SFE at small scales, respectively.
    """

    return (1 + (Mt / Mh) ** g3) ** g4


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


def f_esc(param,Mh):
    """
    Parameters
    ----------
    Mh : float. Halo mass in [Msol/h]
    param : Bunch

    Returns
    ----------
    Escape fraction of ionising photons
    """

    f0  = param.source.f0_esc
    Mp  = param.source.Mp_esc
    pl  = param.source.pl_esc
    fesc = f0 * (Mp / Mh) ** pl
    return np.minimum(fesc,1)


def f_Xh(param,x_e):
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


def Ng_dot_Snapshot(param,rock_catalog, type ='xray'):
    """
    WORKS FOR EXP MAR
    Mean number of ionising photons emitted per sec for a given rockstar snapshot. [s**-1.(cMpc/h)**-3]
    Or  mean Xray energy over the box [erg.s**-1.Mpc/h**-3]
    rock_catalog : rockstar halo catalog
    """
    Halos = Read_Rockstar(rock_catalog,Nmin = param.sim.Nh_part_min)
    H_Masses, z = Halos['M'], Halos['z']
    dMh_dt = param.source.alpha_MAR * H_Masses * (z+1) * Hubble(z, param) ## [(Msol/h) / yr]
    dNg_dt = dMh_dt * f_star_Halo(param, H_Masses) * param.cosmo.Ob/param.cosmo.Om * f_esc(param, H_Masses) * param.source.Nion /sec_per_year /m_H * M_sun  #[s**-1]

    if type =='ion':
        return z, np.sum(dNg_dt) / Halos['Lbox'] ** 3 #[s**-1.(cMpc/h)**-3]

    if type == 'xray':
        sed_xray = param.source.alS_xray
        norm_xray = (1 - sed_xray) / ((param.source.E_max_sed_xray / h_eV_sec) ** (1 - sed_xray) - (param.source.E_min_sed_xray / h_eV_sec) ** (1 - sed_xray))
        E_dot_xray = dMh_dt * f_star_Halo(param, H_Masses) * param.cosmo.Ob / param.cosmo.Om * param.source.cX/param.cosmo.h  ## [erg / s]

        nu_range = np.logspace(np.log10(param.source.E_min_xray / h_eV_sec),np.log10(param.source.E_max_sed_xray / h_eV_sec), 3000, base=10)
        Lumi_xray  = eV_per_erg * norm_xray * nu_range ** (-sed_xray) * Hz_per_eV  # [eV/eV/s]/E_dot_xray
        Ngdot_sed = Lumi_xray / (nu_range * h_eV_sec)  # [photons/eV/s]/E_dot_xray
        Ngdot_xray = np.trapz(Ngdot_sed,nu_range * h_eV_sec)*E_dot_xray  # [photons/s]

        return z, np.sum(E_dot_xray) / Halos['Lbox'] ** 3,   np.sum(Ngdot_xray)/ Halos['Lbox'] ** 3     # [erg.s**-1.Mpc/h**-3], [photons.s-1]




