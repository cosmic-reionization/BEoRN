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
    Input : nu in [Hz], T in [K]
    Returns : BB Spectrum [J.s-1.m−2.Hz−1]
    """
    a_ = 2.0 * h__ * nu**3 / c__**2
    intensity = 4 * np.pi * a_ / ( np.exp(h__*nu/(k__*T)) - 1.0)
    return intensity



def S_fct(Mh, Mt, g3, g4):
    """
    Suppression function in f_star. See eq.6 in arXiv:2305.15466.
    """
    return (1 + (Mt / Mh) ** g3) ** g4


def f_star_Halo(param,Mh):
    """
    fstar * Mh_dot * Ob/Om = Mstar_dot.
    fstar is therefore the conversion from baryon accretion rate  to star formation rate.
    See eq.(5) in arXiv:2305.15466.
    Double power law.
    """
    f_st = param.source.f_st
    Mp = param.source.Mp
    g1 = param.source.g1
    g2 = param.source.g2
    Mt = param.source.Mt
    g3 = param.source.g3
    g4 = param.source.g4
    fstar = np.minimum(2 * f_st / ((Mh / Mp) ** g1 + (Mh / Mp) ** g2) * S_fct(Mh, Mt, g3, g4),1)
    fstar[np.where(Mh < param.source.M_min)] = 0
    return fstar


def f_esc(param,Mh):
    f0  = param.source.f0_esc
    Mp  = param.source.Mp_esc
    pl  = param.source.pl_esc
    fesc = f0 * (Mp / Mh) ** pl
    return np.minimum(fesc,1)



def eps_xray(nu_,param):
    """
    Spectral distribution function of x-ray emission.
    In  [1/s/Hz*(yr*h/Msun)]
    Note : we include fX in cX in this code.
    See Eq.2 in arXiv:1406.4120
    """
    # param.source.cX  ## [erg / s /SFR]

    sed_xray = param.source.alS_xray
    norm_xray = (1 - sed_xray) / ((param.source.E_max_sed_xray / h_eV_sec) ** (1 - sed_xray) - (param.source.E_min_sed_xray / h_eV_sec) ** ( 1 - sed_xray)) ## [Hz**al-1]
   # param.source.cX * eV_per_erg * norm_xray * nu_ ** (-sed_xray) * Hz_per_eV   # [eV/eV/s/SFR]

    return param.source.cX/param.cosmo.h * eV_per_erg * norm_xray * nu_ ** (-sed_xray) /(nu_*h_eV_sec)   # [photons/Hz/s/SFR]


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




