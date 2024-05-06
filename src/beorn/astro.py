"""
Contains various functions related to astrophysical sources.
"""

import numpy as np
from .constants import *
import pickle
from .cosmo import Hubble
from .functions import *

from .profiles_on_grid import sigmoid_fct

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


def f_star_Halo(param,Mh):
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


def f_esc(param,Mh,zz=None):
    """
    Parameters
    ----------
    Mh : float. Halo mass in [Msol/h]
    param : Bunch

    Returns
    ----------
    Escape fraction of ionising photons
    """

    f0 = param.source.f0_esc
    Mp = param.source.Mp_esc
    pl = param.source.pl_esc

    if param.source.f_esc_type == 'cst' or param.source.f_esc_type == 'Licorice_bis':
        fesc = f0 * (Mp / Mh) ** pl ## shape is (Mh)

    elif param.source.f_esc_type == 'Licorice':
        ### This is to match Licorice (in Lic)
        ### In Licorice, if xHII<3% fesc=0.003, and if xHII>3% fesc=0.275 (in each cell)
        ### We cannot imitate this in BEoRN, so we chose to have a z-dependent global f_esc
        ### If this is chosen, the normalisation of f_esc becomes time dependent.
        ### It follows a sigmoid, from 0.003 to param.source.f0_esc

        sigm_ = sigmoid_fct(zz[:,0], c1=3.6, c2=param.source.z_thresh_f_esc) ## fitted to frac_pixels_fesc_high computed from Licorice Boxes
        f0 = sigm_ * f0 + (1-sigm_)*0.003
        #f0[zz[:,0]>param.source.z_thresh_f_esc] = 0.003
        fesc = (f0[:,None] * (Mp / Mh) ** pl)   ##(shape is zz,Mh)

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
    if param.sim.licorice: # Schull 1985 fit.
        C,a,b = 0.9971, 0.2663, 1.3163
        fXh = C * (1-(1-x_e**a)**b)
    else :
        fXh = x_e ** 0.225
    return fXh


def eps_xray(nu,param):
    """
    Parameters
    ----------
    nu : float. Photon frequency in [Hz]
    param : Bunch

    Returns
    ----------
    Spectral distribution function of x-ray emission in [1/s/Hz*(yr*h/Msun)]
    See Eq.2 in arXiv:1406.4120
    Note : fX is included in cX in this code.
    """
    # param.source.cX  is in [erg / s /SFR]
    # pl_xray = param.source.alS_xray
    # norm_xray = (1 - pl_xray) / ((param.source.E_max_sed_xray / h_eV_sec) ** (1 - pl_xray) - (param.source.E_min_sed_xray / h_eV_sec) ** ( 1 - pl_xray)) ## [Hz**al-1]
    # param.source.cX * eV_per_erg * norm_xray * nu_ ** (-sed_xray) * Hz_per_eV   # [eV/eV/s/SFR]

    sed_xray_ = sed_xray(nu,param) # [Hz^-1]
    eps_xray_ = param.source.cX/param.cosmo.h * eV_per_erg * sed_xray_/(nu*h_eV_sec)   # [photons/Hz/s/SFR]

    return eps_xray_


def sed_xray(nu_,param):
    """
    Parameters
    ----------
    nu_ : float. Photon frequency in [Hz]
    param : Bunch

    Returns
    ----------
    Spectral distribution function of x-ray emission in [Hz^-1]
    """

    if param.source.xray_type == 'PL':
      #  print('Only one x-ray source population is chosen.')
        pl_xray = param.source.alS_xray
        norm_xray = (1 - pl_xray) / ((param.source.E_max_sed_xray / h_eV_sec) ** (1 - pl_xray) - ( param.source.E_min_sed_xray / h_eV_sec) ** (1 - pl_xray))  ## [Hz**al-1]
        sed_xray_ = norm_xray * nu_ ** (-pl_xray)  # [Hz^-1]

    elif param.source.xray_type == 'Licorice':
        # print('Licorice Source Model. See Semelin 2017. Adding two x-ray source populations.')
        # IN DEVELOPMENT -----------
        # we renormalize sed_AGN to 1 in 100-2000 eV

        nu_renorm = np.logspace(np.log10(param.source.E_min_sed_xray/h_eV_sec), np.log10(param.source.E_max_sed_xray/h_eV_sec), 1000, base=10)

        ## The AGN x-ray spectrum
        Emin_AGN = 100
        Emax_AGN = 2000
        pl_xray = param.source.alS_xray
        norm_xray = (1 - pl_xray) / ((Emax_AGN/ h_eV_sec) ** (1 - pl_xray) - (Emin_AGN / h_eV_sec) ** (1 - pl_xray))  ## [Hz**al-1]
        sed_AGN = norm_xray * nu_renorm ** (-pl_xray)  # [Hz^-1]

        ## reading in the normalized (to 1) sed from Fragos 2013 for XRB
        energy, sed_XRB = np.loadtxt(param.source.sed_XRB)    #eV, eV^-1

        sed_XRB = sed_XRB/np.trapz(sed_XRB, energy/h_eV_sec) # Hz^-1
        sed_XRB = np.interp(nu_renorm, energy/h_eV_sec, sed_XRB)
        #renormalizing to be sure

        sed_xray_ = param.source.fX_AGN * sed_AGN + (1-param.source.fX_AGN) * sed_XRB

        #renormalizing the xray SED
        sed_xray_ = sed_xray_/ np.trapz(sed_xray_, nu_renorm) #sed_xray_ / np.trapz(sed_xray_, nu_renorm)
        sed_xray_ = np.interp(nu_,nu_renorm,sed_xray_,left=0,right=0)

    return  sed_xray_



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




def N_ion_(param,zz):
    """
    Number of ionising photons per baryons in stars.
    We added a possible redshift dependence for comparison with Licorice.
    """
    if param.source.Nion_z is None:
        return param.source.Nion
    else :
        print(' param.sim.Nion_z is not None. It should be given as a 2D array (zz,Nion(zz)), with z in increasing order')
        return np.interp(zz,param.source.Nion_z[0],param.source.Nion_z[1])


def f_X_(param,zz):
    """
    X-ray efficiency coefficient. Degenerate with c_X, but for comparison with RT code Licorice, 
    we added a redshift dependence of fX. fX=1 is default.
    """
    if isinstance(param.source.fX,float) or isinstance(param.source.fX,int):
        print('param.source.fX is a number.')
        return param.source.fX
    else :
        print('param.source.fX is NOT a number. It should be given as a 2D array (zz,Nion(zz)), with z in increasing order')
        return np.interp(zz,param.source.fX[0],param.source.fX[1])
