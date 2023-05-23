"""
Contains avrious functions related to astrophysical sources.
"""

import numpy as np
from .constants import *
import pickle
from .cosmo import Hubble
from .functions import *



def BB_Planck( nu , T):
    """
    Input : nu in [Hz], T in [K]
    Returns : BB Spectrum [J.s-1.m−2.Hz−1]
    """
    a_ = 2.0 * h__ * nu**3 / c__**2
    intensity = 4 * np.pi * a_ / ( np.exp(h__*nu/(k__*T)) - 1.0)
    return intensity



def S_fct(Mh, Mt, g3, g4):
    return (1 + (Mt / Mh) ** g3) ** g4


def f_star_Halo(param,Mh):
    """
    Double power law. fstar * Mh_dot * Ob/Om = M*_dot. fstar is therefore the conversion from baryon accretion rate  to star formation rate.
    """
    f_st = param.source.f_st
    Mp = param.source.Mp
    g1 = param.source.g1
    g2 = param.source.g2
    Mt = param.source.Mt
    g3 = param.source.g3
    g4 = param.source.g4
    return np.minimum(2 * f_st / ((Mh / Mp) ** g1 + (Mh / Mp) ** g2) * S_fct(Mh, Mt, g3, g4),1)


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



def Read_Rockstar(file,Nmin = 10,Mmin = 1e5,Mmax = 1e15 ,keep_subhalos=True):
    """
    Read in a rockstar halo catalog and return a dictionnary with all the information stored.
    R is in ckpc/h
    Nmin : not working yet
    """

    Halo_File = []
    with open(file) as f:
        for line in f:
            Halo_File.append(line)
    a = float(Halo_File[1][4:])
    z = 1 / a - 1
    LBox = float(Halo_File[6][10:-7])
    Mpart = float(Halo_File[5][15:-7])
    Halo_File = Halo_File[16:]  ### Rockstar
    H_Masses, H_Radii = [], []
    H_X, H_Y, H_Z = [], [], []
    subhalo_nbr = 0
    for i in range(len(Halo_File)):
        line = Halo_File[i].split(' ')
        if float(line[2]) > Mmin and float(line[2]) < Mmax and Nmin * Mpart<float(line[2]):
            if keep_subhalos: # keep subhalos
                H_Masses.append(float(line[2]))
                H_X.append(float(line[8]))
                H_Y.append(float(line[9]))
                H_Z.append(float(line[10]))
                H_Radii.append(float(line[5]))
            else :# do not keep subhalos
                if float(line[-1]) ==-1 : #not a subhalo
                    H_Masses.append(float(line[2]))
                    H_X.append(float(line[8]))
                    H_Y.append(float(line[9]))
                    H_Z.append(float(line[10]))
                    H_Radii.append(float(line[5]))
                else :
                    subhalo_nbr+=1 # add one to the count

    H_Masses, H_X, H_Y, H_Z, H_Radii = np.array(H_Masses), np.array(H_X), np.array(H_Y), np.array(H_Z), np.array(H_Radii)
    Dict = {'M':H_Masses,'X':H_X,'Y':H_Y,'Z':H_Z, 'R':H_Radii,'z':z,'Lbox':LBox,'subhalos': subhalo_nbr}

    return Dict



def from_Rockstar_to_Dict(rockstar_folder,output_folder):
    """
    rockstar_folder : path to the folder where the rockstar halo catalogs are stored.
    output_folder : where you want to store the dictionnaries.

    This functions reads in Rockstar halo catalogs and store the information we need as a dictionnary (halo masses etc..). Used to speed up.
    """
    import os


    for ii, filename in enumerate(os.listdir(rockstar_folder)):
        catalog = rockstar_folder + filename
        halo_catalog = Read_Rockstar(catalog)
        save_f(file = output_folder +'dct_'+ filename[4:],obj=halo_catalog)




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




