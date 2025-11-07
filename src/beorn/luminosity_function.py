"""
Global quantity computed directly from halo catalog
"""

import os.path
import numpy as np
from .cosmo import Hubble

from .constants import *
from .astro import f_star_Halo
from .functions import *


def Luminosity_Function(param):
    """""""""
    Computes the luminosity function of the snapshots located in folder param.sim.halo_catalogs.
    No dust obscuration.
    See 2009.01245 
    Output : 
    At every z, the [M_UV, Phi_Uv], two arrays.
    !!!! Halo masses should be given in the halo catalog as Msol/h 
    """""""""
    print('Computing luminosity function. There is no dust obscuration implemented.')
    M_bin = np.logspace(7, 13, 50, base=10)
    Ob, Om, h0 = param.cosmo.Ob, param.cosmo.Om, param.cosmo.h
    #alS = param.source.alS_lyal  ### power law lyman alpha SED
    #Anorm = (1 - alS) / (nu_LL ** (1 - alS) - nu_al ** (1 - alS))
    #Enu = 1 / (nu_LL - nu_al) * integrate.quad(lambda nu: h_eV_sec * Anorm * nu ** (-alS + 1), nu_al, nu_LL)[0]  # mean UV energy
    #K__UV = m_p_in_Msun * h0 *sec_per_year / (param.source.N_al * (Enu / eV_per_erg))  # Msol/h/yr/(erg/s)
    M_min = param.source.M_min
    K__UV = 1.15 * h0 *10**-28 # Msol/h/yr/(erg/s)

    print('K__UV is',K__UV)
    catalog_dir = param.sim.halo_catalogs

    Luminosity = {}
    for ii, filename in enumerate(os.listdir(catalog_dir)):
        catalog = catalog_dir + filename
        halo_catalog = load_f(catalog)  # Read_Rockstar(catalog)
        Mlist = halo_catalog['M']
        if len(Mlist) > 0:
            z = halo_catalog['z']
            Lbox = halo_catalog['Lbox']
            digitize = np.digitize(np.log(Mlist),np.log(M_bin))  ## means that M_bin[digitize[i]-1]<Mlist[i]<M_bin[digitize[i]]
            count = np.unique(digitize, return_counts=True)
            Mstar_dot = param.source.alpha_MAR * M_bin * f_star_Halo(param, M_bin) * Hubble(z, param) * (z + 1) * Ob / Om
            Mstar_dot[np.where(M_bin<M_min)]=0
            L_uv = Mstar_dot / K__UV
            M_UV = 51.63 - 0.4 ** -1 * np.log10(L_uv)
            dn_dMh = np.zeros((len(M_bin)))
            dn_dMh[count[0]] = count[1] / Lbox ** 3 / np.diff(M_bin)[count[0]]

            dMUV_dMh = np.gradient(M_UV, M_bin)

            phi_Muv = -h0 ** 3 * dn_dMh / dMUV_dMh  ### Mpc**-3.mag**-1

            indices = np.intersect1d(np.where(M_UV > -23), np.where(M_UV < -11))
            Luminosity[str(round(z, 3))] = np.array((M_UV[indices], phi_Muv[indices],M_bin[np.max(np.where(dn_dMh > 0))],M_UV[np.max(np.where(dn_dMh > 0))], M_bin[indices]))
            # Luminosity[str(round(z,3))] = {'Muv':M_UV[:],'phi_uv':phi_Muv[:],'dn_dMh':dn_dMh,'Mh':M_bin,}

            # np.array((M_UV[indices],phi_Muv[indices],M_bin[np.max(np.where(dn_dMh>0))],M_UV[np.max(np.where(dn_dMh>0))],M_bin[indices]))

    return Luminosity

