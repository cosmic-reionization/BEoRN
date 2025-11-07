"""
Mass Accretion Model
"""
import numpy as np
from scipy.interpolate import splrep, splev, interp1d
from scipy.integrate import odeint

from .cosmo import D, hubble
from .constants import *
from .cosmo import Hubble, hubble
from .halomassfunction import HaloMassFunction
from .parameters import Parameters


def mass_accretion(zz, mm, parameters: Parameters):
    if parameters.source.mass_accretion_model == 'EXP':
        Mh_history, dMh_dt = mass_accretion_EXP(zz, mm, parameters)
    elif parameters.source.mass_accretion_model == 'EPS':
        Mh_history, dMh_dt = mass_accretion_EPS(zz, mm, parameters)
    return Mh_history, dMh_dt


def mass_accretion_EPS(zz, mm, parameters: Parameters):
    """
    Assuming EPS formula
    (see Eq. 6 in 1409.5228)

    mm : array. The initial mass bin at z = zstart (self.M_Bin).
    zz : decreasing array of redshifts.

    Returns :
    Mh and dMh_dt, two 2D arrays of shape (zz, mm)
    """
    zz = np.flip(zz) # flip the z array so that it increases : zz = 6...25 etc. This way we solve the evolution of h masses backward in time, since M_Bin is defined as the h masses at the final redshift.
    aa = 1 / (zz + 1)
    Dgrowth = []
    for i in range(len(zz)):
        Dgrowth.append(D(aa[i], parameters))  # growth factor
    Dgrowth = np.array(Dgrowth)


    parameters.halo_mass_function.z = [0]  # we just want the linear variance
    # TODO: don't update parameters on the fly. They were set by the user!
    parameters.halo_mass_function.m_min = parameters.simulation.halo_mass_bin_min * 1e-5 ## Need small enough value for the source term below (0.6*M)
    parameters.halo_mass_function.m_max = parameters.simulation.halo_mass_bin_max
    HMF = HaloMassFunction(parameters)
    HMF.generate_HMF(parameters)
    var_tck = splrep(HMF.tab_M, HMF.sigma2)

    # free parameter
    fM = 0.6
    fracM = np.full(len(mm), fM)
    frac = interp1d(mm, fracM, axis=0, fill_value='extrapolate')

    Dg_tck = splrep(zz, Dgrowth)
    D_growth = lambda z: splev(z, Dg_tck)
    dDda = lambda z: splev(z, Dg_tck, der=1)

    Maccr = np.zeros((len(zz), len(mm)))
    source = lambda M, z: (2 / np.pi) ** 0.5 * M / (splev(frac(M) * M, var_tck, ext=1) - splev(M, var_tck, ext=1)) ** 0.5 * 1.686 / D_growth(z) ** 2 * dDda( z)

    Maccr[:, :] = odeint(source, mm, zz)
    Maccr = np.nan_to_num(Maccr, nan=0)

    Raccr = Maccr / mm[None, :]
    dMaccrdz = np.gradient(Maccr, zz, axis=0, edge_order=1)
    dMaccrdt = - dMaccrdz * (1 + zz)[:, None] * hubble(zz, parameters)[:, None] * sec_per_year / km_per_Mpc

    # remove NaN
    Raccr[np.isnan(Raccr)] = 0.0
    dMaccrdz[np.isnan(dMaccrdz)] = 0.0
    dMaccrdt[np.isnan(dMaccrdt)] = 0.0
    dMaccrdt = dMaccrdt.clip(min=0)

    return np.flip(Raccr * mm,axis=0), np.flip(dMaccrdt,axis=0)



def mass_accretion_EXP(zz, mm, parameters: Parameters):
    """
    Parameters
    ----------
    param : dictionary containing all the input parameters
    mm : arr. Halo mass (Msol/h) at the redshift z = min(zz)
    zz  : arr of redshifts

    Returns
    ----------
    Mh and dM_dt in EXP MAR. Two 2d array of shape (zz, mm)
    dM_dt is in [Msol/h/yr]
    """
    Mhalo = mm * np.exp(parameters.source.mass_accretion_alpha * (np.min(zz) - zz[:, None]))  # shape is [zz, Mass]
    dMh_dt = dMh_dt_EXP(parameters, Mhalo, zz[:, None])
    return Mhalo, dMh_dt

def dMh_dt_EXP(parameters: Parameters, Mh, z):
    """
    Parameters
    ----------
    param : dictionary containing all the input parameters
    Mh : arr. Halo masss
    z  : arr of redshifts, shape(Mh).

    Returns
    ----------
    Halo mass accretion rate, i.e. time derivative of halo mass (dMh/dt in [Msol/h/yr])
    """
    return parameters.source.mass_accretion_alpha * Mh * (z + 1) * Hubble(z, parameters)


def Mhalo_EXP(Mh_z6,zz):
    return Mh_z6*np.exp(-0.79*(zz-6))
