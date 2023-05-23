
import math
import numpy as np
from scipy.interpolate import splrep,splev,interp1d
from .constants import *
from .cosmo import D, rhoc_of_z





delta_c = 1.686

def delt_c(z,param):
    return delta_c/D(1/(1+z),param)


def wf_sharpk(y):
    return np.heaviside(1 - y, 0)

def wf_tophat(x):
    return 3 * (np.sin(x) - x * np.cos(x)) / (x) ** 3


def read_powerspectrum(param):
    """
    Linear power spectrum from file
    """
    names= 'k, P'
    PS = np.genfromtxt(param.cosmo.ps,usecols=(0,1),comments='#',dtype=None, names=names) #
    return PS

def Variance_tophat(param,mm):
    """
    Sigma**2 at z=0 computed with a tophat filter. Used to compute the barrier.
    Output : Var and dlnVar_dlnM
    We reintroduce little h units to be consistent with Power spec units.
    """
    ps = read_powerspectrum(param)
    kk_ = ps['k']
    PS_ = ps['P']
    rhoc = 2.775e11 ### with h
    R_ = ((3 * mm / (4 * rhoc * param.cosmo.Om * np.pi)) ** (1. / 3))
    #Var = np.trapz(kk_ ** 2 * PS_ * wf_tophat(kk_ * R_[:, None]) ** 2 / (2 * np.pi ** 2), kk_, axis=-1)
    Var = np.trapz(kk_ ** 2 * PS_ * wf_tophat(kk_ * R_) ** 2 / (2 * np.pi ** 2), kk_)
   # dlnVar_dlnM = np.gradient(np.log(Var), np.log(mm) )
    return Var #, dlnVar_dlnM


def bias(z,param,Mass = None,bias_type='Tinker'):
    M = Mass
    if bias_type == 'ST':
        #ST bias
        q = 0.707 # sometimes called a
        p = 0.3
        dcz = delt_c(z,param)
        var = Variance_tophat(param,M)
        nu = dcz ** 2.0 / var
        # cooray and sheth
        e1 = (q * nu - 1.0) / delta_c
        E1 = 2.0 * p / dcz / (1.0 + (q * nu) ** p)
        bias = 1.0 + e1 + E1

    elif (bias_type == 'Tinker'):
        # tinker bias
        dcz = delt_c(z,param)
        dc = delta_c
        var = Variance_tophat(param,M)
        nu = dcz ** 2.0 / var
        y = np.log10(200)
        A = 1 + 0.24 * y * np.exp(-(4 / y) ** 4)
        a = 0.44 * y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4)
        c = 2.4
        bias = 1 - A * nu ** (a / 2) / (nu ** (a / 2) + dc ** a) + B * nu ** (b / 2) + C * nu ** (c / 2)
    return bias



def rho_2h(bias_, cosmo_corr_ ,param, z):
    """
    Large scale 2halo profile, matter density around a halo. In Msol/cMpc**3
    """
    return (bias_ * cosmo_corr_ * D(1/(z+1)**2,param) * param.cosmo.profile + 1.0) * param.cosmo.Om * rhoc_of_z(param, 0) * param.cosmo.clumping


def rhoNFW_fct(rbin,param):
    """
    NFW density profile. We do not use it in the code.
    """
    Mvir = param.source.M_halo
    cvir = param.source.C_halo
    rvir = (3.0*Mvir/(4.0 * np.pi * 200*rhoc_of_z(param)))**(1.0/3.0)
    rho0 = 200*rhoc_of_z(param)*cvir**3.0/(3.0*np.log(1.0+cvir)-3.0*cvir/(1.0+cvir))
    x = cvir*rbin/rvir
    return rho0/(x * (1.0+x)**2.0)

def R_halo(M_halo,z,param):
    """
    M_halo in Msol.
    Output : Halo radius is physical unit [pMpc]
    """
    return (3*M_halo/(4*math.pi*200*rhoc_of_z(param,z)*(1+z)**3))**(1.0/3)



def profile(bias_,cosmo_corr_,param, z):
    """
    Global profile of total matter, in Msol/Mpc**3
    This is a comoving density [(Msol)/(cMpc)**3] as a function of comoving radius [cMpc/h]
    """
    return rho_2h(bias_, cosmo_corr_, param, z)     #+ rhoNFW_fct(rbin,param)



def bar_density_2h(rgrid,param,z,Mass):
    """
    2h profiles in nbr of [baryons /co-cm**3]
    rgrid : radial coordinates in physical units [pMpc/h].
    Mass : Halo mass in Msol/h. Used to compute the bias.
    """
    # Profiles
    cosmofile = param.cosmo.corr_fct
    vc_r, vc_corr = np.loadtxt(cosmofile, usecols=(0, 1), unpack=True)
    corr_tck = splrep(vc_r, vc_corr, s=0)
    cosmo_corr = splev(rgrid * (1 + z) , corr_tck)  # r_grid * (1+self.z)  in cMpc/h --> To reach the correct scales and units for the correlation fucntion
    halo_bias = bias(z, param,Mass)
    # baryonic density profile in [co-cm**-3]
    norm_profile = profile(halo_bias, cosmo_corr, param, z) * param.cosmo.Ob / param.cosmo.Om * M_sun / (cm_per_Mpc) ** 3 / m_H

    return norm_profile
