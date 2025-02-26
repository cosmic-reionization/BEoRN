"""

FUNCTIONS RELATED TO COSMOLOGY
TODO Use astropy instead
"""
import os.path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import splrep,splev
from .constants import *
from astropy.cosmology import FlatLambdaCDM
from .parameters import Parameters

def hubble(z, parameters: Parameters):
    """
    Hubble parameter [km.s-1.Mpc-1]
    """
    Om = parameters.cosmology.Om
    Ol = 1.0-Om
    H0 = 100.0*parameters.cosmology.h
    return H0 * (Om*(1+z)**3 + (1.0 - Om - Ol)*(1+z)**2 + Ol)**0.5


def Hubble(z, parameters: Parameters):
    """
    Hubble parameter [yr-1]
    """
    Om, Ol = parameters.cosmology.Om, parameters.cosmology.Ol
    return parameters.cosmology.h * 100.0 * sec_per_year / km_per_Mpc * np.sqrt(Om*(1+z)**3 + (1.0-Om-Ol)*(1+z)**2+Ol)


def comoving_distance(z, parameters: Parameters):
    """
    Comoving distance between z[0] and z[-1]
    """
    return cumulative_trapezoid(c_km_s/hubble(z,parameters),z,initial=0)  # [Mpc]


def T_cmb(z):
    """
    CMB temperature [K]
    """
    return Tcmb0*(1+z)



def T_smooth_radio(z,parameters):
    """
    Smooth Background radiation temperature when a radio excess is present, i.e Ar is non zero
    """
    Tcmb0 = parameters.cosmology.Tcmb
    Ar = parameters.radio.Ar
    Ar = np.array(Ar) # this line is when you want a z-dependent Ar. (used it to reproduce fig 2 of 2008.04315)
    Beta_Rad = parameters.radio.Beta_Rad
    nu = 1420/(1+z) #### in MHz
    return Tcmb0*(1+z)*(Ar*(nu/78)**Beta_Rad)


def read_powerspectrum(parameters: Parameters):
    """
    Linear power spectrum from file
    """
    names='k, P'
    PS = np.genfromtxt(parameters.cosmology.ps,usecols=(0,1),comments='#',dtype=None, names=names)
    return PS


def T_adiab(z, parameters: Parameters):
    """
    Temperature of the gas assuming it decoupled from CMB at z = parameters.cosmology.z_decoupl and then cooled adiabatically.
    """
    return Tcmb0 * (1 + z) ** 2 / (1 + parameters.cosmology.z_decoupling)

def T_adiab_fluctu(z, parameters: Parameters, delta_b):
    """
    Fluctuating adiabatic background.
    delta_b : matter overdensity
    """
    return T_adiab(z,parameters) * (1 + delta_b) ** (2 / 3)



#define Hubble factor H=H0*E
def E(x, parameters: Parameters):
    return np.sqrt(parameters.cosmology.Om*(x**(-3))+1-parameters.cosmology.Om)

def D_non_normalized(a, parameters: Parameters):
    """
    a : input array 
    Integrate from a~0 (0.001) to a. We checked that it gives same results than integrate.quad for z=0 to 30
    """
    if np.any(a<0.001):
        print('Integration pb in Growth Factor.')
        exit()
    integrand = np.linspace(0.001, a, 100)
    w = np.trapz(1 / (integrand * E(integrand,parameters)) ** 3, integrand, axis=0)
    return (5 * parameters.cosmology.Om * E(a,parameters) / 2) * w

#define D normalized
def D(a,param):
    """
    Growth factor. Normalized to 1 at z = 0.
    """
    return D_non_normalized(a,param)/D_non_normalized(1, param)


def rhoc_of_z(parameters: Parameters,z):
    """
    Redshift dependence of critical density
    (in comoving units)
    Outputs is in Msol/cMpc**3
    """
    Om = parameters.cosmology.Om
    rhoc = 2.775e11 * parameters.cosmology.h**2  ## in Msol/cMpc**3
    return rhoc * (Om * (1.0 + z) ** 3.0 + (1.0 - Om)) / (1.0 + z) ** 3.0



def siny_ov_y(y):
    s = np.sin(y) / y
    s[np.where(y > 100)] = 0
    return s

def cosmo_astropy(param):
    Ob, Om, h0 = param.cosmo.Ob, param.cosmo.Om, param.cosmo.h
    cosmo = FlatLambdaCDM(H0=100 * h0, Om0= Om, Ob0 = Ob, Tcmb0=2.725)
    return cosmo



def correlation_fct(param):
    """
    Old function that we are not using now (2024). It might be usefull in the future to compute the 2-h term profile for non-homogeneous IGM.
    If the path to a new power spectrum (other than in src/files/PCDM_Planck.dat) is given in param.cosmo.ps,
    then the corr_function at z=0 is recomputed and then written at the location param.cosmo.corr_fct.
    Otherwise this function simply prints that it will read in the corr_fct from param.cosmo.corr_fct.
    """
    rmin = 0.005
    rmax = 100            # Mpc/h. Controls the maximum comoving scale that we compute. Can be important for very large scales 2h profile at high redshift. 100 should be safe
    PS_ = param.cosmo.ps  # z=0 linear power spectrum of matter perturb.
    path_to_corr_file = param.cosmo.corr_fct

    if os.path.isfile(path_to_corr_file):
        print('Correlation function already computed : par.cosmo.corr_fct')
    else:
        try:
            names = "k, PS"
            Power_Spec = np.loadtxt(PS_)
        except IOError:
            print('IOERROR: Cannot read power spec. Try: par.cosmo.ps = "/path/to/file"')
            exit()
        print('Computing the z=0 correlation function from the PS given in par.cosmo.ps')
        bin_N = 200
        bin_r = np.logspace(np.log(rmin), np.log(rmax), bin_N, base=np.e)
        krange = Power_Spec[:, 0]
        PS_values = Power_Spec[:, 1]
        bin_corr = np.trapz(krange ** 3 * PS_values * siny_ov_y(krange * bin_r[:, None]) / 2 / np.pi ** 2,np.log(krange))
        try:
            np.savetxt(path_to_corr_file, np.transpose([bin_r, bin_corr]))
            print('Saving the correlation function in ' + path_to_corr_file)
        except IOError:
            print('IOERROR: cannot write Cosmofct file in a non-existing directory!')
            exit()





def Tspin_fct(Tcmb,Tk,xtot):
    return ((1 / Tcmb + xtot / Tk ) / (1 + xtot)) ** -1


def dTb_fct(z, Tk, xtot, delta_b,x_HII, parameters: Parameters):
    """
    nHI_norm : (1+delta_b)*(1-xHII) , or also rho_HI/rhob_mean
    Returns : Birghtness Temperature in mK.
    """
    factor = dTb_factor(parameters)
    return factor * np.sqrt(1 + z) * (1 - Tcmb0 * (1 + z) / Tk) * (1 - x_HII) * xtot / (1 + xtot) * (1+delta_b)

def dTb_factor(parameters: Parameters):
    """
    Constant factor in dTb formula
    """
    Om, h0, Ob = parameters.cosmology.Om, parameters.cosmology.h, parameters.cosmology.Ob
    return 27 * Ob * h0 ** 2 / 0.023 * np.sqrt(0.15 / Om / h0 ** 2 / 10)


def Tvir_to_M(Tvir, z, param):
    '''
    Convert virial temperature to mass.

    Parameters:
        Tvir (float or array): The virial temperature(s) in K.
        z (float): the redshift.

    Returns:
        Mass in solar mass unit.
    '''
    Om = param.cosmo.Om
    Ol = param.cosmo.Ol
    Ok = 1 - Om - Ol
    Omz = Om * (1 + z) ** 3 / (Om * (1 + z) ** 3 + Ol + Ok * (1 + z) ** 2)
    d = Omz - 1
    Delc = 18 * np.pi ** 2 + 82 * d - 39 * d ** 2
    mu = 0.6  # 0.59 for fully ionized primordial gas, 0.61 for a gas with ionized H and singly ionized He, 1.22 for neutral primordial gas.
    conv_fact = 1.98e4 * (mu / 0.6) * (Om * Delc / Omz / 18 / np.pi ** 2) ** (1. / 3) * ((1 + z) / 10)
    M = 1e8 / param.cosmo.h * (Tvir / conv_fact) ** (3. / 2)
    return M


def M_to_Tvir(M, z, param):
    '''
    Convert mass to virial temperature.

    Parameters:
        M (float or array): The mass(es) in solar mass unit.
        z (float): the redshift.

    Returns:
        Virial temperature in K.
    '''
    Om = param.cosmo.Om
    Ol = param.cosmo.Ol
    Ok = 1 - Om - Ol
    Omz = Om * (1 + z) ** 3 / (Om * (1 + z) ** 3 + Ol + Ok * (1 + z) ** 2)
    d = Omz - 1
    Delc = 18 * np.pi ** 2 + 82 * d - 39 * d ** 2
    mu = 0.6  # 0.59 for fully ionized primordial gas, 0.61 for a gas with ionized H and singly ionized He, 1.22 for neutral primordial gas.
    conv_fact = 1.98e4 * (mu / 0.6) * (Om * Delc / Omz / 18 / np.pi ** 2) ** (1. / 3) * ((1 + z) / 10)
    Tvir = conv_fact * (M * param.cosmo.h / 1e8) ** (2. / 3)
    return Tvir



def Thomson_optical_depth(zz, xHII, param):
    """
    Cumulative optical optical depth of array zz.
    xHII : global ionisation fraction history
    See e.g. Eq. 6 of 1406.4120 or eq. 12 from 2101.01712, or eq. 84 from Planck_2018_results_L06.
    """
    # check if zz array is in increasing order.
    is_increasing = zz[0]<zz[-1]
    if not is_increasing: zz,xHII = np.flip(zz),np.flip(xHII)

    z0 = zz[0]
    if z0 > 0:  ## the integral has to be done starting from z=0
        low_z = np.arange(0, z0, 0.5)
        zz = np.concatenate((low_z, zz))
        xHII = np.concatenate((np.full(len(low_z), xHII[0]), xHII))

    if xHII[0] < 1:
        xHII[0] = 1
        print(
            'Warning: reionisation is not complete at the lower redshift available!! The CMB otpical depth calculation will be wrong.')

    from scipy.integrate import cumtrapz, trapz, odeint
    Ob = param.cosmo.Ob
    h0 = param.cosmo.h

    # hydrogen and helium cross sections
    sHII = sigma_T * 1e4 * (h0 / cm_per_Mpc) ** 2  # [Mpc/h]^2
    nb0 = rhoc0 * Ob / (m_p_in_Msun * h0)  # [h/Mpc]^3
    # H abundances
    nHII = xHII * nb0 * (1 + zz) ** 3  # [h/Mpc]^3
    # proper line element
    dldz = c_km_s * h0 / hubble(zz, param) / (1 + zz)  # [Mpc/h]
    # integrate
    tau_int = dldz * (nHII * sHII)  # + nHeI*sHeI + nHeII*sHeII)
    tau = cumtrapz(tau_int, x=zz, axis=0, initial=0.0)

    return zz, tau  # [np.where(zz>=z0)]


def R_of_M(M):
    R = (3 * M / (200 * rhoc0 * 0.31 * np.pi * 4)) ** (1 / 3)
    return R_of_M