"""

FUNCTIONS RELATED TO COSMOLOGY
"""
import os.path
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid

try:
    from numpy import trapezoid as _trapz
except ImportError:
    from numpy import trapz as _trapz

from ..constants import sec_per_year, km_per_Mpc, c_km_s, Tcmb0, sigma_T, cm_per_Mpc, rhoc0, m_p_in_Msun
from ..structs import Parameters

def dark_energy_density_factor(a, parameters: Parameters):
    """rho_DE(a) / rho_DE(a=1) for the CPL (Chevallier-Polarski-Linder)
    dark-energy parameterization w(a) = w0 + wa*(1-a):

        rho_DE(a)/rho_DE(1) = a^(-3*(1+w0+wa)) * exp(-3*wa*(1-a))

    Reduces to 1 (a cosmological constant) when w0=-1, wa=0 — the defaults —
    so :func:`E`/:func:`hubble`/:func:`hubble_per_yr` are numerically
    unchanged for any existing script that doesn't set w0/wa.
    """
    w0 = parameters.cosmology.w0
    wa = parameters.cosmology.wa
    return a**(-3.0*(1.0 + w0 + wa)) * np.exp(-3.0*wa*(1.0 - a))


def hubble(z, parameters: Parameters):
    """
    Hubble parameter [km.s-1.Mpc-1]
    """
    Om = parameters.cosmology.Om
    Ol = 1.0-Om
    H0 = 100.0*parameters.cosmology.h0
    a = 1.0 / (1.0 + z)
    return H0 * np.sqrt(Om*(1+z)**3 + Ol*dark_energy_density_factor(a, parameters))


def hubble_per_yr(z, parameters: Parameters):
    """
    Hubble parameter [yr-1]
    """
    Om = parameters.cosmology.Om
    Ol = 1.0 - Om
    a = 1.0 / (1.0 + z)
    return parameters.cosmology.h0 * 100.0 * sec_per_year / km_per_Mpc * np.sqrt(Om*(1+z)**3 + Ol*dark_energy_density_factor(a, parameters))


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


def read_powerspectrum(ps_path: Path):
    """
    Linear power spectrum from file
    """
    names='k, P'
    PS = np.genfromtxt(ps_path,usecols=(0,1),comments='#',dtype=None, names=names)
    return PS


def T_adiab(z, parameters: Parameters):
    """
    Temperature of the gas assuming it decoupled from CMB at z = parameters.solver.z_decoupling and then cooled adiabatically.
    """
    return Tcmb0 * (1 + z) ** 2 / (1 + parameters.solver.z_decoupling)

def T_adiab_fluctu(z, parameters: Parameters, delta_b):
    """
    Fluctuating adiabatic background.
    delta_b : matter overdensity
    """
    return T_adiab(z,parameters) * (1 + delta_b) ** (2 / 3)



#define Hubble factor H=H0*E
def E(a, parameters: Parameters):
    """Dimensionless Hubble rate E(a) = H(a)/H0 for flat CPL dark energy
    (w0=-1, wa=0 by default -> cosmological constant)."""
    Om = parameters.cosmology.Om
    Ol = 1.0 - Om
    return np.sqrt(Om*(a**(-3)) + Ol*dark_energy_density_factor(a, parameters))

def D_non_normalized(a, parameters: Parameters):
    """
    a : input array
    Integrate from a~0 (0.001) to a. We checked that it gives same results than integrate.quad for z=0 to 30
    """
    if np.any(a<0.001):
        print('Integration pb in Growth Factor.')
        exit()
    integrand = np.linspace(0.001, a, 100)
    w = _trapz(1 / (integrand * E(integrand,parameters)) ** 3, integrand, axis=0)
    return (5 * parameters.cosmology.Om * E(a,parameters) / 2) * w


def _omega_m_of_a(a, parameters: Parameters):
    """Matter density parameter Omega_m(a) = Om*a^-3 / E(a)^2."""
    return parameters.cosmology.Om * a**(-3) / E(a, parameters)**2


def _omega_de_of_a(a, parameters: Parameters):
    """Dark-energy density parameter Omega_DE(a) = Ol*rho_DE(a)/rho_DE(1) / E(a)^2."""
    Ol = 1.0 - parameters.cosmology.Om
    return Ol * dark_energy_density_factor(a, parameters) / E(a, parameters)**2


def D_cpt92_non_normalized(a, parameters: Parameters):
    """Unnormalized linear growth factor via the Carroll, Press & Turner
    (1992, ARA&A, 30, 499) analytic fitting formula — accurate to ~1% for
    flat LCDM (w0=-1, wa=0). This is the same formula used by py21cmfast's
    ``dicke()`` (UsefulFunctions.c), so selecting this method reproduces
    py21cmfast's growth factor for direct 2LPT comparisons.

    Its ~1% accuracy claim is validated only for a cosmological constant;
    for w0/wa != (-1, 0) this still evaluates (Omega_m(a), Omega_DE(a) are
    well-defined for any CPL cosmology), but the fit's accuracy has not been
    validated away from w=-1 — prefer 'linder2005'/'linder_cahn2007' there.
    """
    Om_a = _omega_m_of_a(a, parameters)
    Ol_a = _omega_de_of_a(a, parameters)
    g = 2.5*Om_a / (Om_a**(4.0/7.0) - Ol_a + (1.0 + Om_a/2.0)*(1.0 + Ol_a/70.0))
    return g * a


def _growth_index_gamma(w):
    """Linder (2005) growth-index gamma(w), continuous at w=-1."""
    return 0.55 + 0.05*(1.0 + w) if w >= -1.0 else 0.55 + 0.02*(1.0 + w)


def D_linder2005_non_normalized(a, parameters: Parameters):
    """Unnormalized linear growth factor via the Linder (2005, PhRvD, 72,
    043529) growth-index approximation: d ln D/d ln a = Omega_m(a)^gamma,
    with a single gamma evaluated at w(z=1). Under BEoRN's default CPL
    parameters (w0=-1, wa=0) this reduces to the classic fixed gamma=0.55
    (Omega_m(z)^0.55) growth-rate approximation.
    """
    from scipy.integrate import quad
    w0, wa = parameters.cosmology.w0, parameters.cosmology.wa
    w1 = w0 + 0.5*wa  # w(a=0.5), i.e. w(z=1)
    gamma = _growth_index_gamma(w1)

    def integrand(ap):
        return (_omega_m_of_a(ap, parameters)**gamma - 1.0) / ap

    def _D(a_scalar):
        ln_D, _ = quad(integrand, 1e-3, a_scalar, limit=200)
        return a_scalar * np.exp(ln_D)

    return np.vectorize(_D)(a)


def D_linder_cahn2007_non_normalized(a, parameters: Parameters):
    """Unnormalized linear growth factor via the Linder & Cahn (2007,
    Astropart.Phys. 28, 481) scale-factor-dependent growth index: same
    ODE as 'linder2005' but with gamma(a) tracking w(a) at every point
    of the integral rather than a single gamma(w(z=1)). Only differs from
    'linder2005' when wa != 0; under the default w0=-1, wa=0 both reduce
    to the fixed gamma=0.55 approximation.
    """
    from scipy.integrate import quad
    w0, wa = parameters.cosmology.w0, parameters.cosmology.wa

    def integrand(ap):
        w = w0 + wa*(1.0 - ap)
        gamma = _growth_index_gamma(w)
        return (_omega_m_of_a(ap, parameters)**gamma - 1.0) / ap

    def _D(a_scalar):
        ln_D, _ = quad(integrand, 1e-3, a_scalar, limit=200)
        return a_scalar * np.exp(ln_D)

    return np.vectorize(_D)(a)


_GROWTH_FACTOR_METHODS = {
    'integral': D_non_normalized,
    'cpt92': D_cpt92_non_normalized,
    'linder2005': D_linder2005_non_normalized,
    'linder_cahn2007': D_linder_cahn2007_non_normalized,
}


#define D normalized
def D(a, param):
    """
    Growth factor. Normalized to 1 at z = 0.

    The computation method is controlled by
    ``param.cosmology.growth_factor_method`` — see
    :class:`~beorn.structs.parameters.CosmologyParameters` for the available
    options and their references.
    """
    method = param.cosmology.growth_factor_method
    try:
        D_unnormalized = _GROWTH_FACTOR_METHODS[method]
    except KeyError:
        raise ValueError(
            f"Unknown growth_factor_method {method!r}; expected one of "
            f"{sorted(_GROWTH_FACTOR_METHODS)}."
        )
    return D_unnormalized(a, param) / D_unnormalized(1.0, param)


def rhoc_of_z(parameters: Parameters,z):
    """
    Redshift dependence of critical density
    (in comoving units)
    Outputs is in Msol/cMpc**3
    """
    Om = parameters.cosmology.Om
    rhoc = 2.775e11 * parameters.cosmology.h0**2  ## in Msol/cMpc**3
    return rhoc * (Om * (1.0 + z) ** 3.0 + (1.0 - Om)) / (1.0 + z) ** 3.0



def siny_ov_y(y):
    s = np.sin(y) / y
    s[np.where(y > 100)] = 0
    return s

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
            _names = "k, PS"
            Power_Spec = np.loadtxt(PS_)
        except IOError:
            print('IOERROR: Cannot read power spec. Try: par.cosmo.ps = "/path/to/file"')
            exit()
        print('Computing the z=0 correlation function from the PS given in par.cosmo.ps')
        bin_N = 200
        bin_r = np.logspace(np.log(rmin), np.log(rmax), bin_N, base=np.e)
        krange = Power_Spec[:, 0]
        PS_values = Power_Spec[:, 1]
        bin_corr = _trapz(krange ** 3 * PS_values * siny_ov_y(krange * bin_r[:, None]) / 2 / np.pi ** 2,np.log(krange))
        try:
            np.savetxt(path_to_corr_file, np.transpose([bin_r, bin_corr]))
            print('Saving the correlation function in ' + path_to_corr_file)
        except IOError:
            print('IOERROR: cannot write Cosmofct file in a non-existing directory!')
            exit()





def Tspin_fct(Tcmb,Tk,xtot):
    return ((1 / Tcmb + xtot / Tk ) / (1 + xtot)) ** -1


def dTb_factor(parameters: Parameters):
    """
    Constant factor in dTb formula
    """
    Om, h0, Ob = parameters.cosmology.Om, parameters.cosmology.h0, parameters.cosmology.Ob
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
    M = 1e8 / param.cosmo.h0 * (Tvir / conv_fact) ** (3. / 2)
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
    Tvir = conv_fact * (M * param.cosmo.h0 / 1e8) ** (2. / 3)
    return Tvir



def Thomson_optical_depth(zz, xHII, param):
    """
    Cumulative optical optical depth of array zz.
    xHII : global ionisation fraction history
    See e.g. Eq. 6 of 1406.4120 or eq. 12 from 2101.01712, or eq. 84 from Planck_2018_results_L06.
    """
    # check if zz array is in increasing order.
    is_increasing = zz[0]<zz[-1]
    if not is_increasing:
        zz, xHII = np.flip(zz), np.flip(xHII)

    z0 = zz[0]
    if z0 > 0:  ## the integral has to be done starting from z=0
        low_z = np.arange(0, z0, 0.5)
        zz = np.concatenate((low_z, zz))
        xHII = np.concatenate((np.full(len(low_z), xHII[0]), xHII))

    if xHII[0] < 1:
        xHII[0] = 1
        print(
            'Warning: reionisation is not complete at the lower redshift available!! The CMB otpical depth calculation will be wrong.')

    from scipy.integrate import cumtrapz
    Ob = param.cosmo.Ob
    h0 = param.cosmo.h0

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
    _R = (3 * M / (200 * rhoc0 * 0.31 * np.pi * 4)) ** (1 / 3)
    return R_of_M
