"""
Here we compute the Lyman_alpha and collisional coupling coefficient, in order to produce full dTb maps
"""

import numpy as np
from .constants import *
from .cross_sections import sigma_HI
import pkg_resources
from .cosmo import comoving_distance, Hubble, hubble
from .astro import f_star_Halo
from scipy.interpolate import splrep,splev,interp1d
from scipy.integrate import cumtrapz


def T_cmb(z):
    return Tcmb0 * (1+z)


def kappa_coll():
    """
    [cm^3/s]
    """

    names = 'T, kappa'
    path_to_file = pkg_resources.resource_filename('beorn', "input_data/kappa_eH.dat")
    eH = np.genfromtxt(path_to_file, usecols=(0, 1), comments='#', dtype=float, names=names)

    names = 'T, kappa'
    path_to_file = pkg_resources.resource_filename('beorn', 'input_data/kappa_HH.dat')
    HH = np.genfromtxt(path_to_file, usecols=(0, 1), comments='#', dtype=float, names=names)

    return HH, eH


def x_coll(z, Tk, xHI, rho_b):
    """
    Collisional coupling coefficient. 1d profile around a given halo.

    z     : redshift
    Tk    : 1d radial gas kinetic temperature profile [K]
    xHI   : 1d radial ionization fraction profile
    rho_b : baryon density profile in nbr of [H atoms /cm**3] (physical cm)
    Returns : x_coll 1d profile.

    """

    # nH and e- densities
    n_HI  = rho_b * xHI
    n_HII = rho_b * (1-xHI) # [1/cm^3]

    # prefac (Eq.10 in arXiv:1109.6012)
    Tcmb = T_cmb(z)
    prefac = Tstar / A10 / Tcmb  # [s]

    HH, eH = kappa_coll()
    kappa_eH_tck = splrep(eH['T'], eH['kappa'])
    kappa_eH = splev(Tk, kappa_eH_tck, ext=3)  # [cm^3/s]
    kappa_HH_tck = splrep(HH['T'], HH['kappa'])
    kappa_HH = splev(Tk, kappa_HH_tck, ext=3)

    x_HH = prefac * kappa_HH * n_HI
    x_eH = prefac * kappa_eH * n_HII
    return x_HH  + x_eH

def x_coll_coef(param):
    """
    Coefficient to turn rho/rho_mean into a baryon density in nbr of [H atoms /phys-cm**3]
    """
    Om, Ob, h0 = param.cosmo.Om, param.cosmo.Ob, param.cosmo.h
    coef = rhoc0 * h0 ** 2 * Ob * (1 + z) ** 3 * M_sun / cm_per_Mpc ** 3 / m_H
    return coef

def S_alpha(zz, Tgas, xHI):
    """
    Suppression factor S_alpha, dimensionless.
    Following method in astro-ph/0608032
    """

    # Eq.43
    tau_GP = 3.0e5 * xHI * ((1 + zz) / 7.0) ** 1.5
   # gamma = 1 / tau_GP

    # Eq. 55
    S_al = np.exp(-0.803 * Tgas ** (-2 / 3) * (1e-6 * tau_GP) ** (1 / 3))
  #  print('CAREFULL SALPHA IS 1')

    return S_al


def eps_lyal(nu,param):
    """
    Lymam-alpha part of the spectrum.
    See cosmicdawn/sources.py
    Return : eps (multiply by SFR and you get some [photons.yr-1.Hz-1])
    """
    h0    = param.cosmo.h
    N_al  = param.source.N_al  #9690 number of lya photons per protons (baryons) in stars
    alS = param.source.alS_lyal

    nu_min_norm  = nu_al
    nu_max_norm  = nu_LL

    Anorm = (1-alS)/(nu_max_norm**(1-alS) - nu_min_norm**(1-alS))
    Inu   = lambda nu: Anorm * nu**(-alS)

    eps_alpha = Inu(nu)*N_al/(m_p_in_Msun * h0)

    return eps_alpha



def phi_alpha(x,E):
    """
    Fraction of the absorbed photon energy that goes into excitation. [Dimensionless]
    From Dijkstra, Haiman, Loeb. Apj 2004.

    Parameters
    ----------
    x : ionized hydrogen fraction at location
    E : energy in eV

    Returns
    -------
    float
    """
    return 0.39*(1-x**(0.4092*a_alpha(x,E)))**1.7592

def a_alpha(x,E):
    """
    Used in phi_alpha.
    """
    return 2/np.pi * np.arctan(E/120 * (0.03/x**1.5 + 1)**0.25)



def J0_xray_lyal(r_grid,xHII, n_HI, Edot,z, param):
    """
    Xray flux that contributes to lyman alpha coupling. [pcm-2.s-1.Hz-1]. Will be added next to rho_alpha to compute x_alpha.
    Expression used in Thomas and Zaroubi 2011, and taken  from the appendix of Dijkstra 2004, a limit from the xray background...

    Parameters
    ----------
    r_grid : radial distance form source [pMpc/h-1]
    Edot : xray source energy in [eV.s-1] (float)
    xHII : ionized fraction. (array of size r_grid)
    n_HI : number density of hydrogen atoms in the cell [pcm-3] (array of size r_grid)
    z : redshift
    Returns
    -------
    float
    """
    xHII = xHII.clip(min=1e-50) #to avoid warnings

    sed_xray = param.source.alS_xray
    norm_xray = (1 - sed_xray) / ((param.source.E_max_sed_xray / h_eV_sec) ** (1 - sed_xray) - (param.source.E_min_sed_xray / h_eV_sec) ** (1 - sed_xray))   #Hz**(alpha-1)
    E_range = np.logspace(np.log10(50), np.log10(2000), 200, base=10)  # eq(3) Thomas.2011
    nu_range = Hz_per_eV * E_range

    cumul_nHI = cumtrapz(n_HI, r_grid, initial=0.0)  ## Mpc/h.cm-3
    Edotflux = Edot / (4 * np.pi * r_grid**2 * cm_per_Mpc ** 2 / param.cosmo.h ** 2)  # eV.s-1.pcm-2

    tau = cm_per_Mpc / param.cosmo.h * (cumul_nHI[:, None] * sigma_HI(E_range))  # shape is (r_grid,E_range)

    Nxray_arr = np.exp(-tau) * Edotflux[:, None] * norm_xray * nu_range[None, :] ** (-sed_xray) * Hz_per_eV  # [eV/eV/s/pcm^2], (r_grid,E_range)  array to integrate

    to_int = Nxray_arr * sigma_HI(E_range)[None, :] * phi_alpha(xHII[:, None], E_range)

    integral = np.trapz(to_int, E_range, axis=1)  # shape is r_grid
    return c__ * 1e2 * sec_per_year / (4 * np.pi * Hubble(z, param) * nu_al) * n_HI * integral / (h_eV_sec * nu_al) # [s-1.pcm-2.Hz-1]



def J_xray_no_redshifting(r_grid, n_HI, Edot, param):
    """
    Xray flux that contributes to heating. [pcm-2.s-1.Hz-1]. Will be added next to rho_alpha to compute x_alpha

    Parameters
    ----------
    r_grid : radial distance form source [pMpc/h-1]
    Edot : xray source energy in [eV.s-1] (float)
    xHII : ionized fraction. (array of size r_grid)
    n_HI : number density of hydrogen atoms in the cells [pcm-3] (array of size r_grid)
    z : redshift
    Returns
    -------
    array of size (r_grid,E_range) -- a xray spectrum at each r.
    """
    sed_xray = param.source.alS_xray
    norm_xray = (1 - sed_xray) / ((param.source.E_max_sed_xray / h_eV_sec) ** (1 - sed_xray) - (param.source.E_min_sed_xray / h_eV_sec) ** (1 - sed_xray))   #Hz**(alpha-1)
    E_range = np.logspace(np.log10(param.source.E_min_xray), np.log10(param.source.E_max_xray), 1000, base=10)  # eq(3) Thomas.2011
    nu_range = Hz_per_eV * E_range

    cumul_nHI = cumtrapz(n_HI, r_grid, initial=0.0)  ## Mpc/h.cm-3
    Edotflux = Edot / (4 * np.pi * r_grid**2 * cm_per_Mpc ** 2 / param.cosmo.h ** 2)  # eV.s-1.pcm-2

    tau = cm_per_Mpc / param.cosmo.h * (cumul_nHI[:, None] * sigma_HI(E_range))  # shape is (r_grid,E_range)

    Nxray_arr = np.exp(-tau) * Edotflux[:, None] * norm_xray * nu_range[None, :] ** (-sed_xray) * Hz_per_eV  # [eV/eV/s/pcm^2], (r_grid,E_range)  array to integrate

    return E_range, Nxray_arr  /nu_range[None, :]   # [photons/Hz/s/pcm^2], (r_grid,E_range)





def J_xray_with_redshifting(r_grid, n_HI, Mhalo, zz,param):
    """
    Same as above, but acounting for the redshifting of photons from source center to distance r.
    Returns
    -------
    array of size (E_range,r_grid) -- a xray spectrum at each r.
    """
    Ob, Om, h0 = param.cosmo.Ob, param.cosmo.Om, param.cosmo.h
    sed_xray = param.source.alS_xray
    norm_xray = (1 - sed_xray) / ((param.source.E_max_sed_xray / h_eV_sec) ** (1 - sed_xray) - (param.source.E_min_sed_xray / h_eV_sec) ** (1 - sed_xray))  # Hz**(alpha-1)
    E_range = np.logspace(np.log10(param.source.E_min_xray), np.log10(param.source.E_max_xray), 40, base=10)  # eq(3) Thomas.2011
    nu_range = Hz_per_eV * E_range

    alpha = param.source.alpha_MAR
    cumul_nHI = cumtrapz(n_HI, r_grid, initial=0.0)  ## Mpc/h.cm-3

    z_max = 35
    zrange = z_max - zz
    N_prime = int(zrange / 0.01)  ## ly al bin prime
    if (N_prime < 4):
        N_prime = 4
    z_prime = np.logspace(np.log(zz), np.log(z_max), N_prime, base=np.e)

    M_emission = Mhalo * np.exp(alpha * (zz - z_prime))
    dMh_dt_em = alpha * M_emission * (z_prime + 1) * Hubble(z_prime,param)  ## [(Msol/h) / yr], SFR at zprime, at emission
    E_dot_xray_em = dMh_dt_em * f_star_Halo(param,M_emission) * Ob / Om * param.source.cX * eV_per_erg / h0  # eV/s at emission

    eps_xray_em = E_dot_xray_em * norm_xray * (nu_range[:, None] * (1 + z_prime) / (1 + zz)) ** ( -sed_xray) * Hz_per_eV  ## eV/eV/s at emission with correct frequency to redshift to nu at zz.

    rcom_prime = comoving_distance(z_prime, param) * h0    # comoving distance in [cMpc/h]
    eps_int = interp1d(rcom_prime, eps_xray_em, axis=1, fill_value=0.0, bounds_error=False)
    tau = cm_per_Mpc / param.cosmo.h * (cumul_nHI * sigma_HI(E_range[:, None]))  # shape is (r_grid,E_range)

    flux_m = np.exp(-tau) * eps_int(r_grid * (1 + zz)) / ( 4 * np.pi * r_grid ** 2 * cm_per_Mpc ** 2 / param.cosmo.h ** 2)

    return E_range, flux_m/nu_range[:,None]       # [photons/Hz/s/pcm^2], (r_grid,E_range)

