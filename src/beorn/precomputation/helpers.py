"""Helper routines for precomputation of radiation profiles.

This module provides utilities used by the 1D radiation-profile
solver, including photon production rates, mean X-ray ionisation
quantities, temperature fits and cumulative optical depth
calculations.
"""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.interpolate import splrep, splev, interp1d
from scipy.integrate import trapezoid, solve_ivp, cumulative_trapezoid
import logging
logger = logging.getLogger(__name__)

from ..couplings import eps_lyal
from ..structs.parameters import Parameters
from ..cross_sections import sigma_HI, sigma_HeI
from ..constants import sec_per_year, m_H, M_sun, m_p_in_Msun, km_per_Mpc, h_eV_sec, cm_per_Mpc, E_HI, E_HeI, c_km_s, Tcmb0, nu_LL, rhoc0
from ..astro import f_star_Halo, f_esc, eps_xray
from ..cosmo import comoving_distance, hubble



def Ngdot_ion(parameters: Parameters, zz: np.ndarray, Mh: np.ndarray, dMh_dt: np.ndarray) -> np.ndarray:
    """Compute the ionizing photon production rate Ngamma_dot.

    The routine computes the number of ionizing photons emitted per
    second for a given halo mass evolution and mass-accretion rate,
    taking into account the chosen source model in ``parameters``.

    Args:
        parameters (Parameters): Simulation and source parameters.
        zz (numpy.ndarray): Redshift grid used for the mass-accretion.
        Mh (numpy.ndarray): Halo mass evolution array (M_sun/h).
        dMh_dt (numpy.ndarray): Mass accretion rate array.

    Returns:
        numpy.ndarray: Photon production rate in s^-1 with the same
        broadcastable shape as the inputs.
    """
    Ob, Om, h0 = parameters.cosmology.Ob, parameters.cosmology.Om, parameters.cosmology.h

    if parameters.source.source_type == 'SED':
        Ngam_dot_ion = dMh_dt / h0 * f_star_Halo(parameters, Mh) * Ob / Om * f_esc(parameters, Mh) * parameters.source.Nion / sec_per_year / m_H * M_sun
        Ngam_dot_ion[np.where(Mh < parameters.source.halo_mass_min)] = 0
        return Ngam_dot_ion
    elif parameters.source.source_type == 'Ghara':
        # TODO - this is untested and returns the wrong shape
        logger.warning('Ghara source type is chosen, Nion becomes just a fine tuning multiplicative factor')
        Mh = zz**0 * Mh * parameters.source.Nion # to make sure it has the correct shape
        Ngam_dot_ion = 1.33 * 1e43 * Mh/h0
        Ngam_dot_ion[np.where(Mh < parameters.source.halo_mass_min)] = 0
        return Ngam_dot_ion  # eq (1) from 3D vs 1D RT schemes.
    elif parameters.source.source_type == 'constant':
        # TODO - this is untested and returns the wrong shape
        logger.info('constant number of ionising photons chosen. Param.source.Nion becomes Ngam_dot_ion.')
        return np.full(len(zz), parameters.source.Nion)
    elif parameters.source.source_type == 'Ross':
        return Mh / h0 * Ob / Om / (10 * 1e6 * sec_per_year) / m_p_in_Msun
    else:
        raise TypeError(f'Source Type {parameters.source.source_type} not allowed.')




def mean_gamma_ion_xray(parameters: Parameters, sfrd, zz):
    """Compute mean X-ray primary and secondary ionisation rates.

    This routine integrates the cosmological X-ray background to
    estimate the mean primary and secondary ionisation rates used to
    evolve the global free-electron fraction.

    Args:
        parameters (Parameters): Cosmology and source parameters.
        sfrd (array): Star-formation-rate density (M_sun/h/yr/(Mpc/h)^3).
        zz (array): Redshift grid (decreasing order recommended).

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: ``(Gamma_ion, Gamma_sec_ion)``
        arrays containing the primary and secondary X-ray ionisation
        rates (s^-1) evaluated on ``zz``.
    """

    Ob = parameters.cosmology.Ob
    h0 = parameters.cosmology.h
    zstar = 35
    Emin = parameters.source.energy_cutoff_min_xray
    Emax = parameters.source.energy_cutoff_max_xray
    NE = 50

    nb0 = rhoc0 * Ob / (m_p_in_Msun * h0)  # [h/Mpc]^3

    # zprime binning
    dz_prime = 0.1

    # define frequency bin
    nu_min = Emin / h_eV_sec
    nu_max = Emax / h_eV_sec
    N_mu = NE
    nu = np.logspace(np.log(nu_min), np.log(nu_max), N_mu, base=np.e)

    f_He_bynumb = 1 - parameters.cosmology.HI_frac
    # hydrogen
    nH0 = (1 - f_He_bynumb) * nb0
    # helium
    nHe0 = f_He_bynumb * nb0

    Gamma_ion = np.zeros(len(zz))  # primary ion
    Gamma_sec_ion = np.zeros(len(zz))  # secondary ion

    sfrd_interp = interp1d(zz, sfrd, fill_value='extrapolate')

    for i in range(len(zz)):
        J_X_nu_z = np.zeros(len(nu))
        if (zz[i] < zstar):
            # rr_comoving = rr * (1 + zz[i])
            z_max = zstar
            zrange = z_max - zz[i]
            N_prime = int(zrange / dz_prime)

            if (N_prime < 4):
                N_prime = 4
            z_prime = np.logspace(np.log(zz[i]), np.log(z_max), N_prime, base=np.e)

            for j in range(len(nu)):
                tau_prime = cum_optical_depth(z_prime, nu[j] * h_eV_sec, parameters)
                eps_X = eps_xray(nu[j] * (1 + z_prime) / (1 + zz[i]), parameters)  * sfrd_interp(z_prime)  # [1/s/Hz/(Mpc/h)^3]
                itd = c_km_s * h0 / hubble(z_prime, parameters) * eps_X * np.exp(-tau_prime)

                J_X_nu_z[j] = (1 + zz[i]) ** 2 / (4 * np.pi) * trapezoid(itd, z_prime) * (h0/cm_per_Mpc)**2       # [1/s/Hz * (1/cm)^2]


        itlH = nH0 * sigma_HI(nu * h_eV_sec) * J_X_nu_z
        itlHe = nHe0 * sigma_HeI(nu * h_eV_sec) * J_X_nu_z
        Gamma_ion[i] = 4*np.pi * trapezoid((itlH+itlHe),nu) / nb0  # s^-1

        itlH = itlH * (nu * h_eV_sec - E_HI) / E_HI
        itlHe = itlHe * (nu * h_eV_sec - E_HeI) / E_HeI
        Gamma_sec_ion[i] = 4 * np.pi * trapezoid((itlH + itlHe), nu) / nb0  # [1/s]

    return Gamma_ion, Gamma_sec_ion


def T_gas_fit(zz):
    """Approximate fit for the gas temperature.

    The fit follows Eq. 2 in arXiv:1005.2416 and is used for quick
    estimates of the gas temperature as a function of redshift.

    Args:
        zz (array): Redshift values.

    Returns:
        numpy.ndarray: Approximate gas temperature in Kelvin.
    """
    a = 1/(1+zz)
    a1 = 1.0/119.0
    a2 = 1.0/115.0
    Tgas = Tcmb0/a/(1.0+(a/a1)/(1.0+(a2/a)**(3.0/2.0)))
    return Tgas


def solve_xe(parameters: Parameters, mean_G_ion, mean_Gsec_ion, zz: np.ndarray):
    """
    Parameters
    ----------
    param : dictionary containing all the input parameters
    mean_G_ion,mean_Gsec_ion : output of Mean_Gamma_ion_xray
    zz : redshift in decreasing order.

    Returns
    ----------
    Mean free electron fraction in the neutral medium. We use it to compute the fraction of energy deposited as heat by e- originating from ionisation by xray: fXh = xe**0.225
    """
    logger.info('Computing x_e(z) from the sfrd, including first and secondary ionisations....')
    h0 = parameters.cosmology.h
    Ob = parameters.cosmology.Ob
    f_He_bynumb = 1 - parameters.cosmology.HI_frac

    xe0 = 2e-4
    aa = list((1 / (1 + zz)))

    nb0 = rhoc0 * Ob / (m_p_in_Msun * h0)  # [h/Mpc]^3

    # fit from astro-ph/9909275
    tt = T_gas_fit(zz) / 1e4
    alB = 1.14e-19 * 4.309 * tt ** (-0.6166) / (1 + 0.6703 * tt ** 0.53) * 1e6  # [cm^3/s]
    alB = alB * (h0 / cm_per_Mpc) ** 3
    alB_tck = splrep(aa, alB)
    alphaB = lambda a: splev(a, alB_tck)

    # Energy deposition from first ionisation, see astro-ph/060723 (Eq.12) or 1509.07868 (Eq.3)
   # Gamma_HI = np.interp(zz, aa, mean_G_ion, right=0)
   # G_sec_ion_tck = np.interp(zz, aa, mean_G_ion, right=0)

    fXion = lambda xe: (1 - xe) / 2.5  # approx from Fig.4 of 0910.4410
    gamma_HI = lambda a, xe: np.interp(a, aa, mean_Gsec_ion, right=0) * fXion(xe)
    nH = lambda a: (1 - f_He_bynumb) * nb0 / a ** 3

    # x_e
    source = lambda a, xe: (np.interp(a, aa, mean_G_ion, right=0)  + gamma_HI(a, xe)) * (1 - xe) / (a * hubble(1 / a - 1, parameters) / km_per_Mpc) - \
                           alphaB(a) * nH(a) * xe ** 2 / (a * hubble(1 / a - 1, parameters) / km_per_Mpc)

    result = solve_ivp(source, [aa[0], aa[-1]], xe0, t_eval=aa)
    x_e = result.y
    logger.debug('.....done computing x_e(z).')
    return x_e




def rho_alpha_profile(parameters: Parameters, z_bins: np.ndarray, r_grid: np.ndarray, halo_mass: np.ndarray, halo_mass_derivative: np.ndarray):
    """
    Ly-al coupling profile
    of shape (r_grid)
    - r_grid : physical distance around halo center in [pMpc/h]
    - zz  : redshift
    - MM  : halo mass

    Return rho_alpha : shape is (zz,rr,MM). Units : [pcm-2.s-1.Hz-1]
    """
    # TODO: remove hardcoded values
    z_star = 35
    h0 = parameters.cosmology.h
    rectrunc = 23

    # rec fraction
    names = 'n, f'
    path_to_file = Path(importlib.util.find_spec('beorn').origin).parent / 'input_data' / 'recfrac.dat'
    rec = np.genfromtxt(path_to_file, usecols=(0, 1), comments='#', dtype=float, names=names)

    nu_n = nu_LL * (1 - 1 / rec['n'] ** 2)
    nu_n[nu_n == 0] = np.inf

    # rho_alpha = np.zeros((len(z_bins), len(r_grid), len(MM[0, :])))

    rho_alpha = np.zeros((len(r_grid), parameters.simulation.halo_mass_bin_n - 1, len(parameters.simulation.halo_mass_accretion_alpha) - 1, len(z_bins)))

    for i, z in enumerate(z_bins):
        if z > z_star:
            continue

        # Precompute z_max and z_prime for all k
        z_max = (1 - (rec['n'][2:rectrunc] + 1) ** (-2)) / (1 - (rec['n'][2:rectrunc]) ** (-2)) * (1 + z) - 1
        z_max = np.minimum(z_max, z_star)
        z_ranges = z_max - z

        N_primes = np.maximum((z_ranges / 0.01).astype(int), 4)  # Ensure at least 4 points
        z_primes = [np.logspace(np.log(z), np.log(z_max_k), N_prime_k, base=np.e) for z_max_k, N_prime_k in zip(z_max, N_primes)]

        # Compute rcom_prime for all k
        rcom_primes = [comoving_distance(z_prime, parameters) * h0 for z_prime in z_primes]

        # Compute dMdt_star interpolator
        dMdt_star = halo_mass_derivative[..., :i+1] * f_star_Halo(parameters, halo_mass[..., :i+1]) * parameters.cosmology.Ob / parameters.cosmology.Om
        dMdt_star_int = interp1d(
            np.concatenate((np.array([z_star]), z_bins[:i + 1])),
            np.concatenate((np.zeros_like(dMdt_star[..., :1]), dMdt_star), axis=-1),
            axis=-1,
            fill_value='extrapolate'
        )

        # Vectorized computation for all k
        flux = np.zeros((len(r_grid), parameters.simulation.halo_mass_bin_n - 1, len(parameters.simulation.halo_mass_accretion_alpha) - 1))
        for k, (z_prime, rcom_prime) in enumerate(zip(z_primes, rcom_primes)):
            nu_prime = nu_n[k + 2] * (1 + z_prime) / (1 + z)
            eps_al = eps_lyal(nu_prime, parameters)[None, None, :] * dMdt_star_int(z_prime)
            eps_int = interp1d(rcom_prime, eps_al, axis=-1, fill_value=0.0, bounds_error=False)

            flux_k = eps_int(r_grid * (1 + z)) * rec['f'][k + 2]
            flux += np.moveaxis(flux_k, 2, 0)

        # Compute rho_alpha for this redshift
        rho_alpha_ = flux / (4 * np.pi * r_grid ** 2)[:, None, None]
        rho_alpha[..., i] = rho_alpha_ * (h0 / cm_per_Mpc) ** 2 / sec_per_year  # [pcm-2.s-1.Hz-1]

    return rho_alpha



def cum_optical_depth(zz, E, parameters: Parameters):
    """Compute the cumulative optical depth for photons traveling from emitters.

    The function computes the optical depth tau(z) for a photon observed
    at the first redshift in ``zz`` with energy ``E`` and integrated
    over the provided redshift range. It is used by the X-ray heating
    and ionisation calculations.

    Args:
        zz (array): Redshift array over which to integrate (increasing or decreasing order accepted).
        E (float|array): Observed photon energy at ``zz[0]`` in eV, or an
            array of energies.
        parameters (Parameters): Cosmology and composition parameters.

    Returns:
        numpy.ndarray: Optical depth array evaluated along ``zz``.
    """
    Ob = parameters.cosmology.Ob
    h0 = parameters.cosmology.h

    # Energy of a photon observed at (zz[0], E) and emitted at zz
    if isinstance(E, np.ndarray):
        Erest = E[:, None] * ((1 + zz)/(1 + zz[0]))[None, :]
    else:
        Erest = E * (1 + zz)/(1 + zz[0])

    #hydrogen and helium cross sections
    sHI = sigma_HI(Erest) * (h0/cm_per_Mpc)**2   #[Mpc/h]^2
    sHeI = sigma_HeI(Erest) * (h0/cm_per_Mpc)**2  #[Mpc/h]^2

    nb0 = rhoc0*Ob/(m_p_in_Msun*h0)                    # [h/Mpc]^3

    f_He_bynumb = 1 - parameters.cosmology.HI_frac

    #H and He abundances
    nHI = (1-f_He_bynumb)*nb0 *(1+zz)**3       # [h/Mpc]^3
    nHeI = f_He_bynumb * nb0 *(1+zz)**3

    #proper line element
    dldz = c_km_s*h0/hubble(zz, parameters)/(1+zz) # [Mpc/h]

    #integrate
    # logger.debug(f"{nHI.shape=}, {sHI.shape=}, {nHeI.shape=}, {sHeI.shape=}")
    if isinstance(E, np.ndarray):
        tau_int = dldz[None, :] * (nHI[None, :] * sHI + nHeI[None, :] * sHeI)
    else:
        tau_int = dldz * (nHI*sHI + nHeI*sHeI)
    # logger.debug(f"{dldz.shape=}, {tau_int.shape=}")
    # make sure to integrate along z (axis=-1)
    tau = cumulative_trapezoid(tau_int, x=zz, initial = 0.0, axis = -1)

    return tau

