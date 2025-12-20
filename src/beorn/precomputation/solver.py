"""1D radiation profile solver used to create inputs for the painter.

This module contains :class:`RadiationProfileSolver` which computes the
radial emission, heating and ionization profiles for a matrix of halo
masses and accretion rates. The results are stored in a
:class:`RadiationProfiles` dataclass for later use by the painting
stage.
"""

import numpy as np
from scipy.integrate import trapezoid, solve_ivp
from scipy.interpolate import interp1d
import logging
logger = logging.getLogger(__name__)
try:
    from mpi4py import MPI
    MPI_ENABLED = True
except RuntimeError:
    # mpi fails to import because the host system does not have it installed
    MPI_ENABLED = False

from ..cosmo import comoving_distance, hubble
from ..cross_sections import alpha_HII
from ..cross_sections import sigma_HI, sigma_HeI
from ..io import Handler
from ..structs.parameters import Parameters
from ..structs.radiation_profiles import RadiationProfiles
from .. import constants
# TODO: replace these unit conversions by astropy units
from ..constants import m_p_in_Msun, km_per_Mpc, h_eV_sec, cm_per_Mpc, E_HI, E_HeI, kb_eV_per_K, rhoc0
from ..astro import f_Xh, f_star_Halo, eps_xray
from .helpers import Ngdot_ion, mean_gamma_ion_xray, solve_xe, rho_alpha_profile, cum_optical_depth
from .massaccretion import mass_accretion


class RadiationProfileSolver:
    """Compute radiation, heating and ionization profiles around sources.

    The solver produces arrays of X-ray emissivity, heating rates and
    Lyman-alpha emissivity as a function of radius for a set of halo
    masses and accretion parameters. Results are returned as a
    :class:`RadiationProfiles` instance.
    """

    def __init__(self, parameters: Parameters, redshifts: np.ndarray):
        """
        Args:
            parameters: Parameters dataclass
            redshifts: array of redshifts at which to compute the profiles, in decreasing order
        """

        # self.z_initial = parameters.solver.Z  # starting redshift
        # if self.z_initial < 35:
        #     # TODO: add this warning as a validator of the parameters dataclass
        #     logger.warning('z_start (parameters.solver.zmax) should be larger than 35.')

        # TODO remove hardcoded values
        rmin = 1e-2
        rmax = 600
        Nr = 200
        self.r_grid = np.logspace(np.log10(rmin), np.log10(rmax), Nr) ##cMpc/h
        self.parameters = parameters
        self.z_bins = redshifts


    def get_or_compute_profiles(self, handler: Handler) -> RadiationProfiles:
        """Load profiles from cache or compute and save them if unavailable.

        Args:
            handler (Handler): IO handler capable of loading/saving
                :class:`RadiationProfiles` objects.

        Returns:
            RadiationProfiles: Loaded or freshly computed profiles.
        """
        try:
            profiles = handler.load_file(self.parameters, RadiationProfiles)
            logger.info("Loaded radiation profiles from cache.")
            return profiles
        except FileNotFoundError:
            logger.info("Radiation profiles not found in cache. Launching a single computation process.")

        # we need to consider that this is likely being run in MPI mode. Only a single rank should perform the computation and save the results, others should wait and then load the results
        if MPI_ENABLED:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            if rank == 0:
                profiles = self.solve()
                handler.write_file(self.parameters, profiles)
                logger.info(f"Rank {rank} computed profiles and saved them to cache.")
                # notify other ranks that computation is done
                comm.Barrier()
            else:
                # wait for rank 0 to finish computation and saving
                comm.Barrier()
                profiles = handler.load_file(self.parameters, RadiationProfiles)
                logger.info(f"Rank {rank} loaded radiation profiles from cache after computation by rank 0.")
        else:
            profiles = self.solve()
            handler.write_file(self.parameters, profiles)
            logger.info("Radiation profiles computed and saved to cache.")
        return profiles


    def solve(self) -> RadiationProfiles:
        # if MPI is being used, only compute on rank 0
        if MPI_ENABLED:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            if rank != 0:
                logger.warning(f"RadiationProfileSolver.solve() called on non-root MPI rank {rank=}. Only the root rank will perform the computation.")
                return

        # we will compute the profiles for specific values of M and dM/dt. Later on we will assume that the profiles are the same for all halos in a bin around these values
        # so we need both: the values at the center of the bin and the values at the edges of the bin
        halo_mass_bins, _ = mass_accretion(
            self.parameters,
            self.z_bins,
            self.parameters.simulation.halo_mass_bins,
            self.parameters.simulation.halo_mass_accretion_alpha
        )
        # the halo mass for the bin centers is the one used throughout the profile computation
        halo_mass, halo_mass_derivative = mass_accretion(
            self.parameters,
            self.z_bins,
            self.parameters.simulation.halo_mass_bin_centers,
            self.parameters.simulation.halo_mass_accreation_alpha_bin_centers
        )
        self.halo_mass_evolution = halo_mass
        self.halo_mass_derivative = halo_mass_derivative
        # both arrays have shape [mass_bins_center, alpha_bins_center, z_bins]

        if self.parameters.solver.fXh == 'constant':
            logger.info('param.solver.fXh is set to constant. We will assume f_X,h = 2e-4**0.225')
            x_e = np.full(len(self.z_bins), 2e-4)
        else:
            zz_, sfrd_ = global_qty.compute_sfrd(self.parameters, self.z_bins, halo_mass, halo_mass_derivative)
            sfrd = np.interp(self.z_bins, zz_, sfrd_, right=0)
            Gamma_ion, Gamma_sec_ion = mean_gamma_ion_xray(self.parameters, sfrd, self.z_bins)

            x_e = solve_xe(self.parameters, Gamma_ion, Gamma_sec_ion, self.z_bins)
            logger.info('param.solver.fXh is not set to constant. We will compute the free e- fraction x_e and assume fXh = x_e**0.225.')

        logger.info(f"Computing profiles for {self.z_bins.size} redshifts, {self.parameters.simulation.halo_mass_bins.size - 1} halo mass bins and {self.parameters.simulation.halo_mass_accretion_alpha.size - 1} alpha bins.")
        r_bubble = self.R_bubble()

        rho_xray = self.rho_xray(self.r_grid, x_e)
        rho_heat = self.rho_heat(rho_xray)

        r_lyal = np.logspace(-5, 2, 1000, base=10)
        # TODO - recheck this:
        # physical distance for lyal profile. Never goes further away than 10**2 = 100 pMpc/h (checked)

        rho_alpha = rho_alpha_profile(self.parameters, self.z_bins, r_lyal, halo_mass, halo_mass_derivative)
        logger.debug(f"Results have shapes: {rho_xray.shape=}, {rho_heat.shape=}, {r_bubble.shape=}, {r_lyal.shape=}, {rho_alpha.shape=}")

        return RadiationProfiles(
            parameters = self.parameters,
            z_history = self.z_bins,
            halo_mass_bins = halo_mass_bins,
            rho_xray = rho_xray,
            rho_heat = rho_heat,
            rho_alpha = rho_alpha,
            R_bubble = r_bubble,
            r_lyal = r_lyal,
            r_grid_cell = self.r_grid,
        )



    def R_bubble(self):
        """Compute the ionized bubble radius evolution for each mass/alpha bin.

        Returns:
            numpy.ndarray: Bubble radii (comoving cMpc/h) with shape
            (mass_bins-1, alpha_bins-1, len(z_bins)).
        """

        Ngam_dot = Ngdot_ion(
            self.parameters,
            self.z_bins[None, None, :],
            # brought to the same shape as the mass arrays
            self.halo_mass_evolution,
            self.halo_mass_derivative
        )  # s-1
        assert np.all(np.isfinite(Ngam_dot)), "Ngam_dot contains NaN values. Check the parameters and the mass accretion."
        Ob, h0 = self.parameters.cosmology.Ob, self.parameters.cosmology.h

        # \bar{n}^0_H - mean comoving number density of baryons [Mpc/h]**-3
        baryon_density = (Ob * constants.rhoc0) / (constants.m_p_in_Msun * h0)
        # scale factors corresponding to the redshifts
        scale_factors = 1 / (self.z_bins + 1)
        # b_0(z) - physical baryon density
        physical_baryon_density = baryon_density / scale_factors** 3
        # clumping factor
        clumping_factor = self.parameters.cosmology.clumping

        nb0_interp  = interp1d(scale_factors, physical_baryon_density, fill_value = 'extrapolate')
        Ngam_interp = interp1d(scale_factors, Ngam_dot, axis = -1, fill_value = 'extrapolate')

        def volume_derivative(a, volume):
            z = 1 / a - 1
            photon_number = Ngam_interp(a)
            baryon_number = nb0_interp(a)
            # logger.debug(f"{photon_number.shape=}, {volume.shape=}")
            volume = volume.reshape(photon_number.shape)
            return km_per_Mpc / (hubble(z, self.parameters) * a) * (photon_number / baryon_density - alpha_HII(1e4) * clumping_factor / cm_per_Mpc ** 3 * h0 ** 3 * baryon_number * volume).flatten()  # eq 65 from barkana and loeb

        # the time dependence will be given by the redshifts (added later)
        volume_shape = (self.parameters.simulation.halo_mass_bin_n - 1, len(self.parameters.simulation.halo_mass_accretion_alpha) - 1)
        v0 = np.zeros(volume_shape)

        sol = solve_ivp(
            volume_derivative,
            t_span = [scale_factors[0], scale_factors[-1]],
            y0 = v0.flatten(),
            t_eval = scale_factors
        )
        bubble_volume = sol.y
        bubble_volume.clip(min = 0, out = bubble_volume)

        # since solve_ivp works with 1d arrays we have a flattened version currently, where the last axis is the "time"
        # even though the computation was made using the scale factors they have the same order as the redshifts. We keep them in the last axis, to match the other profiles
        bubble_volume = bubble_volume.reshape((*volume_shape, scale_factors.size))
        bubble_radius = (bubble_volume * 3 / (4 * np.pi)) ** (1 / 3)
        return bubble_radius



    def rho_xray(self, rr: np.ndarray, xe: np.ndarray):
        """Compute X-ray energy deposition profiles as a function of radius.

        Args:
            rr (np.ndarray): Radial grid (comoving cMpc/h) where profiles are evaluated.
            xe (np.ndarray): Free electron fraction history corresponding to ``self.z_bins``.

        Returns:
            numpy.ndarray: X-ray heating / energy deposition arrays with
            shape (len(rr), mass_bins-1, alpha_bins-1, len(z_bins)).
        """

        Om = self.parameters.cosmology.Om
        Ob = self.parameters.cosmology.Ob
        h0 = self.parameters.cosmology.h
        # TODO: remove hardcoded values
        z_star = 35
        Emin = self.parameters.source.energy_cutoff_min_xray
        Emax = self.parameters.source.energy_cutoff_max_xray
        NE = 50

        nb0 = rhoc0 * Ob / (m_p_in_Msun * h0)  # [h/Mpc]^3

        # zprime binning
        dz_prime = 0.1

        # define frequency bin
        nu_min = Emin / constants.h_eV_sec
        nu_max = Emax / constants.h_eV_sec
        N_mu = NE
        nu = np.logspace(np.log(nu_min), np.log(nu_max), N_mu, base=np.e)

        f_He_bynumb = 1 - self.parameters.cosmology.HI_frac
        # hydrogen
        nH0 = (1-f_He_bynumb) * nb0
        # helium
        nHe0 = f_He_bynumb * nb0

        M_star_dot = (Ob / Om) * f_star_Halo(self.parameters, self.halo_mass_evolution) * self.halo_mass_derivative
        M_star_dot[np.where(self.halo_mass_evolution < self.parameters.source.halo_mass_min)] = 0

        # compute the N prime array before hand
        # TODO what is N prime?
        # for this computation we consider the maximum redshift to be z_star
        z_range = z_star - self.z_bins
        N_prime = z_range / dz_prime
        # we cast to int later on because this gives the number of points
        N_prime = np.maximum(N_prime, 4).astype(int) # TODO explain why 4 exactly

        rho_xray = np.zeros((len(rr), self.parameters.simulation.halo_mass_bin_n - 1, len(self.parameters.simulation.halo_mass_accretion_alpha) - 1, len(self.z_bins)))

        for i, z in enumerate(self.z_bins):
            # it only makes sense to compute the profile for z < zstar
            if z > z_star:
                continue

            # lookback redshift
            z_prime = np.logspace(np.log(z), np.log(z_star), N_prime[i], base=np.e)
            rcom_prime = comoving_distance(z_prime, self.parameters) * h0  # comoving distance

            # TODO: why do we interpolate here if we have the analytical expression?
            if i == 0: # if zz[0]<zstar, then concatenate two numbers..
                dMdt_int = interp1d(
                    x = np.concatenate(([z_star], self.z_bins[:i+1])),
                    y = np.stack(
                        (
                            np.zeros_like(M_star_dot[..., 0]),
                            M_star_dot[..., 0]
                        ),
                        axis = -1
                    ),
                    axis = -1,
                    fill_value='extrapolate'
                )
            else:
                dMdt_int = interp1d(
                    x = self.z_bins[:i + 1],
                    y = M_star_dot[..., :i+1],
                    axis = -1,
                    fill_value = 'extrapolate'
                )

            # as described in the paper, we express the emission of xrays as a function of distance
            # this is precomputed for a range of parameters: alpha, Mh, z
            # the main component of the emission is given by an integral over the frequency
            # to compute the integral we prepend the nu dependence as the first axis of the flux array (flux[nu, r, Mh, alpha])

            def integrand(nu_val: np.ndarray):
                # In the following we will always keep the nu_val in the 0th axis of the resulting array

                tau_prime = cum_optical_depth(z_prime, nu_val * constants.h_eV_sec, self.parameters)
                # tau_prime has a redshift component and the nu component (2d)

                nu_prime = nu_val[:, None] * ((1 + z_prime) / (1 + z))[None, :]
                eps_X = eps_xray(nu_prime, self.parameters)
                # eps_X has a redshift component and the nu component (2d)

                # complication - the integrand is expressed in terms of the radial distance
                # we perform a hack to interpret eps(z) as a function of r
                # the mass also had an M0, alpha dependence, so we the missing axis
                integral_factors = (np.exp(- tau_prime) * eps_X)[:,  None, None, :] * dMdt_int(z_prime)[None, ...]
                integral_factors_interpolated = interp1d(rcom_prime, integral_factors, axis=-1, fill_value=0.0, bounds_error=False)
                # but r should be the first axis after nu: 0, 1, 2, 3 -> 0, 3, 1, 2
                integral_factors_r = integral_factors_interpolated(rr)
                integral_factors_r = np.moveaxis(integral_factors_r, -1, 1)

                # the final integrand is a function of the frequency and the radial distance
                prefactor = ((nH0 / nb0) * sigma_HI(nu_val * h_eV_sec) * (nu_val * h_eV_sec - E_HI) + (nHe0 / nb0) * sigma_HeI(nu_val * h_eV_sec) * (nu_val * h_eV_sec - E_HeI))   # [cm^2 * eV] 4 * np.pi *
                return prefactor[:, None, None, None] * integral_factors_r

            integrated_flux = trapezoid(integrand(nu), nu, axis=0)
            heat = integrated_flux
            fXh = f_Xh(xe[i])
            rho = fXh * 1 / (4 * np.pi * (rr/(1+z)) ** 2)[:, None, None] * heat / (cm_per_Mpc/h0) ** 2
            # logger.debug(f"{fXh.shape=}, {rr.shape=}, {nu.shape=}, {rho.shape=}")
            rho_xray[..., i] = rho

        return rho_xray


    def rho_heat(self, rho_xray: np.ndarray):
        """Solve the temperature evolution from X-ray heating rates.

        The routine integrates the heating equation over scale factor to
        produce a temperature profile from the supplied heating rate
        ``rho_xray``.

        Args:
            rho_xray (np.ndarray): Output of :meth:`rho_xray`.

        Returns:
            numpy.ndarray: Temperature profiles with the same mass/alpha/z
            axes as the inputs.
        """
        # add the decoupling redshift as "initial condition"
        z0 = self.parameters.cosmology.z_decoupling
        zz = np.insert(self.z_bins, 0, z0)

        # prepend 0 to the rho_xray array to account for the additional z bin
        rho_xray = np.concatenate((np.zeros_like(rho_xray[..., 0])[..., None], rho_xray), axis=-1)

        # the shape of the xray profile at a given redshift is:
        # (rr, M_bin, alpha_bin)
        single_rho_xray_shape = rho_xray[..., 0].shape

        # scale factor
        aa = 1 / (1 + zz)
        # allow us to query the xray profile at any scale factor (<-> redshift)
        rho_interpolated = interp1d(aa, rho_xray, axis=-1, fill_value="extrapolate")

        def right_hand_side(a, y):
            # since solve_ivp works with 1d arrays y is currently flattened
            gamma_heat = 2 * rho_interpolated(a) / (3 * kb_eV_per_K * a * hubble(1 / a - 1, self.parameters)) * km_per_Mpc
            return gamma_heat.flatten() - 2 * y / a

        y0 = np.zeros(single_rho_xray_shape)
        result = solve_ivp(right_hand_side, [aa[0], aa[-1]], y0.flatten(), t_eval=aa)#, atol=1e-2, rtol=1e-2)
        source_in_time = result.y
        # don't keep the initial condition at the first time step (time is in the last axis)
        # logger.debug(f"{source_in_time.shape=}")
        rho_heat = source_in_time[..., 1:]

        # currently rho_heat has the shape (rr * M_bin * alpha_bin, zz) because all dimensions are flattened (except redshift)
        # we need to reshape it to (rr, M_bin, alpha_bin, zz)
        rho_heat_full = rho_heat.reshape((*single_rho_xray_shape, -1))

        return rho_heat_full
