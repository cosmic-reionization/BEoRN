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

from ..cosmo import comoving_distance, hubble, hubble_per_yr
from ..cross_sections import alpha_HII
from ..cross_sections import sigma_HI, sigma_HeI
from ..io import Handler
from ..structs.parameters import Parameters
from ..structs.radiation_profiles import RadiationProfiles, RadiationProfilesFStarGrid
from .. import constants
# TODO: replace these unit conversions by astropy units
from ..constants import m_p_in_Msun, km_per_Mpc, h_eV_sec, cm_per_Mpc, E_HI, E_HeI, kb_eV_per_K, rhoc0
from ..astro import f_Xh, f_star_Halo, eps_xray
from .helpers import Ngdot_ion, rho_alpha_profile, cum_optical_depth
from .massaccretion import mass_accretion
from copy import deepcopy
logger = logging.getLogger(__name__)
try:
    from mpi4py import MPI
    MPI_ENABLED = True
except (ImportError, RuntimeError):
    MPI_ENABLED = False


class RadiationProfileSolver:
    def profile_cache_namespace(self) -> str:
        """Return the cache namespace for the radiation profiles."""
        return "radiation_profiles_legacy"
    
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
            redshifts: array of redshifts at which to compute the profiles, in decreasing order.
                If not already in decreasing order, it is sorted (a warning is logged) --
                :meth:`rho_heat` integrates over scale factor and requires it, since it
                prepends ``solver.z_decoupling`` (the earliest, highest-z point) and relies
                on the resulting scale-factor array increasing monotonically end to end;
                an out-of-order input desyncs ``t_span``/``t_eval`` and ``solve_ivp`` raises.
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

        redshifts = np.asarray(redshifts, dtype=float)
        if redshifts.size > 1 and not np.all(np.diff(redshifts) <= 0):
            logger.warning(
                f"RadiationProfileSolver requires redshifts in decreasing order, but got "
                f"z={redshifts[0]:.3f}→{redshifts[-1]:.3f} "
                f"({redshifts.size} points). Sorting descending."
            )
            redshifts = np.sort(redshifts)[::-1]
        self.z_bins = redshifts


    def get_or_compute_profiles(self, handler: Handler, force_recompute: bool = False) -> RadiationProfiles:
        """Load profiles from cache or compute and save them if unavailable.

        Args:
            handler (Handler): IO handler capable of loading/saving
                :class:`RadiationProfiles` objects.
            force_recompute (bool): If ``True``, ignore any cached profiles and
                recompute from scratch.  Useful after changing mass bin range or
                other profile-affecting parameters.  Default ``False``.

        Returns:
            RadiationProfiles: Loaded or freshly computed profiles.
        """
        if not force_recompute:
            try:
                profiles = handler.load_file(self.parameters, RadiationProfiles, cache_namespace=self.profile_cache_namespace())
                # Validate that the cached redshift grid matches what is currently requested.
                # A hash mismatch would already produce FileNotFoundError above, but an
                # explicit check here gives a clear diagnostic if the cache is somehow stale.
                if not np.array_equal(profiles.z_history, self.z_bins):
                    raise ValueError(
                        f"Cached radiation profiles cover "
                        f"z={profiles.z_history[0]:.2f}→{profiles.z_history[-1]:.2f} "
                        f"({len(profiles.z_history)} steps) but solver.redshifts requests "
                        f"z={self.z_bins[0]:.2f}→{self.z_bins[-1]:.2f} "
                        f"({len(self.z_bins)} steps). "
                        "Delete the cached file or use force_recompute=True."
                    )
                logger.info(
                    f"Loaded radiation profiles from cache "
                    f"(z={profiles.z_history[0]:.2f}→{profiles.z_history[-1]:.2f}, "
                    f"{len(profiles.z_history)} steps)."
                )
                return profiles
            except FileNotFoundError:
                logger.info("Radiation profiles not found in cache. Launching a single computation process.")
        else:
            logger.info("force_recompute=True — recomputing radiation profiles.")
            logger.info(self.parameters.summary_str())

        # we need to consider that this is likely being run in MPI mode. Only a single rank should perform the computation and save the results, others should wait and then load the results
        # Sections that do not affect 1D profile shapes — excluded from the
        # sidecar YAML so it only documents what the profiles actually depend on.
        _PROFILES_YAML_EXCLUDE = {"simulation", "cosmo_sim"}

        if MPI_ENABLED:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            if rank == 0:
                try:
                    profiles = self.solve()
                    handler.write_file(self.parameters, profiles, cache_namespace=self.profile_cache_namespace())
                    if profiles._file_path is not None:
                        self.parameters.to_yaml(
                            profiles._file_path.with_suffix('.yaml'),
                            exclude_keys=_PROFILES_YAML_EXCLUDE,
                        )
                    logger.info(f"Rank {rank} computed profiles and saved them to cache.")
                    # notify other ranks that computation is done
                    comm.Barrier()
                except Exception:
                    logger.exception(
                        "Rank 0 failed during profile computation. "
                        "Calling MPI_Abort to unblock other ranks."
                    )
                    comm.Abort(1)
            else:
                # wait for rank 0 to finish computation and saving
                comm.Barrier()
                profiles = handler.load_file(self.parameters, RadiationProfiles, cache_namespace=self.profile_cache_namespace())
                logger.info(f"Rank {rank} loaded radiation profiles from cache after computation by rank 0.")
        else:
            profiles = self.solve()
            handler.write_file(self.parameters, profiles, cache_namespace=self.profile_cache_namespace())
            if profiles._file_path is not None:
                self.parameters.to_yaml(
                    profiles._file_path.with_suffix('.yaml'),
                    exclude_keys=_PROFILES_YAML_EXCLUDE,
                )
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
            self.parameters.solver.halo_mass_bins,
            self.parameters.solver.halo_mass_accretion_alpha
        )
        # the halo mass for the bin centers is the one used throughout the profile computation
        halo_mass, halo_mass_derivative = mass_accretion(
            self.parameters,
            self.z_bins,
            self.parameters.solver.halo_mass_bin_centers,
            self.parameters.solver.halo_mass_accretion_alpha_bin_centers
        )
        self.halo_mass_evolution = halo_mass
        self.halo_mass_derivative = halo_mass_derivative
        # both arrays have shape [mass_bins_center, alpha_bins_center, z_bins]

        if self.parameters.solver.fXh == 'constant':
            logger.info('param.solver.fXh is set to constant. We will assume f_X,h = 2e-4**0.225')
            x_e = np.full(len(self.z_bins), 2e-4)
        else:
            raise NotImplementedError(
                "Computing x_e from the SFRD (parameters.solver.fXh != 'constant') is not yet implemented. "
                "The old global_qty.compute_sfrd() was removed during the v2 cleanup. "
                "Set parameters.solver.fXh = 'constant' to use the default approximation (x_e = 2e-4)."
            )

        logger.info(f"Computing profiles for {self.z_bins.size} redshifts, {self.parameters.solver.halo_mass_bins.size - 1} halo mass bins and {self.parameters.solver.halo_mass_accretion_alpha.size - 1} alpha bins.")
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



    def _z_star_at(self, z: float) -> float:
        """Return the maximum lookback redshift for X-ray integration at redshift *z*.

        When ``parameters.source.source_age`` is ``None`` (default), the
        original hardcoded value of 35 is used.  Otherwise the lookback
        window is capped at ``source_age`` Myr by integrating
        dt = dz / ((1+z) H(z)) backward from *z*.

        Args:
            z (float): Current redshift at which the profile is evaluated.

        Returns:
            float: Maximum redshift to integrate back to (z_star).
        """
        z_source_start = self.parameters.solver.z_source_start
        t_source_age   = self.parameters.source.t_source_age
        if t_source_age is None:
            return z_source_start

        # Build a fine redshift grid from z up to z_source_start and integrate
        # dt/dz = 1 / ((1+z') H(z')) to find where the accumulated lookback
        # time equals t_source_age.
        z_grid = np.linspace(z, z_source_start, 5000)
        # H(z) in yr⁻¹ → dt in Myr per dz step
        dt_dz  = 1e-6 / ((1 + z_grid) * hubble_per_yr(z_grid, self.parameters))  # Myr per unit z
        dz     = np.diff(z_grid)
        cumulative_age = np.concatenate(([0.0], np.cumsum(0.5 * (dt_dz[:-1] + dt_dz[1:]) * dz)))

        # Find the redshift where the cumulative lookback time reaches t_source_age
        idx = np.searchsorted(cumulative_age, t_source_age)
        if idx >= len(z_grid):
            return z_source_start
        return float(z_grid[idx])

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
        Ob, h0 = self.parameters.cosmology.Ob, self.parameters.cosmology.h0

        # \bar{n}^0_H - mean comoving number density of baryons [Mpc/h]**-3
        baryon_density = (Ob * constants.rhoc0) / (constants.m_p_in_Msun * h0)
        # scale factors corresponding to the redshifts
        scale_factors = 1 / (self.z_bins + 1)
        # clumping factor
        clumping_factor = self.parameters.solver.clumping

        # physical_baryon_density = baryon_density / a**3 is an exact analytic
        # formula; computing it via interp1d of pre-tabulated values introduces
        # numerical fragility for large baryon_density values (~2e67 in code
        # units) and can return NaN at the boundary in some environments.
        # We therefore compute it analytically inside volume_derivative.
        Ngam_interp = interp1d(scale_factors, Ngam_dot, axis = -1, fill_value = 'extrapolate')

        def volume_derivative(a, volume):
            z = 1 / a - 1
            photon_number = Ngam_interp(a)
            baryon_number = baryon_density / a ** 3   # physical baryon number density
            # logger.debug(f"{photon_number.shape=}, {volume.shape=}")
            volume = volume.reshape(photon_number.shape)
            return km_per_Mpc / (hubble(z, self.parameters) * a) * (photon_number / baryon_density - alpha_HII(1e4) * clumping_factor / cm_per_Mpc ** 3 * h0 ** 3 * baryon_number * volume).flatten()  # eq 65 from barkana and loeb

        # the time dependence will be given by the redshifts (added later)
        volume_shape = (self.parameters.solver.halo_mass_nbin - 1, len(self.parameters.solver.halo_mass_accretion_alpha) - 1)
        v0 = np.zeros(volume_shape)

        rtol   = self.parameters.solver.ode_rtol
        atol   = self.parameters.solver.ode_atol
        method = self.parameters.solver.ode_method
        sol = solve_ivp(
            volume_derivative,
            t_span = [scale_factors[0], scale_factors[-1]],
            y0 = v0.flatten(),
            t_eval = scale_factors,
            method = method,
            rtol = rtol,
            atol = atol,
        )
        if not sol.success or not isinstance(sol.y, np.ndarray):
            mass_centers = self.parameters.solver.halo_mass_bin_centers
            alpha_centers = self.parameters.solver.halo_mass_accretion_alpha_bin_centers

            # ── component-level diagnostics at t0 ──────────────────────────
            a0 = scale_factors[0]
            z0 = self.z_bins[0]
            photon0   = Ngam_interp(a0)       # (mass, alpha) at first scale factor
            baryonN0  = baryon_density / a0 ** 3
            H0_val    = hubble(float(z0), self.parameters)
            prefactor = km_per_Mpc / (H0_val * a0)

            # locate bins where any component is non-finite
            phot_bad  = ~np.isfinite(photon0)   # shape (mass, alpha)
            Ngam_nan_zslices = ~np.isfinite(Ngam_dot)  # shape (mass, alpha, z)
            Ngam_nan_any = Ngam_nan_zslices.any(axis=-1)  # (mass, alpha)

            diag_lines = [
                f"  scale_factors monotone-increasing:  {bool(np.all(np.diff(scale_factors) > 0))}",
                f"  scale_factors finite:               {bool(np.all(np.isfinite(scale_factors)))}",
                f"  hubble(z={z0:.3f}):                 {H0_val:.4e}",
                f"  prefactor km_per_Mpc/(H*a):         {prefactor:.4e}",
                f"  baryon_density (comoving):          {baryon_density:.4e}",
                f"  baryon_density/a0^3 (physical):     {baryonN0:.4e}",
                f"  Ngam_dot all-finite:                {bool(np.isfinite(Ngam_dot).all())}",
                f"  Ngam_interp(a0) all-finite:         {bool(np.isfinite(photon0).all())}",
                f"  # bins where Ngam_interp(a0) is NaN:        {int(phot_bad.sum())}",
                f"  # bins where Ngam_dot has ANY NaN z-slice:  {int(Ngam_nan_any.sum())}",
            ]

            # detail for first few bad bins
            bad_indices = list(zip(*np.where(phot_bad | Ngam_nan_any)))[:8]
            for mi, ai in bad_indices:
                ngam_z = Ngam_dot[mi, ai, :]
                nan_zidx = np.where(~np.isfinite(ngam_z))[0]
                diag_lines.append(
                    f"  BAD bin mass={mass_centers[mi]:.3e}  alpha={alpha_centers[ai]:.4f}  "
                    f"Ngam_interp(a0)={photon0[mi, ai]:.3e}  "
                    f"Ngam_dot[0]={ngam_z[0]:.3e}  "
                    f"Ngam_dot NaN at z-idx={nan_zidx.tolist()}  "
                    f"Ngam_dot range=[{ngam_z[np.isfinite(ngam_z)].min() if np.isfinite(ngam_z).any() else float('nan'):.3e}, "
                    f"{ngam_z[np.isfinite(ngam_z)].max() if np.isfinite(ngam_z).any() else float('nan'):.3e}]"
                )

            logger.warning(
                "R_bubble: ODE solver failed (status=%s, nfev=%s): %s  "
                "Retrying with LSODA.\nDiagnostics:\n%s",
                sol.status, getattr(sol, 'nfev', '?'), sol.message,
                '\n'.join(diag_lines),
            )
            sol = solve_ivp(
                volume_derivative,
                t_span = [scale_factors[0], scale_factors[-1]],
                y0 = v0.flatten(),
                t_eval = scale_factors,
                method = 'LSODA',
            )
        if not sol.success or not isinstance(sol.y, np.ndarray):
            raise RuntimeError(
                f"R_bubble ODE solver failed with both RK45 and LSODA "
                f"(status={sol.status}, nfev={getattr(sol, 'nfev', '?')}): {sol.message}"
            )
        bubble_volume = sol.y
        if not np.isfinite(bubble_volume).all():
            nan_count = (~np.isfinite(bubble_volume)).sum()
            raise RuntimeError(
                f"R_bubble: ODE solver ({sol.status=}) returned {nan_count} "
                f"non-finite values despite reporting success. "
                f"Check the diagnostic WARNING above for the NaN source."
            )
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
        h0 = self.parameters.cosmology.h0
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

        f_He_bynumb = 1 - self.parameters.solver.HI_frac
        # hydrogen
        nH0 = (1-f_He_bynumb) * nb0
        # helium
        nHe0 = f_He_bynumb * nb0

        M_star_dot = (Ob / Om) * f_star_Halo(self.parameters, self.halo_mass_evolution) * self.halo_mass_derivative
        M_star_dot[np.where(self.halo_mass_evolution < self.parameters.source.halo_mass_min)] = 0

        rho_xray = np.zeros((len(rr), self.parameters.solver.halo_mass_nbin - 1, len(self.parameters.solver.halo_mass_accretion_alpha) - 1, len(self.z_bins)))

        # Build dMdt_int once over the full redshift history.
        # We anchor M_star_dot = 0 at z_source_start (no emission above the starting
        # redshift) and include all z_bins. z_prime is always queried in
        # [z_current, z_star_i], so the interpolator is never extrapolated below
        # the lowest z_bin.
        _z_anchor = self.parameters.solver.z_source_start
        dMdt_int = interp1d(
            x = np.concatenate(([_z_anchor], self.z_bins)),
            y = np.concatenate((np.zeros_like(M_star_dot[..., :1]), M_star_dot), axis=-1),
            axis = -1,
            fill_value = 'extrapolate',
        )

        for i, z in enumerate(self.z_bins):
            # Determine the maximum lookback redshift for this snapshot.
            # When source_age is set, this caps the integration window to the
            # halo's lifetime; otherwise the full history back to z=35 is used.
            z_star_i = self._z_star_at(z)

            if z > z_star_i:
                continue

            # Number of integration points over the lookback window [z, z_star_i]
            N_prime_i = max(4, int((z_star_i - z) / dz_prime))

            # lookback redshift grid for this snapshot
            z_prime = np.logspace(np.log(z), np.log(z_star_i), N_prime_i, base=np.e)
            rcom_prime = comoving_distance(z_prime, self.parameters) * h0  # comoving distance

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
        z0 = self.parameters.solver.z_decoupling
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

        rtol   = self.parameters.solver.ode_rtol
        atol   = self.parameters.solver.ode_atol
        method = self.parameters.solver.ode_method
        y0 = np.zeros(single_rho_xray_shape)
        result = solve_ivp(
            right_hand_side, [aa[0], aa[-1]], y0.flatten(),
            t_eval=aa, method=method, rtol=rtol, atol=atol,
        )
        if not result.success:
            logger.warning(
                "rho_heat ODE did not converge (rtol=%g, atol=%g): %s. "
                "Results may be inaccurate — tighten ode_rtol/ode_atol or check inputs.",
                rtol, atol, result.message,
            )
        source_in_time = result.y
        # don't keep the initial condition at the first time step (time is in the last axis)
        # logger.debug(f"{source_in_time.shape=}")
        rho_heat = source_in_time[..., 1:]

        # currently rho_heat has the shape (rr * M_bin * alpha_bin, zz) because all dimensions are flattened (except redshift)
        # we need to reshape it to (rr, M_bin, alpha_bin, zz)
        rho_heat_full = rho_heat.reshape((*single_rho_xray_shape, -1))

        return rho_heat_full

class RadiationProfileFstSolver(RadiationProfileSolver):
    def profile_cache_namespace(self) -> str:
        """Return the cache namespace for f_st-grid radiation profiles."""
        f_st_min = float(self.f_st_grid[0])
        f_st_max = float(self.f_st_grid[-1])
        f_st_n = int(self.f_st_grid.size)
        return f"radiation_profiles_fstar_grid_fmin_{f_st_min:.6g}_fmax_{f_st_max:.6g}_n_{f_st_n}"

    def get_or_compute_profiles(self, handler: Handler) -> "RadiationProfilesFStarGrid":
        """Load f_st-grid profiles from cache or compute and save them if unavailable."""
        try:
            profiles = handler.load_file(
                self.parameters,
                RadiationProfilesFStarGrid,
                cache_namespace=self.profile_cache_namespace(),
            )
            logger.info("Loaded f_st-grid radiation profiles from cache.")
            return profiles
        except FileNotFoundError:
            logger.info("f_st-grid radiation profiles not found in cache. Launching a single computation process.")

        if MPI_ENABLED:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            if rank == 0:
                try:
                    profiles = self.solve()
                    handler.write_file(
                        self.parameters,
                        profiles,
                        cache_namespace=self.profile_cache_namespace(),
                    )
                    logger.info(f"Rank {rank} computed f_st-grid profiles and saved them to cache.")
                    comm.Barrier()
                except Exception:
                    logger.exception("Rank 0 failed while computing f_st-grid radiation profiles.")
                    comm.Abort(1)
                    raise
            else:
                comm.Barrier()
                profiles = handler.load_file(
                    self.parameters,
                    RadiationProfilesFStarGrid,
                    cache_namespace=self.profile_cache_namespace(),
                )
                logger.info(f"Rank {rank} loaded f_st-grid radiation profiles from cache after computation by rank 0.")
        else:
            profiles = self.solve()
            handler.write_file(
                self.parameters,
                profiles,
                cache_namespace=self.profile_cache_namespace(),
            )
            logger.info("f_st-grid radiation profiles computed and saved to cache.")
        return profiles
    """Compute radiation profiles on a (mass, alpha, f_st, z) grid.

    This solver extends the legacy single-f_st solver without modifying its
    behavior. It precomputes one profile cube per f_st value and stacks the
    results into a :class:`RadiationProfilesFStarGrid` instance.
    """

    def __init__(self, parameters: Parameters, redshifts: np.ndarray):
        super().__init__(parameters, redshifts)
        self.f_st_grid = self._build_f_st_grid()

    def _build_f_st_grid(self) -> np.ndarray:
        source = self.parameters.source
        distribution = getattr(source, "f_st_paint_distribution", "lognormal").lower()
        f_st_center = float(source.f_st)
        f_st_min = float(getattr(source, 'f_st_grid_min', 0.01))
        f_st_max = float(getattr(source, 'f_st_grid_max', 1.0))
        f_st_n = int(getattr(source, 'f_st_grid_n', 31))
        if f_st_n < 2:
            raise ValueError("f_st_grid_n must be at least 2")
        if f_st_min <= 0 or f_st_max <= 0:
            raise ValueError("f_st grid values must be strictly positive")
        if f_st_max > 1.0:
            raise ValueError(f"f_st_grid_max={f_st_max} exceeds 1.0, which is unphysical for a stellar fraction")
        if f_st_min >= f_st_max:
            raise ValueError("f_st_grid_min must be smaller than f_st_grid_max")
        if distribution == "lognormal":
            k_star = (f_st_n - 1) // 2
            dlog = (np.log10(f_st_max) - np.log10(f_st_center)) / (f_st_n - 1 - k_star)
            log_min = np.log10(f_st_center) - k_star * dlog
            f_st_min = 10**log_min
            logger.info(f"Redefine f_st_min to {f_st_min:.4f}")
            logger.info("Return f_st_grid in log space")
            return np.logspace(np.log10(f_st_min), np.log10(f_st_max), f_st_n, base=10)
        return np.linspace(f_st_min, f_st_max, f_st_n)

    def _parameters_with_f_st(self, f_st: float) -> Parameters:
        parameters = deepcopy(self.parameters)
        parameters.source.f_st = float(f_st)
        return parameters

    def _compute_single_f_st_profiles(
        self,
        parameters: Parameters,
        x_e: np.ndarray,
        halo_mass: np.ndarray,
        halo_mass_derivative: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r_lyal = np.logspace(-5, 2, 1000, base=10)
        original_parameters = self.parameters
        try:
            self.parameters = parameters
            r_bubble = self.R_bubble()
            rho_xray = self.rho_xray(self.r_grid, x_e)
            rho_heat = self.rho_heat(rho_xray)
            rho_alpha = rho_alpha_profile(
                parameters,
                self.z_bins,
                r_lyal,
                halo_mass,
                halo_mass_derivative,
            )
        finally:
            self.parameters = original_parameters

        return r_bubble, rho_xray, rho_heat, rho_alpha

    def solve(self) -> RadiationProfilesFStarGrid:
        if MPI_ENABLED:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            if rank != 0:
                logger.warning(
                    f"RadiationProfileFstSolver.solve() called on non-root MPI rank {rank=}. Only the root rank will perform the computation."
                )
                return

        halo_mass_bins, _ = mass_accretion(
            self.parameters,
            self.z_bins,
            self.parameters.solver.halo_mass_bins,
            self.parameters.solver.halo_mass_accretion_alpha
        )

        halo_mass, halo_mass_derivative = mass_accretion(
            self.parameters,
            self.z_bins,
            self.parameters.solver.halo_mass_bin_centers,
            self.parameters.solver.halo_mass_accretion_alpha_bin_centers
        )

        self.halo_mass_evolution = halo_mass
        self.halo_mass_derivative = halo_mass_derivative

        if self.parameters.solver.fXh == 'constant':
            logger.info('param.solver.fXh is set to constant. We will assume f_X,h = 2e-4**0.225')
            x_e = np.full(len(self.z_bins), 2e-4)
        else:
            raise NotImplementedError(
                "Computing x_e from the SFRD (parameters.solver.fXh != 'constant') is not yet implemented. "
                "The old global_qty.compute_sfrd() was removed during the v2 cleanup. "
                "Set parameters.solver.fXh = 'constant' to use the default approximation (x_e = 2e-4)."
            )

        logger.info(
            f"Computing profiles for {self.z_bins.size} redshifts, "
            f"{self.parameters.solver.halo_mass_nbin - 1} halo mass bins, "
            f"{self.parameters.solver.halo_mass_accretion_alpha.size - 1} alpha bins and "
            f"{self.f_st_grid.size} f_st values."
        )

        mass_n = self.parameters.solver.halo_mass_nbin - 1
        alpha_n = len(self.parameters.solver.halo_mass_accretion_alpha) - 1
        z_n = len(self.z_bins)
        f_n = self.f_st_grid.size

        r_bubble = np.zeros((mass_n, alpha_n, f_n, z_n))
        rho_xray = np.zeros((len(self.r_grid), mass_n, alpha_n, f_n, z_n))
        rho_heat = np.zeros_like(rho_xray)
        r_lyal = np.logspace(-5, 2, 1000, base=10)
        rho_alpha = np.zeros((len(r_lyal), mass_n, alpha_n, f_n, z_n))

        for f_index, f_st in enumerate(self.f_st_grid):
            logger.info("Computing radiation profiles for f_st=%.5f (%d/%d)", f_st, f_index + 1, f_n)
            parameters_f_st = self._parameters_with_f_st(f_st)

            rb, rx, rh, ra = self._compute_single_f_st_profiles(
                parameters_f_st,
                x_e,
                halo_mass,
                halo_mass_derivative,
            )

            r_bubble[:, :, f_index, :] = rb
            rho_xray[:, :, :, f_index, :] = rx
            rho_heat[:, :, :, f_index, :] = rh
            rho_alpha[:, :, :, f_index, :] = ra

        return RadiationProfilesFStarGrid(
            parameters=self.parameters,
            z_history=self.z_bins,
            halo_mass_bins=halo_mass_bins,
            f_st_grid=self.f_st_grid,
            rho_xray=rho_xray,
            rho_heat=rho_heat,
            rho_alpha=rho_alpha,
            R_bubble=r_bubble,
            r_lyal=r_lyal,
            r_grid_cell=self.r_grid,
        )
