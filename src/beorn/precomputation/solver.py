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

from ..cosmo import comoving_distance, hubble, Hubble
from ..cross_sections import alpha_HII
from ..cross_sections import sigma_HI, sigma_HeI
from ..io import Handler
from ..structs.parameters import Parameters
from ..structs.radiation_profiles import RadiationProfiles, RadiationProfilesFStarGrid
from .. import constants
# TODO: replace these unit conversions by astropy units
from ..constants import m_p_in_Msun, km_per_Mpc, h_eV_sec, cm_per_Mpc, E_HI, E_HeI, kb_eV_per_K, rhoc0
from ..astro import f_Xh, f_star_Halo, eps_xray
from .helpers import (Ngdot_ion, rho_alpha_profile, rho_alpha_profile_direct, cum_optical_depth,
                      emission_history_interpolator, emission_rate_fn)
from .massaccretion import mass_accretion, trajectory_at
from copy import deepcopy
logger = logging.getLogger(__name__)
try:
    from mpi4py import MPI
    MPI_ENABLED = True
except (ImportError, RuntimeError):
    MPI_ENABLED = False


def _is_legacy(parameters) -> bool:
    """Legacy numerics? Objects without the v2 switch (e.g. test doubles) are treated as legacy."""
    return bool(getattr(parameters.solver, "is_legacy", True))


def _profile_integrator(parameters) -> str:
    """v2 lookback-integral scheme: 'direct' (step B, default) or 'loop' (first v2 version)."""
    return getattr(parameters.solver, "profile_integrator", "loop")


class RadiationProfileSolver:
    """Compute radiation, heating and ionization profiles around sources.

    The solver produces arrays of X-ray emissivity, heating rates and
    Lyman-alpha emissivity as a function of radius for a set of halo
    masses and accretion parameters. Results are returned as a
    :class:`RadiationProfiles` instance.
    """

    def profile_cache_namespace(self) -> str:
        """Return the cache namespace for the radiation profiles."""
        return "radiation_profiles_legacy"
    
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
        # halo_mass_bins holds the *mass bin edges* traced back in redshift, and is what the
        # painter compares a halo's mass against to pick its mass bin.  Its alpha axis must be
        # the bin CENTRES, not the edges: the painter indexes it and the profiles below with
        # the same alpha_index (0..n_alpha-1), so an edges-length axis would silently hand it
        # the bin's lower edge while the profile it applies was built at the centre.  That
        # mismatch inflates every boundary by exp(half_width*(z_min-z)) and pushes halos into
        # too-low mass bins.  See fst_stochastic/docs/convergence.md.
        halo_mass_bins, _ = mass_accretion(
            self.parameters,
            self.z_bins,
            self.parameters.solver.halo_mass_bins,
            self.parameters.solver.halo_mass_accretion_alpha_bin_centers
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

        logger.info(f"Computing profiles for {self.z_bins.size} redshifts, {self.parameters.solver.halo_mass_bins.size - 1} halo mass bins and {self.parameters.solver.halo_mass_accretion_alpha.size - 1} alpha bins (numerics={getattr(self.parameters.solver, 'numerics', 'legacy')}, integrator={_profile_integrator(self.parameters)}).")
        r_lyal = np.logspace(-5, 2, 1000, base=10)
        # TODO - recheck this:
        # physical distance for lyal profile. Never goes further away than 10**2 = 100 pMpc/h (checked)

        if _is_legacy(self.parameters):
            r_bubble = self.R_bubble()
            rho_xray = self.rho_xray(self.r_grid, x_e)
            rho_heat = self.rho_heat(rho_xray)
            rho_alpha = rho_alpha_profile(self.parameters, self.z_bins, r_lyal, halo_mass, halo_mass_derivative)
        else:
            r_bubble, rho_xray, rho_heat, rho_alpha = self._continuous_profiles(self.parameters, r_lyal)
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



    # ------------------------------------------------------------------------------------
    # v2 numerics (solver.numerics == 'v2'): continuous source onset
    # ------------------------------------------------------------------------------------
    # Emission rates are evaluated on the analytic bin-centre trajectory at every integration
    # point instead of being tabulated at the profile redshifts and interpolated between them.
    # The source.halo_mass_min cutoff therefore acts at the exact crossing time of each
    # trajectory (in the legacy scheme a source switching on between two profile redshifts is
    # credited with a linear ramp from the earlier one). See
    # fst_stochastic/docs/src_fix_plan_2026-10-01.md (D1-D4) and the prototype
    # fst_stochastic/review/phase3_onset.py, which this reproduces.

    def _z_ref(self) -> float:
        """Reference redshift of the mass grid (where the bin edges are defined): min(z_bins)."""
        return float(np.min(self.z_bins))

    def _continuous_profiles(self, parameters: Parameters, r_lyal: np.ndarray):
        """(R_bubble, rho_xray, rho_heat, rho_alpha) on the profile grid, v2 scheme."""
        m_ref = parameters.solver.halo_mass_bin_centers
        alphas = parameters.solver.halo_mass_accretion_alpha_bin_centers
        z_ref = self._z_ref()
        r_bubble = self._R_bubble_continuous(parameters, m_ref, alphas, z_ref)
        rho_xray, rho_heat = self._xray_and_heat_continuous(parameters, m_ref, alphas, z_ref)
        rate = emission_rate_fn(parameters, m_ref, alphas, z_ref)
        if _profile_integrator(parameters) == 'direct':
            rho_alpha = rho_alpha_profile_direct(parameters, self.z_bins, r_lyal, rate, m_ref.size, alphas.size)
        else:
            halo_mass, halo_mass_derivative = mass_accretion(parameters, self.z_bins, m_ref, alphas)
            rho_alpha = rho_alpha_profile(parameters, self.z_bins, r_lyal, halo_mass, halo_mass_derivative,
                                          rate_fn=rate)
        return r_bubble, rho_xray, rho_heat, rho_alpha

    def _R_bubble_continuous(self, parameters: Parameters, m_ref, alphas, z_ref) -> np.ndarray:
        """Ionized-bubble radius by an exponential integrator (v2).

        The bubble volume obeys the linear ODE dV/da = P(a) - B(a) V with source
        P = K Ngdot/n_b and recombination B = K alpha_B C h^3/cm^3 n_b/a^3, K = km_per_Mpc/(H a).
        Over each of ``solver.bubble_substeps`` uniform sub-steps h per profile interval:
        V <- V exp(-dI) + h/2 (P_j exp(-dI) + P_{j+1}), dI = h (B_j + B_{j+1})/2, with Ngdot
        evaluated analytically on the trajectory at every sub-step (so the onset step lands
        within one sub-step of the exact crossing). Second order, unconditionally stable in
        the recombination term, and independent of the other columns (no shared error norm).
        """
        Ob, h0 = parameters.cosmology.Ob, parameters.cosmology.h0
        nb = (Ob * constants.rhoc0) / (constants.m_p_in_Msun * h0)
        crec = alpha_HII(1e4) * parameters.solver.clumping / cm_per_Mpc ** 3 * h0 ** 3
        nsub = int(parameters.solver.bubble_substeps)
        aa = 1.0 / (1.0 + self.z_bins)
        V = np.zeros((m_ref.size, alphas.size))
        out = np.zeros((m_ref.size, alphas.size, aa.size))
        for k in range(aa.size - 1):
            a_sub = np.linspace(aa[k], aa[k + 1], nsub + 1)
            z_sub = 1.0 / a_sub - 1.0
            Mh, dM = trajectory_at(parameters, z_sub, m_ref, alphas, z_ref)
            N = Ngdot_ion(parameters, z_sub[None, None, :], Mh, dM)
            N[..., z_sub > parameters.solver.z_source_start] = 0.0
            K = km_per_Mpc / (hubble(z_sub, parameters) * a_sub)
            B = K * crec * nb / a_sub ** 3
            P = (K / nb)[None, None, :] * N
            for j in range(nsub):
                h = a_sub[j + 1] - a_sub[j]
                e = np.exp(-0.5 * h * (B[j] + B[j + 1]))
                V = V * e + 0.5 * h * (P[..., j] * e + P[..., j + 1])
            out[..., k + 1] = V
        np.clip(out, 0, None, out=out)
        return (out * 3 / (4 * np.pi)) ** (1 / 3)

    def _internal_grid(self, refine: int):
        """Profile z grid refined ``refine`` x (uniform in scale factor) between neighbours.

        Returns ``(z_int, idx)``: z_int descending, and the positions of the profile redshifts
        in it (kept exact)."""
        aa = 1.0 / (1.0 + self.z_bins)
        pieces = [np.linspace(aa[k], aa[k + 1], refine + 1)[:-1] for k in range(aa.size - 1)]
        idx = np.arange(aa.size) * refine
        a_int = np.concatenate(pieces + [aa[-1:]])
        z_int = 1.0 / a_int - 1.0
        z_int[idx] = self.z_bins
        return z_int, idx

    def _xray_and_heat_continuous(self, parameters: Parameters, m_ref, alphas, z_ref, chunk_bytes: float = 2e9):
        """rho_xray and rho_heat on the profile grid, v2.

        rho_xray is computed on the internal grid (``solver.onset_internal_refine``) with the
        analytic lookback rate and lookback spacing ``solver.xray_dz_prime``; the heating is the
        exact quadrature of d(a^2 T)/da = a^2 gamma on that grid (no ODE solver, never
        negative); both are subsampled at the profile redshifts. Alpha columns are processed in
        chunks so the internal-grid arrays stay below ~``chunk_bytes``.
        """
        z_int, idx = self._internal_grid(int(parameters.solver.onset_internal_refine))
        xe = np.full(z_int.size, 2e-4)
        n_r, n_m, n_a = self.r_grid.size, m_ref.size, alphas.size
        per_alpha = 2 * n_r * n_m * z_int.size * 8
        step = max(1, int(chunk_bytes // max(per_alpha, 1)))
        rho_xray = np.zeros((n_r, n_m, n_a, self.z_bins.size))
        rho_heat = np.zeros_like(rho_xray)
        for a0 in range(0, n_a, step):
            sl = slice(a0, min(n_a, a0 + step))
            rate = emission_rate_fn(parameters, m_ref, alphas[sl], z_ref)
            xray = self._rho_xray_direct if _profile_integrator(parameters) == 'direct' else self._rho_xray_on
            x_int = xray(parameters, self.r_grid, z_int, xe, rate, n_m, sl.stop - sl.start,
                         float(parameters.solver.xray_dz_prime))
            h_int = self._heat_quadrature(parameters, z_int, x_int)
            rho_xray[:, :, sl, :] = x_int[..., idx]
            rho_heat[:, :, sl, :] = h_int[..., idx]
        return rho_xray, rho_heat

    def _rho_xray_on(self, parameters: Parameters, rr, z_grid, xe, rate_fn, n_mass, n_alpha, dz_prime):
        """Same physics as :meth:`rho_xray`, on an arbitrary z grid with a given lookback-rate
        callable and lookback spacing (``rho_xray`` is this with ``self.z_bins``, the tabulated
        interpolator and dz' = 0.1)."""
        Om, Ob, h0 = parameters.cosmology.Om, parameters.cosmology.Ob, parameters.cosmology.h0
        Emin, Emax, NE = parameters.source.energy_cutoff_min_xray, parameters.source.energy_cutoff_max_xray, 50
        nb0 = rhoc0 * Ob / (m_p_in_Msun * h0)
        nu = np.logspace(np.log(Emin / constants.h_eV_sec), np.log(Emax / constants.h_eV_sec), NE, base=np.e)
        f_He_bynumb = 1 - parameters.solver.HI_frac
        nH0, nHe0 = (1 - f_He_bynumb) * nb0, f_He_bynumb * nb0
        original = self.parameters
        self.parameters = parameters          # _z_star_at reads self.parameters
        try:
            out = np.zeros((len(rr), n_mass, n_alpha, len(z_grid)))
            for i, z in enumerate(z_grid):
                z_star_i = self._z_star_at(z)
                if z > z_star_i:
                    continue
                N_prime_i = max(4, int((z_star_i - z) / dz_prime))
                z_prime = np.logspace(np.log(z), np.log(z_star_i), N_prime_i, base=np.e)
                rcom_prime = comoving_distance(z_prime, parameters) * h0
                rate = rate_fn(z_prime)

                def integrand(nu_val):
                    tau_prime = cum_optical_depth(z_prime, nu_val * constants.h_eV_sec, parameters)
                    nu_prime = nu_val[:, None] * ((1 + z_prime) / (1 + z))[None, :]
                    eps_X = eps_xray(nu_prime, parameters)
                    fac = (np.exp(-tau_prime) * eps_X)[:, None, None, :] * rate[None, ...]
                    fac_r = interp1d(rcom_prime, fac, axis=-1, fill_value=0.0, bounds_error=False)(rr)
                    fac_r = np.moveaxis(fac_r, -1, 1)
                    pref = ((nH0 / nb0) * sigma_HI(nu_val * h_eV_sec) * (nu_val * h_eV_sec - E_HI)
                            + (nHe0 / nb0) * sigma_HeI(nu_val * h_eV_sec) * (nu_val * h_eV_sec - E_HeI))
                    return pref[:, None, None, None] * fac_r

                heat = trapezoid(integrand(nu), nu, axis=0)
                out[..., i] = f_Xh(xe[i]) / (4 * np.pi * (rr / (1 + z)) ** 2)[:, None, None] * heat / (cm_per_Mpc / h0) ** 2
        finally:
            self.parameters = original
        return out

    # ---- vectorised variants of _rho_xray_on (fst_stochastic/docs/src_fix_plan_2026-10-01.md §10)
    # _rho_xray_direct is used when solver.profile_integrator == 'direct' (the v2 default);
    # validated against _rho_xray_on by fst_stochastic/review/check_profile_vectorise.py, which
    # also holds the intermediate step A (frequency integral first, same sum reordered).

    def _xray_spectral_kernel(self, parameters: Parameters, z, z_star_i, dz_prime):
        """Frequency-integrated kernel of the X-ray lookback integral at one redshift ``z``.

        Returns ``(z_prime, rcom_prime, g)`` with
        g(z') = trapz_nu[ pref(nu) exp(-tau(z, z', nu)) eps_X(nu (1+z')/(1+z)) ],
        on the same lookback grid as :meth:`_rho_xray_on`. g does not depend on the halo
        column, so the frequency integral is done once per z instead of once per column."""
        Ob, h0 = parameters.cosmology.Ob, parameters.cosmology.h0
        Emin, Emax, NE = parameters.source.energy_cutoff_min_xray, parameters.source.energy_cutoff_max_xray, 50
        nb0 = rhoc0 * Ob / (m_p_in_Msun * h0)
        nu = np.logspace(np.log(Emin / constants.h_eV_sec), np.log(Emax / constants.h_eV_sec), NE, base=np.e)
        f_He_bynumb = 1 - parameters.solver.HI_frac
        nH0, nHe0 = (1 - f_He_bynumb) * nb0, f_He_bynumb * nb0
        N_prime_i = max(4, int((z_star_i - z) / dz_prime))
        z_prime = np.logspace(np.log(z), np.log(z_star_i), N_prime_i, base=np.e)
        rcom_prime = comoving_distance(z_prime, parameters) * h0
        tau_prime = cum_optical_depth(z_prime, nu * constants.h_eV_sec, parameters)
        nu_prime = nu[:, None] * ((1 + z_prime) / (1 + z))[None, :]
        eps_X = eps_xray(nu_prime, parameters)
        pref = ((nH0 / nb0) * sigma_HI(nu * h_eV_sec) * (nu * h_eV_sec - E_HI)
                + (nHe0 / nb0) * sigma_HeI(nu * h_eV_sec) * (nu * h_eV_sec - E_HeI))
        g = trapezoid(pref[:, None] * np.exp(-tau_prime) * eps_X, nu, axis=0)
        return z_prime, rcom_prime, g

    def _rho_xray_direct(self, parameters: Parameters, rr, z_grid, xe, rate_fn, n_mass, n_alpha, dz_prime,
                         chunk_elems: float = 5e7):
        """X-ray lookback integral of :meth:`_rho_xray_on` (``profile_integrator='direct'``).

        The frequency integral is done once per redshift (:meth:`_xray_spectral_kernel`), and the
        emission rate is evaluated analytically at the lookback redshift z'(r) of every radius
        instead of being linearly interpolated between lookback points; only the smooth kernel g
        is interpolated. The rate evaluation is batched over many redshifts at once (batches of
        about ``chunk_elems`` rate values)."""
        h0 = parameters.cosmology.h0
        n_r, n_z = len(rr), len(z_grid)
        zr = np.full((n_z, n_r), np.nan)          # z'(r) per redshift; nan = beyond the lookback
        gr = np.zeros((n_z, n_r))
        original = self.parameters
        self.parameters = parameters
        try:
            for i, z in enumerate(z_grid):
                z_star_i = self._z_star_at(z)
                if z > z_star_i:
                    continue
                z_prime, rcom_prime, g = self._xray_spectral_kernel(parameters, z, z_star_i, dz_prime)
                inside = rr <= rcom_prime[-1]
                zr[i, inside] = np.interp(rr[inside], rcom_prime, z_prime)
                gr[i, inside] = np.interp(rr[inside], rcom_prime, g)
        finally:
            self.parameters = original
        geom = f_Xh(xe)[:, None] / (4 * np.pi * (rr[None, :] / (1 + z_grid[:, None])) ** 2) / (cm_per_Mpc / h0) ** 2
        weight = (geom * gr).ravel()                       # (z * r)
        zr = zr.ravel()
        valid = np.flatnonzero(np.isfinite(zr) & (weight != 0))
        out = np.zeros((n_mass, n_alpha, n_z * n_r))
        step = max(1, int(chunk_elems // max(1, n_mass * n_alpha)))
        for s in range(0, valid.size, step):
            sel = valid[s:s + step]
            out[..., sel] = rate_fn(zr[sel]) * weight[sel]
        return np.moveaxis(out.reshape(n_mass, n_alpha, n_z, n_r), -1, 0)   # (r, M, alpha, z)

    @staticmethod
    def _heat_quadrature(parameters: Parameters, z_grid, rho_xray, nsub: int = 64) -> np.ndarray:
        """Exact solution of dT/da = gamma - 2T/a with T(a0) = 0: T(a) = a^-2 int a'^2 gamma da'.

        gamma = 2 rho_xray / (3 k_B a H) * km_per_Mpc, with rho_xray linear in a between grid
        points (the interpolant the legacy ODE uses). The integrand is linear in rho on each
        interval, so the only quadrature is over the smooth factor 2 a km_per_Mpc/(3 k_B H(a)),
        done with ``nsub`` trapezoid sub-steps (converged to ~1e-5). Never negative."""
        aa = 1.0 / (1.0 + np.asarray(z_grid, dtype=float))
        out = np.zeros_like(rho_xray)
        acc = np.zeros(rho_xray.shape[:-1])
        for k in range(aa.size - 1):
            a_sub = np.linspace(aa[k], aa[k + 1], nsub + 1)
            w = (a_sub - aa[k]) / (aa[k + 1] - aa[k])
            f = 2.0 * a_sub * km_per_Mpc / (3.0 * kb_eV_per_K * hubble(1.0 / a_sub - 1.0, parameters))
            c0, c1 = trapezoid(f * (1 - w), a_sub), trapezoid(f * w, a_sub)
            acc = acc + rho_xray[..., k] * c0 + rho_xray[..., k + 1] * c1
            out[..., k + 1] = acc / aa[k + 1] ** 2
        return out

    def _z_star_at(self, z: float) -> float:
        """Return the maximum lookback redshift for X-ray integration at redshift *z*.

        When ``parameters.source.t_source_age`` is ``None`` (default),
        ``parameters.solver.z_source_start`` is used.  Otherwise the lookback
        window is capped at ``t_source_age`` Myr by integrating
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
        dt_dz  = 1e-6 / ((1 + z_grid) * Hubble(z_grid, self.parameters))  # Myr per unit z
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

        # Build dMdt_int once over the full redshift history, zero at z_source_start (no
        # emission before sources start). z_prime is always queried in [z_current, z_star_i],
        # so it is never extrapolated below the lowest z_bin.
        dMdt_int = emission_history_interpolator(self.z_bins, M_star_dot, self.parameters.solver.z_source_start)

        for i, z in enumerate(self.z_bins):
            # Determine the maximum lookback redshift for this snapshot.
            # When t_source_age is set, this caps the integration window to the
            # halo's lifetime; otherwise the full history back to z_source_start is used.
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
        # Start the heating integration at the first profile redshift with T = 0
        # (fix_plan_2026-09-03 finding 8). Sources begin at z_source_start, and the
        # profile grid now starts there (profile_redshifts extends it upward), so
        # rho_xray is ~0 at the top edge. The old code seeded the ODE at
        # z_decoupling (=135) and prepended rho_xray=0 there, which made the
        # interpolation ramp X-ray heating linearly in a all the way down from z=135 --
        # injecting spurious heat before any source existed. T=0 at z_source_start is
        # the correct initial condition (X-ray heating is zero above it; the adiabatic
        # temperature is handled separately by T_adiab_fluctu at paint time).
        zz = self.z_bins

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
        # The ODE now integrates directly over the profile grid (no prepended
        # z_decoupling initial-condition point to drop), so every z-slice is kept.
        # rho_heat has shape (rr * M_bin * alpha_bin, zz) with all non-redshift axes
        # flattened; reshape to (rr, M_bin, alpha_bin, zz).
        rho_heat_full = source_in_time.reshape((*single_rho_xray_shape, -1))

        return rho_heat_full

class RadiationProfileFstSolver(RadiationProfileSolver):
    """Compute radiation profiles on a (mass, alpha, f_st, z) grid.

    This solver extends the legacy single-f_st solver without modifying its
    behavior. It precomputes one profile cube per f_st value and stacks the
    results into a :class:`RadiationProfilesFStarGrid` instance.
    """

    def profile_cache_namespace(self) -> str:
        """Return the cache namespace for f_st-grid radiation profiles.

        The namespace, not ``profiles_fstar_hash``, is what distinguishes f_st grids:
        ``f_st_paint_distribution`` is a paint-only key stripped from that hash, yet it picks
        log or linear spacing in :meth:`_build_f_st_grid`. The spacing is therefore part of
        the name, taken from the grid as built (review_2026-09-14 finding 5).
        """
        f_st_min = float(self.f_st_grid[0])
        f_st_max = float(self.f_st_grid[-1])
        f_st_n = int(self.f_st_grid.size)
        return (f"radiation_profiles_fstar_grid_fmin_{f_st_min:.6g}_fmax_{f_st_max:.6g}_n_{f_st_n}"
                f"_{self.f_st_grid_spacing}")

    def get_or_compute_profiles(self, handler: Handler, force_recompute: bool = False) -> "RadiationProfilesFStarGrid":
        """Load f_st-grid profiles from cache or compute and save them if unavailable.

        Args:
            force_recompute: If ``True``, ignore any cached cube and recompute from
                scratch (finding 6 -- otherwise ``--force-recompute-profiles`` is a
                no-op on the single-rank path, which re-finds and returns the cache).
        """
        if not force_recompute:
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
        else:
            logger.info("force_recompute=True — recomputing f_st-grid radiation profiles.")

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
    def __init__(self, parameters: Parameters, redshifts: np.ndarray):
        super().__init__(parameters, redshifts)
        self.f_st_grid = self._build_f_st_grid()

    def _build_f_st_grid(self) -> np.ndarray:
        source = self.parameters.source
        distribution = getattr(source, "f_st_paint_distribution", "lognormal").lower()
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
            # k_star = (f_st_n - 1) // 2
            # dlog = (np.log10(f_st_max) - np.log10(f_st_center)) / (f_st_n - 1 - k_star)
            # log_min = np.log10(f_st_center) - k_star * dlog
            # f_st_min = 10**log_min
            # logger.info(f"Redefine f_st_min to {f_st_min:.4f}")
            # logger.info("Return f_st_grid in log space")
            self.f_st_grid_spacing = "log"
            return np.logspace(np.log10(f_st_min), np.log10(f_st_max), f_st_n, base=10)
        self.f_st_grid_spacing = "lin"
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
        if not _is_legacy(parameters):
            # v2: per-f_st parameters carry f_st into the analytic emission rate
            r_bubble, rho_xray, rho_heat, rho_alpha = self._continuous_profiles(parameters, r_lyal)
            return r_bubble, rho_xray, rho_heat, rho_alpha
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

        # Alpha axis must be the bin CENTRES here, matching the profiles below -- see the
        # note in RadiationProfileSolver.solve.
        halo_mass_bins, _ = mass_accretion(
            self.parameters,
            self.z_bins,
            self.parameters.solver.halo_mass_bins,
            self.parameters.solver.halo_mass_accretion_alpha_bin_centers
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
