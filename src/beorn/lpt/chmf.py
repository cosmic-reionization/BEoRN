"""Conditional Halo Mass Function (CHMF) based on Extended Press-Schechter theory.

EPS theory (Bond et al. 1991; Lacey & Cole 1993) gives the conditional
first-crossing distribution for halos forming in a region with local linear
overdensity delta_env smoothed on mass scale M_env:

    dn/d ln M |_delta = (rho_m / M) |d ln sigma_eff / d ln M| f_PS(nu_eff)

where:
    sigma_eff^2(M) = sigma^2(M) - sigma^2(M_env)    (conditional variance)
    nu_eff         = (delta_c - delta_env) / sigma_eff
    f_PS(nu)       = sqrt(2/pi) * nu * exp(-nu^2/2)

Averaging the CHMF over all cells with their LPT overdensities recovers the
unconditional Press-Schechter HMF — the key self-consistency check of EPS.
"""
from __future__ import annotations

import logging
import warnings
import numpy as np

from ..structs import Parameters, HaloCatalog
from .linear_power import PowerSpectrum

logger = logging.getLogger(__name__)


class CHMF:
    """Conditional Halo Mass Function via Extended Press-Schechter theory.

    Precomputes sigma^2(M) at z=0 from the linear power spectrum using
    vectorised numerical integration over ln k, then interpolates for
    fast evaluation at arbitrary M.

    Args:
        parameters:  BEoRN Parameters object.
        ps_method:   Power spectrum method passed to :func:`get_power_spectrum`.
                     Default ``'eisenstein_hu'``.
        delta_c:     Linear collapse threshold. ``None`` (default) reads
                     ``parameters.halo_sim.delta_c`` (itself ``1.686`` by
                     default).
        power_spectrum: Pre-built :class:`~beorn.lpt.linear_power.PowerSpectrum`
                     instance to use directly instead of constructing one from
                     ``ps_method``/``ps_kwargs`` (issue #42, O10) — pass this
                     to share a single instance with e.g. an :class:`LPTBase`
                     solver instead of each building its own (and each paying
                     its own A_s normalisation). ``ps_method``/``ps_kwargs``
                     are ignored when set.
        **ps_kwargs: Forwarded to the power spectrum constructor.
    """

    def __init__(
        self,
        parameters: Parameters,
        ps_method: str = 'eisenstein_hu',
        delta_c: float | None = None,
        power_spectrum: PowerSpectrum | None = None,
        **ps_kwargs,
    ):
        # Runtime import: beorn.mass_function.base imports beorn.lpt at module
        # load, so a top-level import here would be circular.
        from ..mass_function.base import MassFunction

        self.parameters = parameters
        self.delta_c = delta_c if delta_c is not None else parameters.halo_sim.delta_c
        # sigma^2(M) machinery is delegated to the shared MassFunction base
        # (same 1000 ln-k nodes, 200-point log-M table, top-hat window that
        # used to be duplicated here) — one sigma^2 implementation for both
        # the unconditional HMF and the CHMF.
        self._mf = MassFunction(parameters, window='tophat', ps_method=ps_method,
                                power_spectrum=power_spectrum, **ps_kwargs)
        self.power_spectrum: PowerSpectrum = self._mf.power_spectrum

    # ------------------------------------------------------------------
    # Physical constants in BEoRN units (Mpc/h) — delegated to MassFunction
    # ------------------------------------------------------------------

    @property
    def rho_m(self) -> float:
        """Mean comoving matter density in M_sun (Mpc/h)^{-3}."""
        return self._mf.rho_m

    def R_of_M(self, M: float | np.ndarray) -> float | np.ndarray:
        """Top-hat smoothing radius for mass M in Mpc/h."""
        return self._mf.R_of_M(M)

    def M_of_R(self, R: float) -> float:
        """Mass enclosed in top-hat sphere of radius R (Mpc/h) in M_sun."""
        return self._mf.M_of_R(R)

    # ------------------------------------------------------------------
    # sigma^2(M, z) — delegated to MassFunction
    # ------------------------------------------------------------------

    def sigma2_z0(self, M: float | np.ndarray) -> float | np.ndarray:
        """sigma^2(M) at z=0, interpolated from the precomputed grid."""
        M = np.asarray(M, dtype=float)
        return np.exp(self._mf._sigma2_interp(np.log(M)))

    def _D1(self, z: float) -> float:
        """Linear growth factor D(z), normalised to D(0) = 1."""
        return self._mf._D1(z)

    def sigma2(self, M: float | np.ndarray, z: float) -> float | np.ndarray:
        """sigma^2(M, z) = D1(z)^2 * sigma^2(M, z=0)."""
        return self._mf.sigma2(M, z)

    # ------------------------------------------------------------------
    # Halo mass functions
    # ------------------------------------------------------------------

    def hmf_ps(self, M: np.ndarray, z: float) -> np.ndarray:
        """Unconditional Press-Schechter dn/d ln M in (Mpc/h)^{-3}.

        Args:
            M:  Halo masses in M_sun (1-D array).
            z:  Redshift.

        Returns:
            dn/d ln M — same shape as M, units (Mpc/h)^{-3}.
        """
        M = np.atleast_1d(np.asarray(M, dtype=float))
        eps = 0.01
        s2_p = self.sigma2(M * (1.0 + eps), z)
        s2_m = self.sigma2(M * (1.0 - eps), z)
        sigma_M = np.sqrt(self.sigma2(M, z))
        dln_sigma_dlnM = (np.log(np.sqrt(s2_p)) - np.log(np.sqrt(s2_m))) / (2.0 * eps)

        nu = self.delta_c / sigma_M
        f_nu = np.sqrt(2.0 / np.pi) * nu * np.exp(-0.5 * nu ** 2)
        return (self.rho_m / M) * np.abs(dln_sigma_dlnM) * f_nu

    def st_ps_ratio(self, M: np.ndarray, z: float) -> np.ndarray:
        """Sheth-Tormen / Press-Schechter ratio of the unconditional HMFs.

        R(M, z) = f_ST(nu) / f_PS(nu) with nu = delta_c / sigma(M, z); the
        rho_m/M and d ln sigma/d ln M factors cancel.  Multiplying the
        conditional PS mass function by this ratio is the hybrid prescription
        of Barkana & Loeb (2004) (see e.g. Ghara, Choudhury & Datta 2015,
        appendix A): the delta-dependence keeps the EPS shape while the
        volume average matches the simulation-calibrated ST mass function
        exactly (because the volume average of conditional PS is exactly PS).

        Args:
            M:  Halo masses in M_sun (scalar or 1-D array).
            z:  Redshift.

        Returns:
            Dimensionless ratio, same shape as M.
        """
        from ..mass_function.models import _f_nu, _normalise_A
        # models._f_nu uses the nu = (delta_c / sigma)^2 convention
        nu2 = self.delta_c ** 2 / np.asarray(self.sigma2(M, z), dtype=float)
        f_st = _f_nu(nu2, 0.3, 0.707, _normalise_A(0.3), backend='numpy')
        f_ps = _f_nu(nu2, 0.0, 1.0, 0.5, backend='numpy')
        return f_st / f_ps

    def hmf_st(self, M: np.ndarray, z: float) -> np.ndarray:
        """Unconditional Sheth-Tormen dn/d ln M in (Mpc/h)^{-3}."""
        return self.hmf_ps(M, z) * self.st_ps_ratio(M, z)

    def hmf_chmf_field(
        self,
        M: float,
        delta_field: np.ndarray,
        sigma2_env: float,
        z: float,
    ) -> np.ndarray:
        """Conditional dn/d ln M for all cells, vectorised over a 3-D delta field.

        Args:
            M:           Halo mass in M_sun (scalar).
            delta_field: Linear overdensity at each cell, shape (N, N, N).
            sigma2_env:  sigma^2 on the environmental conditioning scale.
            z:           Redshift.

        Returns:
            Array of shape (N, N, N) with dn/d ln M at each cell in (Mpc/h)^{-3}.
            Cells where sigma2(M) <= sigma2_env return 0.
        """
        sigma2_M = float(self.sigma2(M, z))
        S_eff = sigma2_M - sigma2_env
        if S_eff <= 0.0:
            return np.zeros_like(delta_field, dtype=float)

        sigma_eff = np.sqrt(S_eff)

        # Effective peak height: positive in underdense regions (normal) and
        # negative in very overdense ones (delta > delta_c, already collapsed).
        nu_eff = (self.delta_c - delta_field) / sigma_eff

        # PS first-crossing distribution; clamped to >= 0
        f_nu = np.sqrt(2.0 / np.pi) * nu_eff * np.exp(-0.5 * nu_eff ** 2)
        np.maximum(f_nu, 0.0, out=f_nu)

        # d ln sigma_eff / d ln M via finite difference
        eps = 0.01
        S_p = float(self.sigma2(M * (1.0 + eps), z)) - sigma2_env
        S_m = float(self.sigma2(M * (1.0 - eps), z)) - sigma2_env
        if S_p <= 0.0 or S_m <= 0.0:
            return np.zeros_like(delta_field, dtype=float)
        dln_sigma_eff_dlnM = (np.log(np.sqrt(S_p)) - np.log(np.sqrt(S_m))) / (2.0 * eps)

        return (self.rho_m / M) * abs(dln_sigma_eff_dlnM) * f_nu

    def hmf_st_movingbarrier(
        self,
        M: float,
        delta_field: np.ndarray,
        sigma2_env: float,
        z: float,
    ) -> np.ndarray:
        """Sheth-Tormen conditional dn/d ln M via a moving-barrier excursion-set
        solution (Sheth & Tormen 2002), as implemented by Davies, Mesinger &
        Murray (2025, 21cmFASTv4) eqs. 2-4.

        Unlike :meth:`hmf_chmf_field` rescaled by :meth:`st_ps_ratio` (the
        Barkana & Loeb 2004 hybrid: the pure-PS conditional *shape* times the
        *unconditional* ST/PS ratio, so its delta-dependence is inherited
        entirely from PS), this solves the conditional first-crossing problem
        directly for the ellipsoidal-collapse moving barrier
        ``B(z, sigma) = sqrt(a) delta_crit(z) (1 + beta (a delta_crit(z)^2 / sigma^2)^-alpha)``
        (Jenkins et al. 2001 parameters ``a=0.7, alpha=0.81, beta=0.34``),
        so its delta-dependence need not match PS's at all -- already
        ST-calibrated on its own, with no separate rescaling ratio needed
        (see :attr:`beorn.structs.HaloSimParameters.chmf_recipe`).

        Works entirely in Davies et al.'s own convention -- sigma^2 at z=0
        with a redshift-dependent barrier ``delta_crit(z) = delta_c/D(z)``,
        rather than BEoRN's own (z-evolved sigma^2, fixed delta_c) convention
        used everywhere else in this class -- ``delta_field``/``sigma2_env``
        (both BEoRN-native, z-evolved, i.e. the same inputs
        :meth:`hmf_chmf_field` takes) are converted internally.

        Args:
            M:           Halo mass in M_sun (scalar).
            delta_field: **z-evolved** linear overdensity at each cell (BEoRN's
                         own convention -- same input as :meth:`hmf_chmf_field`),
                         any shape.
            sigma2_env:  **z-evolved** sigma^2 on the environment scale (BEoRN's
                         own convention -- same as :meth:`hmf_chmf_field`'s
                         ``sigma2_env``).
            z:           Redshift.

        Returns:
            dn/d ln M at each cell, same shape as ``delta_field``, in
            (Mpc/h)^{-3}. Zero where the (z=0-convention) sigma^2(M) does not
            exceed the converted conditioning sigma^2 (no valid conditional
            solution).

        Note:
            A differentiable (jax/torch) port of this same formula is
            available via :func:`conditional_dndlnm_diff`/
            :func:`halo_field_diff` with ``hmf_model='ST',
            chmf_recipe='MovingBarrier'``.
        """
        import math
        a, alpha, beta = 0.7, 0.81, 0.34  # Jenkins et al. (2001), as used by Davies et al. (2025)

        D = float(self._D1(z))
        delta_crit_z = self.delta_c / D  # Davies et al.'s own z-dependent barrier value

        x = float(self.sigma2_z0(M))  # Davies et al.'s "sigma^2" -- z=0 convention
        sigma2_cond_z0 = np.asarray(sigma2_env, dtype=float) / D ** 2
        delta_cond_z0 = np.asarray(delta_field, dtype=float) / D

        S_eff = x - sigma2_cond_z0
        valid = S_eff > 0
        S_safe = np.where(valid, S_eff, 1.0)

        # B(z, sigma) - eq. 2 - and its Taylor-expansion correction T - eq. 4.
        # g(x') = B(x') - delta_crit_z = c0_minus + c1 * x'^alpha is a pure
        # power law in x' = sigma^2 (delta_crit_z fixed at this z), so its
        # n-th d/dx'^n derivative has a simple closed form (falling-factorial
        # coefficient); evaluated at x' = x, the halo's own sigma^2 (eq. 4's
        # own argument list), not at sigma2_cond_z0.
        c1 = math.sqrt(a) * delta_crit_z * beta * (a * delta_crit_z ** 2) ** (-alpha)
        c0_minus = math.sqrt(a) * delta_crit_z - delta_crit_z  # (c0 - delta_crit_z)
        B_x = math.sqrt(a) * delta_crit_z * (1.0 + beta * (a * delta_crit_z ** 2 / x) ** (-alpha))

        T = np.zeros_like(S_eff, dtype=float)
        falling = 1.0  # running alpha*(alpha-1)*...*(alpha-n+1); empty product (n=0) = 1
        fact = 1.0
        for n in range(6):
            if n == 0:
                deriv_n = c0_minus + c1 * x ** alpha
            else:
                falling *= (alpha - (n - 1))
                deriv_n = c1 * falling * x ** (alpha - n)
                fact *= n
            T = T + (-S_eff) ** n / fact * deriv_n

        sigma_x = math.sqrt(x)
        # d(sigma)/dM via central finite difference in ln M (z=0 convention),
        # matching hmf_ps/hmf_chmf_field's own numerical-derivative style.
        eps = 0.01
        sigma_p = np.sqrt(self.sigma2_z0(M * (1.0 + eps)))
        sigma_m = np.sqrt(self.sigma2_z0(M * (1.0 - eps)))
        dsigma_dM = (sigma_p - sigma_m) / (2.0 * eps * M)

        dn_dM = ((self.rho_m / (np.sqrt(2.0 * np.pi) * M)) * np.abs(T) * np.abs(dsigma_dM)
                * (2.0 * sigma_x) / S_safe ** 1.5
                * np.exp(-(B_x - delta_cond_z0) ** 2 / (2.0 * S_safe)))
        dn_dM = np.where(valid, dn_dM, 0.0)

        return M * dn_dM  # dn/dlnM = M * dn/dM, matching hmf_chmf_field's own return convention


# ============================================================
# CHMFSampler
# ============================================================

class CHMFSampler:
    """Sample a halo catalog from the CHMF conditioned on an LPT density field.

    Uses per-cell EPS conditioning: each grid cell of size (L/N)^3 is treated
    as the environmental region.  For each mass bin the expected halo count per
    cell is computed from the conditional HMF and Poisson-sampled.

    Args:
        parameters:  BEoRN Parameters object.
        chmf:        Pre-built :class:`CHMF` instance.  If ``None`` one is
                     constructed with the supplied ``ps_method`` and
                     ``ps_kwargs``.
        ps_method:   Power spectrum method forwarded to :class:`CHMF`.
        delta_c:     Linear collapse threshold. ``None`` (default) reads
                     ``parameters.halo_sim.delta_c``.
        hmf_model:   ``'PS'`` — pure EPS conditional sampling whose volume
                     average is Press-Schechter.  ``'ST'`` — rescale each
                     mass bin by the unconditional ST/PS ratio (Barkana &
                     Loeb 2004 hybrid) so the volume average matches
                     Sheth-Tormen instead, as done by 21cmFAST-family codes.
                     ``None`` (default) reads ``parameters.halo_sim.hmf_model``
                     (itself ``'ST'`` by default).
        chmf_recipe: Only meaningful when ``hmf_model='ST'`` (silently
                     ignored for ``'PS'``, which has no ST-calibration route
                     to choose between). Case-insensitive.
                     ``'BarkanaLoeb2004'`` (default) --
                     :meth:`CHMF.hmf_chmf_field` (pure PS-conditional shape)
                     rescaled by the unconditional ST/PS ratio (Barkana &
                     Loeb 2004, ApJ 609, 474 -- not to be confused with
                     Barkana & Loeb 2005, ApJ 624, L65, a different paper).
                     ``'MovingBarrier'`` -- :meth:`CHMF.hmf_st_movingbarrier`,
                     the direct moving-barrier conditional solution (Davies,
                     Mesinger & Murray 2025), already ST-calibrated on its
                     own (no separate rescaling ratio applied). ``None``
                     (default) reads ``parameters.halo_sim.chmf_recipe``.
        **ps_kwargs: Forwarded to the power spectrum constructor.
    """

    _CHMF_RECIPES = {'barkanaloeb2004', 'movingbarrier'}

    def __init__(
        self,
        parameters: Parameters,
        chmf: CHMF | None = None,
        ps_method: str = 'eisenstein_hu',
        delta_c: float | None = None,
        hmf_model: str | None = None,
        chmf_recipe: str | None = None,
        **ps_kwargs,
    ):
        self.parameters = parameters
        self.chmf = chmf if chmf is not None else CHMF(
            parameters, ps_method, delta_c, **ps_kwargs
        )
        hmf_model = (hmf_model if hmf_model is not None else parameters.halo_sim.hmf_model).upper()
        if hmf_model not in ('PS', 'ST'):
            raise ValueError(
                f"Unknown hmf_model {hmf_model!r}. Choose 'PS' or 'ST'."
            )
        self.hmf_model = hmf_model

        chmf_recipe = (chmf_recipe if chmf_recipe is not None
                       else parameters.halo_sim.chmf_recipe)
        recipe_key = chmf_recipe.replace('_', '').lower()
        if recipe_key not in self._CHMF_RECIPES:
            raise ValueError(
                f"Unknown chmf_recipe {chmf_recipe!r}. "
                f"Choose 'BarkanaLoeb2004' or 'MovingBarrier' (case-insensitive)."
            )
        self.chmf_recipe = chmf_recipe
        # MovingBarrier is already ST-calibrated on its own -- applying the
        # unconditional ST/PS ratio on top would double-count the calibration.
        self._use_moving_barrier = (hmf_model == 'ST' and recipe_key == 'movingbarrier')
        self._hmf_field_fn = (self.chmf.hmf_st_movingbarrier if self._use_moving_barrier
                              else self.chmf.hmf_chmf_field)

    def _calibration_ratios(self, M_centers: np.ndarray, z: float) -> np.ndarray:
        """Per-mass-bin calibration factor for the expected counts.

        1 for pure EPS ('PS') and for the already-self-calibrated
        MovingBarrier recipe; the unconditional ST/PS ratio for the
        BarkanaLoeb2004 recipe (the only one that needs it).
        """
        if self.hmf_model == 'ST' and not self._use_moving_barrier:
            return np.asarray(self.chmf.st_ps_ratio(M_centers, z), dtype=float)
        return np.ones(len(M_centers))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tophat_smooth(self, delta: np.ndarray, R: float) -> np.ndarray:
        """Real-space top-hat smoothing via FFT.

        Matches the window :attr:`chmf`'s own ``sigma2(M,z)`` assumes
        (``CHMF`` always builds its ``MassFunction`` with ``window='tophat'``)
        and the smoothing :meth:`beorn.lpt.LPTBase.get_linear_density`
        applies via its own ``R_tophat`` argument -- so the field this
        returns has variance ``sigma2(M_of_R(R), z)`` up to finite-box/grid
        discreteness (issue #54).

        Args:
            delta: Real-space overdensity, shape (N, N, N).
            R:     Top-hat smoothing radius in Mpc/h.

        Returns:
            Smoothed field, same shape and dtype as ``delta``.
        """
        # Runtime import: beorn.mass_function.base imports beorn.lpt at
        # module load, so a top-level import here would be circular.
        from ..mass_function.window import TopHatWindow

        N = delta.shape[0]
        L = self.parameters.Lbox_hunits
        dk = 2.0 * np.pi / L
        kvals = np.fft.fftfreq(N, d=1.0 / N) * dk
        kz_vals = np.fft.rfftfreq(N, d=1.0 / N) * dk
        kx, ky, kz = np.meshgrid(kvals, kvals, kz_vals, indexing='ij')
        k2 = kx ** 2 + ky ** 2 + kz ** 2
        k = np.sqrt(np.where(k2 == 0, 1.0, k2))  # dummy value avoids 0/0 in W
        W = TopHatWindow().W(k * R)
        W = np.where(k2 == 0, 1.0, W)            # preserve the DC (mean) mode
        return np.fft.irfftn(np.fft.rfftn(delta) * W, axes=(0, 1, 2),
                             s=(N, N, N)).astype(delta.dtype)

    def _resolve_environment(
        self, delta_field: np.ndarray | None, R_env: float | None,
        precomputed_delta_env: tuple[np.ndarray, float] | None,
    ):
        """Return ``(delta_env, M_env)``, either passed through as-is from
        ``precomputed_delta_env`` or computed by top-hat-smoothing
        ``delta_field`` internally via :meth:`_environment`.

        See :meth:`sample`'s ``precomputed_delta_env`` argument for why a
        caller would supply this directly (issue #56's
        ``HaloSimParameters.field_oversample``: presmoothing at a finer,
        phase-synchronized resolution before block-averaging down to
        ``Ncell`` gives a less biased ``sigma2(M_env, z)`` match than
        smoothing at ``Ncell``'s own, coarser Nyquist limit).
        """
        if precomputed_delta_env is not None:
            return precomputed_delta_env
        return self._environment(delta_field, R_env)

    def _environment(self, delta_field: np.ndarray, R_env: float | None):
        """Resolve the conditioning field and environment mass.

        Always top-hat-smooths ``delta_field`` to the conditioning radius
        (the cell-equivalent radius when ``R_env`` is ``None``, or ``R_env``
        itself when set and above the cell size) -- EPS self-consistency
        requires ``Var(delta_env) = sigma2(M_env, z)``, and the caller is not
        expected to have done this smoothing themselves (issue #54).

        Returns:
            (delta_env, M_env) — the smoothed conditioning field and the
            environmental mass scale in M_sun.
        """
        params = self.parameters
        cell_size = params.Lbox_hunits / params.simulation.Ncell
        if R_env is None:
            M_env = self.chmf.rho_m * cell_size ** 3
            R_eq = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * cell_size
            return self._tophat_smooth(delta_field, R_eq), M_env
        M_env = self.chmf.M_of_R(R_env)
        if R_env <= cell_size:
            # Smoothing below the grid's own resolution isn't meaningful.
            return delta_field, M_env
        return self._tophat_smooth(delta_field, R_env), M_env

    def _delta_grid(self, delta_env: np.ndarray, n_nodes: int) -> np.ndarray:
        """Static 1-D delta grid spanning the field's actual range.

        Per-cell, :meth:`CHMF.hmf_chmf_field` depends on ``delta_field`` only
        through the scalar map delta -> nu_eff -> f_nu(nu_eff) (M and z fix
        sigma_eff and the mass-function normalisation) — so instead of
        evaluating that map (with its ``exp``) at all N^3 cells for every one
        of the ~40 mass bins, it is evaluated at ``n_nodes`` delta values
        spanning [delta_env.min(), delta_env.max()] and then linearly
        interpolated onto the actual field (issue #42, O6). This cuts the
        transcendental work from ``n_mass_bins * N^3`` to
        ``n_mass_bins * n_nodes`` (e.g. 40*512 vs 40*128^3), leaving only a
        cheap ``np.interp`` per bin on the full grid.

        The range is taken from the data itself (not a fixed a-priori span)
        so every actual delta_env value falls inside it — no extrapolation.
        """
        d_lo, d_hi = float(delta_env.min()), float(delta_env.max())
        if d_hi <= d_lo:
            d_hi = d_lo + 1e-6
        return np.linspace(d_lo, d_hi, n_nodes)

    def _tabulated_n_cond(self, M: float, delta_env: np.ndarray, sigma2_env: float,
                          z: float, d_grid: np.ndarray | None) -> np.ndarray:
        """Conditional dn/dlnM field via the O6 tabulate-then-interpolate path.

        Falls back to the exact per-cell evaluation (:attr:`_hmf_field_fn` --
        :meth:`CHMF.hmf_chmf_field` or :meth:`CHMF.hmf_st_movingbarrier`,
        per :attr:`chmf_recipe`) when ``d_grid`` is ``None``
        (``n_delta_nodes=None`` in the public methods) — kept as the
        validation reference and an explicit opt-out.
        """
        if d_grid is None:
            return self._hmf_field_fn(M, delta_env, sigma2_env, z)
        lam_grid = self._hmf_field_fn(M, d_grid, sigma2_env, z)
        return np.interp(delta_env.ravel(), d_grid, lam_grid).reshape(delta_env.shape)

    def _mass_bins(self, M_env: float, n_mass_bins: int):
        """Log-spaced mass bins between halo_sim.halo_mass_min and
        min(halo_sim.halo_mass_max, M_env)."""
        params = self.parameters
        M_min = params.halo_sim.halo_mass_min
        M_max_req = params.halo_sim.halo_mass_max

        if M_env <= M_min:
            raise ValueError(
                f"Environmental mass M_env = {M_env:.2e} Msun is smaller than "
                f"halo_mass_min = {M_min:.2e} Msun.  Use a larger R_env or a "
                f"coarser grid (smaller N or larger L)."
            )

        M_max = min(M_max_req, M_env * 0.999)
        if M_max < M_max_req:
            warnings.warn(
                f"CHMF can only sample halos with M < M_env = {M_env:.2e} Msun "
                f"(cell mass for N={params.simulation.Ncell}, "
                f"L={params.Lbox_hunits} Mpc/h).  "
                f"Capping M_max at {M_max:.2e} Msun.  "
                f"Increase R_env or use a coarser grid to access higher masses.",
                stacklevel=3,
            )

        M_edges = np.logspace(np.log10(M_min), np.log10(M_max), n_mass_bins + 1)
        M_centers = np.sqrt(M_edges[:-1] * M_edges[1:])
        dln_M = np.diff(np.log(M_edges))
        return M_centers, dln_M

    def expected_counts(
        self,
        delta_field: np.ndarray | None,
        z: float,
        R_env: float | None = None,
        n_mass_bins: int | None = None,
        n_delta_nodes: int | None = 512,
        precomputed_delta_env: tuple[np.ndarray, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expected halo counts per cell and mass bin — the Poisson intensity.

        This is the dense field the Poisson sampler draws from:
        ``lam[b, i, j, k] = dn/dlnM(M_b | delta[i,j,k]) * dlnM_b * V_cell``.
        It is the building block for **expected-number painting** (issue #42,
        G6): any halo-count-linear field equals ``sum_b lam_b * w(M_b)``
        exactly in expectation, with no discrete sampling — smooth in the
        conditioning field and the cosmology, hence the differentiable
        alternative to :meth:`sample`.

        Args:
            delta_field: **Raw** linear overdensity, shape (N, N, N) — see
                :meth:`sample` for the exact contract (no pre-smoothing
                needed; this method smooths it internally).
            z:           Redshift.
            R_env:       Environmental smoothing scale in Mpc/h (as in
                :meth:`sample`). ``None`` (default) reads
                ``parameters.halo_sim.R_env``.
            n_mass_bins: Number of log-spaced mass bins. ``None`` (default)
                reads ``parameters.halo_sim.n_mass_bins``.
            n_delta_nodes: Resolution of the per-bin Λ(M_b, δ) lookup table
                (issue #42, O6) — δ -> dn/dlnM is evaluated at this many
                points spanning the field's actual delta range and linearly
                interpolated onto the full grid, instead of evaluating the
                exact per-cell expression (with its ``exp`` call) at all N^3
                cells for every mass bin. ``None`` disables the table and
                falls back to the exact per-cell evaluation (validation
                reference / opt-out).
            precomputed_delta_env: ``(delta_env, M_env)`` pair to use
                directly instead of top-hat-smoothing ``delta_field``
                internally — for callers that already resolved the
                conditioning field at a finer, phase-synchronized
                resolution and block-averaged it back down to ``Ncell``
                (issue #56's ``HaloSimParameters.field_oversample``, see
                :class:`~beorn.load_input_data.LPTHaloLoader`). When given,
                ``delta_field``/``R_env`` are ignored entirely and
                ``delta_field`` may be ``None``.

        Returns:
            (M_centers, lam) — bin-centre masses (n_mass_bins,) in M_sun and
            expected counts of shape (n_mass_bins, N, N, N).

        Note:
            Memory scales as ``n_mass_bins * N^3 * 8`` bytes (e.g. 40 bins at
            128^3 is ~0.7 GB) — reduce ``n_mass_bins`` or evaluate
            :meth:`CHMF.hmf_chmf_field` per bin for large grids.
        """
        params = self.parameters
        R_env = R_env if R_env is not None else params.halo_sim.R_env
        n_mass_bins = n_mass_bins if n_mass_bins is not None else params.halo_sim.n_mass_bins
        V_cell = (params.Lbox_hunits / params.simulation.Ncell) ** 3

        delta_env, M_env = self._resolve_environment(delta_field, R_env, precomputed_delta_env)
        sigma2_env = float(self.chmf.sigma2(M_env, z))
        M_centers, dln_M = self._mass_bins(M_env, n_mass_bins)
        ratios = self._calibration_ratios(M_centers, z)
        d_grid = (self._delta_grid(delta_env, n_delta_nodes)
                 if n_delta_nodes is not None else None)

        lam = np.empty((len(M_centers),) + delta_env.shape, dtype=np.float64)
        for i_M, M in enumerate(M_centers):
            n_cond = self._tabulated_n_cond(M, delta_env, sigma2_env, z, d_grid)
            lam[i_M] = np.maximum(n_cond * ratios[i_M] * dln_M[i_M] * V_cell, 0.0)
        return M_centers, lam

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def sample(
        self,
        delta_field: np.ndarray | None,
        z: float,
        R_env: float | None = None,
        n_mass_bins: int | None = None,
        seed: int | None = None,
        n_delta_nodes: int | None = 512,
        precomputed_delta_env: tuple[np.ndarray, float] | None = None,
    ) -> HaloCatalog:
        """Sample a halo catalog from the conditional HMF.

        Each grid cell is treated as an independent environment with local
        overdensity ``delta_field[i,j,k]``.  The conditional HMF is evaluated
        per-cell and Poisson-sampled; halos are placed at random positions
        within their host cells.

        Args:
            delta_field: **Raw, linear** matter overdensity at cell
                resolution, shape ``(N, N, N)`` — use
                :meth:`LPTBase.get_linear_density` (no ``R_tophat`` needed;
                this method top-hat-smooths it internally to the conditioning
                scale, issue #54), *not* the CIC :meth:`LPTBase.get_density`:
                EPS self-consistency requires a Gaussian conditioning field,
                and the CIC field's shot-noise tails blow up the CHMF near
                M_env.
            z:           Redshift at which to sample.
            R_env:       Environmental smoothing scale in Mpc/h. ``None``
                         (default) reads ``parameters.halo_sim.R_env`` (itself
                         ``None`` by default — the cell-equivalent top-hat
                         radius is used as the conditioning scale). Either
                         way, ``delta_field`` is smoothed to this scale
                         internally.
            n_mass_bins: Number of log-spaced mass bins between
                         ``halo_sim.halo_mass_min`` and
                         ``min(halo_sim.halo_mass_max, M_env)``. ``None``
                         (default) reads ``parameters.halo_sim.n_mass_bins``.
            seed:        Random seed for reproducible Poisson draws and
                         intra-cell position sampling. ``None`` (default)
                         reads ``parameters.halo_sim.halo_sampler_seed``.
            n_delta_nodes: Resolution of the per-bin Λ(M_b, δ) lookup table
                (issue #42, O6) — see :meth:`expected_counts`. ``None`` falls
                back to the exact per-cell evaluation.
            precomputed_delta_env: ``(delta_env, M_env)`` pair to use
                directly instead of top-hat-smoothing ``delta_field``
                internally — see :meth:`expected_counts`'s identical
                argument. When given, ``delta_field``/``R_env`` are ignored
                entirely and ``delta_field`` may be ``None``.

        Returns:
            :class:`~beorn.structs.HaloCatalog` with positions in Mpc/h and
            masses in M_sun.

        Raises:
            ValueError: If ``R_env`` is set but ``M_env < halo_mass_min``.
        """
        params = self.parameters
        R_env = R_env if R_env is not None else params.halo_sim.R_env
        n_mass_bins = n_mass_bins if n_mass_bins is not None else params.halo_sim.n_mass_bins
        seed = seed if seed is not None else params.halo_sim.halo_sampler_seed
        N = params.simulation.Ncell
        L = params.Lbox_hunits
        cell_size = L / N
        V_cell = cell_size ** 3

        # ── Environment, sigma and mass bins ───────────────────────────
        delta_env, M_env = self._resolve_environment(delta_field, R_env, precomputed_delta_env)
        sigma2_env = float(self.chmf.sigma2(M_env, z))
        M_centers, dln_M = self._mass_bins(M_env, n_mass_bins)
        ratios = self._calibration_ratios(M_centers, z)
        M_min, M_max = M_centers[0], M_centers[-1]
        d_grid = (self._delta_grid(delta_env, n_delta_nodes)
                 if n_delta_nodes is not None else None)

        # ── Poisson sampling ───────────────────────────────────────────
        rng = np.random.default_rng(seed)
        positions_list: list[np.ndarray] = []
        masses_list: list[np.ndarray] = []

        for i_M, M in enumerate(M_centers):
            n_cond = self._tabulated_n_cond(M, delta_env, sigma2_env, z, d_grid)
            N_expected = np.maximum(n_cond * ratios[i_M] * dln_M[i_M] * V_cell, 0.0)
            N_sample = rng.poisson(N_expected)  # shape (N, N, N) int

            occupied = np.argwhere(N_sample > 0)  # (K, 3)
            if occupied.size == 0:
                continue

            counts = N_sample[occupied[:, 0], occupied[:, 1], occupied[:, 2]]
            total = int(counts.sum())

            # Vectorised position placement within cells
            repeated_cells = np.repeat(occupied, counts, axis=0)  # (total, 3)
            offsets = rng.random((total, 3)) * cell_size             # (total, 3)
            pos = repeated_cells * cell_size + offsets               # (total, 3)

            positions_list.append(pos.astype(np.float32))
            masses_list.append(np.full(total, M, dtype=np.float64))

        if positions_list:
            positions = np.concatenate(positions_list, axis=0)
            masses = np.concatenate(masses_list)
        else:
            positions = np.zeros((0, 3), dtype=np.float32)
            masses = np.zeros(0, dtype=np.float64)

        logger.debug(
            "CHMF sampled %d halos at z=%.3f (M range [%.1e, %.1e] Msun)",
            len(masses), z, M_min, M_max,
        )

        return HaloCatalog(
            positions=positions,
            masses=masses,
            parameters=params,
            redshift=float(z),
        )


# ============================================================
# Differentiable conditional HMF (numpy / jax / torch)
# ============================================================

def conditional_dndlnm_diff(
    M,
    delta_env,
    M_env,
    z,
    Om, Ob, h0, ns, sigma_8,
    delta_c=1.686,
    backend='numpy',
    n_k=1000,
    n_nodes=512,
    hmf_model='PS',
    chmf_recipe='BarkanaLoeb2004',
):
    """EPS conditional dn/dlnM — backend-generic pure function.

    Same physics as :meth:`CHMF.hmf_chmf_field`, but sigma^2 comes from the
    direct backend integral (:func:`beorn.mass_function.differentiable.sigma2_M`)
    and d ln(sigma_eff)/d lnM is analytic, so the result is differentiable
    with respect to the cosmology (Om, Ob, h0, ns, sigma_8), delta_c, **and
    the conditioning field delta_env itself** — the gradient path from halo
    abundances back to the LPT density field (issue #42, G6).

    The opt-in gpu/diff counterpart of the numpy class; runs on the device of
    its tensor inputs (CUDA / MPS float32).

    Args:
        M:          Halo mass in M_sun (scalar).
        delta_env:  Linear conditioning overdensity — any shape, any backend
                    array (numpy / jax / torch tensor).
        M_env:      Environmental mass in M_sun (scalar).
        z:          Redshift.
        Om, Ob, h0, ns, sigma_8: cosmological parameters (scalars or
                    zero-dim tensors carrying gradients).
        delta_c:    Linear collapse threshold.
        backend:    'numpy' (default), 'jax' or 'torch'.
        hmf_model:  ``'PS'`` (default) — pure EPS conditional.  ``'ST'`` —
                    Sheth-Tormen calibrated, via whichever ``chmf_recipe`` is
                    selected below.
        chmf_recipe: Only meaningful when ``hmf_model='ST'`` (ignored for
                    ``'PS'``). Case-insensitive. ``'BarkanaLoeb2004'``
                    (default) — the pure-PS conditional shape multiplied by
                    the unconditional ST/PS ratio (a closed-form function of
                    nu, so the result stays differentiable in all arguments).
                    ``'MovingBarrier'`` — the direct moving-barrier
                    conditional solution (Davies, Mesinger & Murray 2025,
                    eqs. 2-4; same physics as :meth:`CHMF.hmf_st_movingbarrier`,
                    ported to be differentiable w.r.t. the cosmology,
                    ``delta_c`` and ``delta_env``), already ST-calibrated on
                    its own with no separate rescaling ratio applied.

    Returns:
        dn/dlnM per cell in (Mpc/h)^{-3}, same shape as ``delta_env``; zero
        where sigma^2(M) <= sigma^2(M_env) or nu_eff < 0.
    """
    import math
    from ..constants import rhoc0
    from ..cosmo.differentiable import get_backend, device_of, as_array, growth_factor
    from ..mass_function.differentiable import sigma2_M

    hmf_model_u = hmf_model.upper()
    if hmf_model_u not in ('PS', 'ST'):
        raise ValueError(f"Unknown hmf_model {hmf_model!r}. Choose 'PS' or 'ST'.")
    recipe_key = chmf_recipe.replace('_', '').lower()
    if recipe_key not in ('barkanaloeb2004', 'movingbarrier'):
        raise ValueError(
            f"Unknown chmf_recipe {chmf_recipe!r}. "
            f"Choose 'BarkanaLoeb2004' or 'MovingBarrier' (case-insensitive)."
        )
    use_moving_barrier = (hmf_model_u == 'ST' and recipe_key == 'movingbarrier')

    name, xp = get_backend(backend)
    device = device_of(name, xp, delta_env, Om, Ob, h0, ns, sigma_8, delta_c)
    delta_env = as_array(delta_env, name, xp, device)
    dc = as_array(delta_c, name, xp, device)
    Om_b = as_array(Om, name, xp, device)
    h0_b = as_array(h0, name, xp, device)
    rho_m = Om_b * rhoc0 / h0_b

    s2_M, dln_M = sigma2_M(M, z, Om, Ob, h0, ns, sigma_8, backend=backend,
                           n_k=n_k, n_nodes=n_nodes, return_dln_dlnM=True)
    s2_env = sigma2_M(M_env, z, Om, Ob, h0, ns, sigma_8, backend=backend,
                      n_k=n_k, n_nodes=n_nodes)

    # sigma2_M runs on the device of its own (possibly plain-scalar) inputs;
    # align its outputs with delta_env's device before mixing (torch raises
    # on cross-device products of 0-dim tensors).
    s2_M = as_array(s2_M, name, xp, device)
    dln_M = as_array(dln_M, name, xp, device)
    s2_env = as_array(s2_env, name, xp, device)

    if use_moving_barrier:
        # Direct moving-barrier conditional solution (Davies, Mesinger &
        # Murray 2025, eqs. 2-4) — differentiable port of
        # CHMF.hmf_st_movingbarrier. Works in their own convention (sigma^2
        # at z=0, redshift-dependent barrier delta_crit(z) = delta_c/D(z));
        # dln(sigma)/dlnM is z-independent (see sigma2_M's own docstring), so
        # the z=0 and z-evolved evaluations share the same dln_M computed
        # above -- only a fresh z=0 sigma^2(M) evaluation is needed.
        a_mb, alpha_mb, beta_mb = 0.7, 0.81, 0.34  # Jenkins et al. (2001)
        D1 = as_array(
            growth_factor(1.0 / (1.0 + z), Om, backend=backend, n_nodes=n_nodes),
            name, xp, device,
        )
        delta_crit_z = dc / D1
        x = as_array(
            sigma2_M(M, 0.0, Om, Ob, h0, ns, sigma_8, backend=backend,
                    n_k=n_k, n_nodes=n_nodes),
            name, xp, device,
        )
        sigma2_cond_z0 = s2_env / D1 ** 2
        delta_cond_z0 = delta_env / D1

        S_eff = x - sigma2_cond_z0
        valid = S_eff > 0
        S_safe = xp.where(valid, S_eff, xp.ones_like(S_eff))

        c1 = xp.sqrt(a_mb) * delta_crit_z * beta_mb * (a_mb * delta_crit_z ** 2) ** (-alpha_mb)
        c0_minus = delta_crit_z * (xp.sqrt(a_mb) - 1.0)
        B_x = xp.sqrt(a_mb) * delta_crit_z * (1.0 + beta_mb * (a_mb * delta_crit_z ** 2 / x) ** (-alpha_mb))

        # Taylor expansion of g(x') = B(x') - delta_crit_z about x' = x (the
        # halo's own sigma^2), matching CHMF.hmf_st_movingbarrier exactly.
        T = xp.zeros_like(S_eff)
        falling = 1.0
        fact = 1.0
        for n in range(6):
            if n == 0:
                deriv_n = c0_minus + c1 * x ** alpha_mb
            else:
                falling *= (alpha_mb - (n - 1))
                deriv_n = c1 * falling * x ** (alpha_mb - n)
                fact *= n
            T = T + (-S_eff) ** n / fact * deriv_n

        sigma_x = xp.sqrt(x)
        # d sigma / dM = (dln sigma/dlnM) * sigma / M — analytic, avoids the
        # numpy class's finite-difference derivative.
        dsigma_dM = dln_M * sigma_x / M

        dn_dM = ((rho_m / (math.sqrt(2.0 * math.pi) * M)) * xp.abs(T) * xp.abs(dsigma_dM)
                * (2.0 * sigma_x) / S_safe ** 1.5
                * xp.exp(-(B_x - delta_cond_z0) ** 2 / (2.0 * S_safe)))
        return xp.where(valid, M * dn_dM, xp.zeros_like(dn_dM))

    S_eff = s2_M - s2_env
    valid = S_eff > 0
    S_safe = xp.where(valid, S_eff, xp.ones_like(S_eff))

    # d ln(sigma_eff)/d lnM = (d sigma^2/d lnM) / (2 S_eff), analytic
    dS2_dlnM = 2.0 * s2_M * dln_M
    dln_sigma_eff = dS2_dlnM / (2.0 * S_safe)

    nu_eff = (dc - delta_env) / xp.sqrt(S_safe)
    f_nu = math.sqrt(2.0 / math.pi) * nu_eff * xp.exp(-0.5 * nu_eff ** 2)
    zero = xp.zeros_like(f_nu)
    f_nu = xp.where(nu_eff > 0, f_nu, zero)          # clamp collapsed cells
    f_nu = xp.where(valid, f_nu, zero)               # S_eff <= 0 -> 0

    result = (rho_m / M) * xp.abs(dln_sigma_eff) * f_nu

    if hmf_model_u == 'ST':
        # Barkana & Loeb (2004) hybrid: unconditional ST/PS ratio
        # f_ST/f_PS = A sqrt(q) (1 + (q nu2)^-p) exp(-(q-1) nu2 / 2)
        # with nu2 = (delta_c / sigma(M, z))^2 — closed form, differentiable.
        from ..mass_function.models import _normalise_A
        p_st, q_st, A_st = 0.3, 0.707, _normalise_A(0.3)
        nu2 = dc ** 2 / s2_M
        ratio = (A_st * math.sqrt(q_st) * (1.0 + (q_st * nu2) ** (-p_st))
                 * xp.exp(-0.5 * (q_st - 1.0) * nu2))
        result = result * ratio

    return result


def halo_field_diff(
    delta_env,
    M_env,
    z,
    Om, Ob, h0, ns, sigma_8,
    *,
    cell_volume,
    M_min,
    M_max=None,
    n_mass_bins=40,
    weights='mass',
    eps=None,
    delta_c=1.686,
    backend='numpy',
    n_k=1000,
    n_nodes=512,
    hmf_model='PS',
    chmf_recipe='BarkanaLoeb2004',
):
    """Differentiable halo field via expected-number painting (issue #42, G6).

    Builds the per-cell, per-mass-bin Poisson intensity
    ``lam_b = dn/dlnM(M_b | delta_env) * dlnM_b * V_cell`` from
    :func:`conditional_dndlnm_diff` and returns the halo-count-linear field

        field = sum_b w(M_b) * n_b,
        n_b   = lam_b                          (eps is None — exact expectation)
        n_b   = lam_b + sqrt(lam_b) * eps_b    (reparameterised shot noise)

    The expectation mode is exact (the EPS mean is the Poisson mean) and
    smooth in every argument; the stochastic mode adds Gaussian shot noise
    with the correct Poisson variance while keeping the noise ``eps`` outside
    the graph (reparameterisation trick), so both modes are differentiable
    w.r.t. the cosmology, ``delta_c`` **and the conditioning field itself**.
    The discrete :meth:`CHMFSampler.sample` catalog remains the default for
    mocks and validation.

    Mass bins are log-spaced with geometric-mean centres, mirroring
    :meth:`CHMFSampler._mass_bins` (M_max is capped at 0.999 M_env).

    Args:
        delta_env:  Linear conditioning overdensity — any shape, any backend
                    array (see :meth:`CHMFSampler.sample` for why it must be
                    the linear field smoothed on M_env).
        M_env:      Environmental mass in M_sun (scalar).
        z:          Redshift.
        Om, Ob, h0, ns, sigma_8: cosmological parameters (scalars or 0-dim
                    tensors carrying gradients).
        cell_volume: Cell volume in (Mpc/h)^3, i.e. (Lbox/Ncell)^3.
        M_min:      Minimum halo mass in M_sun.
        M_max:      Maximum halo mass in M_sun (default: 0.999 M_env).
        n_mass_bins: Number of log-spaced mass bins.
        weights:    ``'counts'`` (w = 1, halo number field), ``'mass'``
                    (w = M_b in M_sun, halo mass field; default), or an array
                    of length ``n_mass_bins``.
        eps:        Optional static noise, shape ``(n_mass_bins,) +
                    delta_env.shape`` — draw once outside the graph, e.g.
                    ``rng.standard_normal((n_mass_bins,) + delta.shape)``.
        delta_c:    Linear collapse threshold.
        backend:    ``'numpy'`` (default), ``'jax'`` or ``'torch'``.
        hmf_model:  ``'PS'`` (default) or ``'ST'``, as in
                    :func:`conditional_dndlnm_diff`.
        chmf_recipe: Only meaningful when ``hmf_model='ST'`` — see
                    :func:`conditional_dndlnm_diff`.

    Returns:
        (field, M_centers) — the weighted halo field (backend array, shape of
        ``delta_env``) and the static bin-centre masses (numpy, M_sun).
    """
    from ..cosmo.differentiable import get_backend, device_of, as_array, as_const

    name, xp = get_backend(backend)
    device = device_of(name, xp, delta_env, Om, Ob, h0, ns, sigma_8)
    delta_env = as_array(delta_env, name, xp, device)

    M_hi = 0.999 * M_env if M_max is None else min(M_max, 0.999 * M_env)
    if M_hi <= M_min:
        raise ValueError(
            f"M_env = {M_env:.2e} Msun leaves no mass range above "
            f"M_min = {M_min:.2e} Msun.")
    M_edges = np.logspace(np.log10(M_min), np.log10(M_hi), n_mass_bins + 1)
    M_centers = np.sqrt(M_edges[:-1] * M_edges[1:])
    dln_M = np.diff(np.log(M_edges))

    if isinstance(weights, str):
        if weights == 'counts':
            w_bins = np.ones(n_mass_bins)
        elif weights == 'mass':
            w_bins = M_centers
        else:
            raise ValueError(
                f"weights must be 'counts', 'mass' or an array; got {weights!r}")
    else:
        w_bins = np.asarray(weights, dtype=float)
        if w_bins.shape != (n_mass_bins,):
            raise ValueError(
                f"weights array must have shape ({n_mass_bins},); "
                f"got {w_bins.shape}")

    if eps is not None:
        eps = np.asarray(eps) if not hasattr(eps, 'shape') else eps
        expected = (n_mass_bins,) + tuple(delta_env.shape)
        if tuple(eps.shape) != expected:
            raise ValueError(
                f"eps must have shape {expected}; got {tuple(eps.shape)}")

    field = xp.zeros_like(delta_env)
    for i_M, M in enumerate(M_centers):
        lam = conditional_dndlnm_diff(
            float(M), delta_env, M_env, z, Om, Ob, h0, ns, sigma_8,
            delta_c=delta_c, backend=backend, n_k=n_k, n_nodes=n_nodes,
            hmf_model=hmf_model, chmf_recipe=chmf_recipe,
        ) * (dln_M[i_M] * cell_volume)

        n_b = lam
        if eps is not None:
            eps_b = as_const(np.asarray(eps[i_M]), name, xp, device) \
                if not (name == 'torch' and xp.is_tensor(eps)) else eps[i_M]
            # safe sqrt: gradient-friendly at lam = 0
            lam_pos = lam > 0
            lam_safe = xp.where(lam_pos, lam, xp.ones_like(lam))
            n_b = lam + xp.where(lam_pos, xp.sqrt(lam_safe),
                                 xp.zeros_like(lam)) * eps_b

        field = field + float(w_bins[i_M]) * n_b

    return field, M_centers
