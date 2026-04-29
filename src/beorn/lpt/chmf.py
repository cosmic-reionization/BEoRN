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
from scipy.interpolate import interp1d

try:
    from numpy import trapezoid as _trapz
except ImportError:
    from numpy import trapz as _trapz

from ..cosmo import D
from ..constants import rhoc0
from ..structs import Parameters, HaloCatalog
from .linear_power import get_power_spectrum, PowerSpectrum

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
        delta_c:     Linear collapse threshold (default 1.686).
        **ps_kwargs: Forwarded to the power spectrum constructor.
    """

    def __init__(
        self,
        parameters: Parameters,
        ps_method: str = 'eisenstein_hu',
        delta_c: float = 1.686,
        **ps_kwargs,
    ):
        self.parameters = parameters
        self.delta_c = delta_c
        self.power_spectrum: PowerSpectrum = get_power_spectrum(
            ps_method, parameters, **ps_kwargs
        )
        self._precompute_sigma()

    # ------------------------------------------------------------------
    # Physical constants in BEoRN units (Mpc/h)
    # ------------------------------------------------------------------

    @property
    def rho_m(self) -> float:
        """Mean comoving matter density in M_sun (Mpc/h)^{-3}.

        rhoc0 = 2.775e11 h^2 Msun/Mpc^3.  Converting to (Mpc/h)^{-3}:
        1 Mpc = h0 Mpc/h  =>  rho_m = Om * rhoc0 / h0.
        """
        return self.parameters.cosmology.Om * rhoc0 / self.parameters.cosmology.h0

    def R_of_M(self, M: float | np.ndarray) -> float | np.ndarray:
        """Top-hat smoothing radius for mass M in Mpc/h."""
        return (3.0 * M / (4.0 * np.pi * self.rho_m)) ** (1.0 / 3.0)

    def M_of_R(self, R: float) -> float:
        """Mass enclosed in top-hat sphere of radius R (Mpc/h) in M_sun."""
        return (4.0 / 3.0) * np.pi * R ** 3 * self.rho_m

    # ------------------------------------------------------------------
    # sigma^2(M) precomputation
    # ------------------------------------------------------------------

    def _precompute_sigma(self) -> None:
        """Build a cubic log-log interpolator for sigma^2(M) at z=0.

        Uses 200 log-spaced mass points over [10^6, 10^17] M_sun and
        integrates k^3 P(k) W^2(kR) / (2 pi^2) over ln k with 1000 k nodes.
        """
        N_k = 1000
        lnk = np.linspace(np.log(1e-4), np.log(1e3), N_k)
        k = np.exp(lnk)

        M_grid = np.logspace(6, 17, 200)
        R_grid = self.R_of_M(M_grid)  # Mpc/h, shape (200,)

        kR = np.outer(k, R_grid)  # (N_k, 200)
        # Top-hat window W(x) = 3(sin x - x cos x)/x^3, Taylor-expanded for x < 1e-3
        W = np.where(
            kR < 1e-3,
            1.0 - kR ** 2 / 10.0 + kR ** 4 / 280.0,
            3.0 * (np.sin(kR) - kR * np.cos(kR)) / kR ** 3,
        )

        Pk = self.power_spectrum.P(k, z=0.0)  # (N_k,)

        # sigma^2 = (1/2pi^2) int k^3 P(k) W^2(kR) d ln k
        integrand = k[:, None] ** 3 * Pk[:, None] * W ** 2 / (2.0 * np.pi ** 2)
        sigma2_grid = _trapz(integrand, lnk, axis=0)  # (200,)

        self._sigma2_interp = interp1d(
            np.log(M_grid),
            np.log(sigma2_grid),
            kind='cubic',
            fill_value='extrapolate',
        )
        logger.debug(
            "CHMF sigma^2 precomputed: "
            "sigma(M=1e12)=%.4f, sigma(M=1e8)=%.4f",
            np.sqrt(np.exp(self._sigma2_interp(np.log(1e12)))),
            np.sqrt(np.exp(self._sigma2_interp(np.log(1e8)))),
        )

    def sigma2_z0(self, M: float | np.ndarray) -> float | np.ndarray:
        """sigma^2(M) at z=0, interpolated from the precomputed grid."""
        M = np.asarray(M, dtype=float)
        return np.exp(self._sigma2_interp(np.log(M)))

    def _D1(self, z: float) -> float:
        """Linear growth factor D(z), normalised to D(0) = 1."""
        a = 1.0 / (1.0 + z)
        return D(a, self.parameters) / D(1.0, self.parameters)

    def sigma2(self, M: float | np.ndarray, z: float) -> float | np.ndarray:
        """sigma^2(M, z) = D1(z)^2 * sigma^2(M, z=0)."""
        return self._D1(z) ** 2 * self.sigma2_z0(M)

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
        delta_c:     Linear collapse threshold (default 1.686).
        **ps_kwargs: Forwarded to the power spectrum constructor.
    """

    def __init__(
        self,
        parameters: Parameters,
        chmf: CHMF | None = None,
        ps_method: str = 'eisenstein_hu',
        delta_c: float = 1.686,
        **ps_kwargs,
    ):
        self.parameters = parameters
        self.chmf = chmf if chmf is not None else CHMF(
            parameters, ps_method, delta_c, **ps_kwargs
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _smooth_field(self, delta: np.ndarray, R_smooth: float) -> np.ndarray:
        """Gaussian-smooth a density field in Fourier space.

        Args:
            delta:    Real-space overdensity, shape (N, N, N).
            R_smooth: Smoothing scale in Mpc/h.

        Returns:
            Smoothed field, same shape.
        """
        N = delta.shape[0]
        L = self.parameters.simulation.Lbox
        dk = 2.0 * np.pi / L
        kvals = np.fft.fftfreq(N, d=1.0 / N) * dk
        kz_vals = np.fft.rfftfreq(N, d=1.0 / N) * dk
        kx, ky, kz = np.meshgrid(kvals, kvals, kz_vals, indexing='ij')
        k2 = kx ** 2 + ky ** 2 + kz ** 2
        W_k = np.exp(-0.5 * k2 * R_smooth ** 2)
        return np.fft.irfftn(np.fft.rfftn(delta) * W_k, s=(N, N, N))

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def sample(
        self,
        delta_field: np.ndarray,
        z: float,
        R_env: float | None = None,
        n_mass_bins: int = 40,
        seed: int | None = None,
    ) -> HaloCatalog:
        """Sample a halo catalog from the conditional HMF.

        Each grid cell is treated as an independent environment with local
        overdensity ``delta_field[i,j,k]``.  The conditional HMF is evaluated
        per-cell and Poisson-sampled; halos are placed at random positions
        within their host cells.

        Args:
            delta_field: Linear matter overdensity at cell resolution,
                shape ``(N, N, N)``.  Typically from :meth:`LPTBase.get_density`.
            z:           Redshift at which to sample.
            R_env:       Environmental smoothing scale in Mpc/h.  If ``None``
                         (default) the cell size is used as the conditioning scale
                         and no additional smoothing is applied.
            n_mass_bins: Number of log-spaced mass bins between
                         ``source.halo_mass_min`` and the environmental mass.
            seed:        Random seed for reproducible Poisson draws and
                         intra-cell position sampling.

        Returns:
            :class:`~beorn.structs.HaloCatalog` with positions in Mpc/h and
            masses in M_sun.

        Raises:
            ValueError: If ``R_env`` is set but ``M_env < halo_mass_min``.
        """
        params = self.parameters
        N = params.simulation.Ncell
        L = params.simulation.Lbox
        cell_size = L / N
        V_cell = cell_size ** 3

        # ── Environmental scale and sigma ──────────────────────────────
        if R_env is None:
            # Condition on each cell: M_env = rho_m * V_cell
            M_env = self.chmf.rho_m * V_cell
            delta_env = delta_field
        else:
            M_env = self.chmf.M_of_R(R_env)
            if R_env > cell_size:
                delta_env = self._smooth_field(delta_field, R_env)
            else:
                delta_env = delta_field

        sigma2_env = float(self.chmf.sigma2(M_env, z))

        # ── Mass range ─────────────────────────────────────────────────
        M_min = params.source.halo_mass_min
        M_max_req = params.source.halo_mass_max

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
                f"(cell mass for N={N}, L={L} Mpc/h).  "
                f"Capping M_max at {M_max:.2e} Msun.  "
                f"Increase R_env or use a coarser grid to access higher masses.",
                stacklevel=2,
            )

        M_edges = np.logspace(np.log10(M_min), np.log10(M_max), n_mass_bins + 1)
        M_centers = np.sqrt(M_edges[:-1] * M_edges[1:])
        dln_M = np.diff(np.log(M_edges))  # uniform in log space

        # ── Poisson sampling ───────────────────────────────────────────
        rng = np.random.default_rng(seed)
        positions_list: list[np.ndarray] = []
        masses_list: list[np.ndarray] = []

        for i_M, M in enumerate(M_centers):
            n_cond = self.chmf.hmf_chmf_field(M, delta_env, sigma2_env, z)
            N_expected = np.maximum(n_cond * dln_M[i_M] * V_cell, 0.0)
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
