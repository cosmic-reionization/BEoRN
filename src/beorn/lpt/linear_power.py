"""Initial matter power spectrum models.

Hierarchy
---------
PowerSpectrum  (ABC)
├── EisensteinHu       — E&H 1998 fitting function (no-wiggle default, wiggle optional)
└── BoltzmannSolver    — stub / file-loader for CAMB or CLASS output

Factory function ``get_power_spectrum(method, parameters, **kwargs)`` is the
recommended entry point.

Units: k in h/Mpc, P(k) in (Mpc/h)^3.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

from ..cosmo import D


class PowerSpectrum(ABC):
    """Base class for the initial matter power spectrum.

    All subclasses must implement :meth:`transfer`.  :meth:`P` assembles the
    full spectrum ``P(k,z) = A_s * k^n_s * T²(k) * D²(z)`` and normalises
    ``A_s`` to the ``sigma_8`` stored in ``parameters.cosmology``.
    """

    def __init__(self, parameters, method: str = 'eisenstein_hu', backend: str = 'numpy'):
        self.parameters = parameters
        self.method = method
        self.backend = backend
        self._A_s_cache: float | None = None

    @abstractmethod
    def transfer(self, k: np.ndarray) -> np.ndarray:
        """Transfer function T(k), where P_lin(k) ∝ k^n_s T²(k).

        Args:
            k: Wavenumbers in h/Mpc.

        Returns:
            T(k), dimensionless, normalised to 1 at k → 0.
        """

    # ------------------------------------------------------------------
    # Growth factor and normalisation
    # ------------------------------------------------------------------

    def _growth_ratio(self, z: float) -> float:
        """D(z)/D(0)."""
        a = 1.0 / (1.0 + z)
        return D(a, self.parameters) / D(1.0, self.parameters)

    def _compute_A_s(self) -> float:
        """Solve for A_s such that sigma_8 matches parameters.cosmology.sigma_8."""
        sigma_8 = self.parameters.cosmology.sigma_8
        ns = self.parameters.cosmology.ns

        def integrand(lnk: float) -> float:
            k = np.exp(lnk)
            x = k * 8.0  # 8 Mpc/h top-hat radius
            W = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
            return k ** (ns + 3) * self.transfer(k) ** 2 * W ** 2 / (2.0 * np.pi ** 2)

        integral, _ = quad(integrand, np.log(1e-4), np.log(1e3), limit=300)
        return sigma_8 ** 2 / integral

    @property
    def A_s(self) -> float:
        if self._A_s_cache is None:
            self._A_s_cache = self._compute_A_s()
        return self._A_s_cache

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def P(self, k: np.ndarray, z: float = 0.0) -> np.ndarray:
        """Linear matter power spectrum P(k, z) in (Mpc/h)^3.

        Args:
            k: Wavenumbers in h/Mpc.
            z: Redshift.

        Returns:
            P(k, z) in (Mpc/h)^3.
        """
        k = np.asarray(k, dtype=float)
        ns = self.parameters.cosmology.ns
        Dz = self._growth_ratio(z)
        return self.A_s * k ** ns * self.transfer(k) ** 2 * Dz ** 2


# ======================================================================
# Eisenstein & Hu (1998)
# ======================================================================

class EisensteinHu(PowerSpectrum):
    """Eisenstein & Hu (1998) fitting function.

    Args:
        parameters: BEoRN Parameters object.
        wiggle:     If False (default) use the smooth no-wiggle approximation
                    (faster, good for most purposes).  If True use the full
                    transfer function with BAO wiggles.
        backend:    Compute backend ('numpy', 'torch', 'jax').
    """

    def __init__(self, parameters, wiggle: bool = False, backend: str = 'numpy'):
        super().__init__(parameters, method='eisenstein_hu', backend=backend)
        self.wiggle = wiggle
        self._precompute()

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _precompute(self):
        cosmo = self.parameters.cosmology
        h = cosmo.h0
        self._Omh2 = cosmo.Om * h ** 2
        self._Obh2 = cosmo.Ob * h ** 2
        self._h = h
        # T_cmb normalised to 2.7 K (T_cmb field is optional in CosmologyParameters)
        from ..constants import Tcmb0
        T_cmb_K = getattr(cosmo, 'T_cmb', Tcmb0)
        self._Theta = T_cmb_K / 2.7  # dimensionless ratio

        if self.wiggle:
            self._precompute_wiggle()
        else:
            self._precompute_nowiggle()

    def _precompute_nowiggle(self):
        """E&H 1998 Section 4.3, Eqs. 29-31."""
        Omh2, Obh2 = self._Omh2, self._Obh2
        fb = Obh2 / Omh2
        # Sound horizon approximation (Mpc/h)
        self._s_nw = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1.0 + 10.0 * Obh2 ** 0.75)
        # Shape correction factor
        self._alpha_Gamma = (
            1.0
            - 0.328 * np.log(431.0 * Omh2) * fb
            + 0.38  * np.log(22.3 * Omh2)  * fb ** 2
        )

    def _precompute_wiggle(self):
        """E&H 1998 Sections 3–4 quantities."""
        Omh2, Obh2, Theta = self._Omh2, self._Obh2, self._Theta
        # Equality and drag epoch (Eqs. 2, 4)
        z_eq = 2.50e4 * Omh2 * Theta ** -4
        k_eq = 7.46e-2 * Omh2 * Theta ** -2  # h/Mpc
        b1 = 0.313 * Omh2 ** -0.419 * (1.0 + 0.607 * Omh2 ** 0.674)
        b2 = 0.238 * Omh2 ** 0.223
        z_d = (
            1291.0 * Omh2 ** 0.251 / (1.0 + 0.659 * Omh2 ** 0.828)
            * (1.0 + b1 * Obh2 ** b2)
        )
        # Photon-baryon ratio at z_eq and z_d (Eq. 6)
        R_eq = 31.5e-3 * Obh2 * Theta ** -4 * (1e3 / z_eq)
        R_d  = 31.5e-3 * Obh2 * Theta ** -4 * (1e3 / z_d)
        # Sound horizon at drag epoch (Mpc/h, Eq. 6)
        self._s_w = (
            2.0 / (3.0 * k_eq) * np.sqrt(6.0 / R_eq)
            * np.log(
                (np.sqrt(1.0 + R_d) + np.sqrt(R_d + R_eq))
                / (1.0 + np.sqrt(R_eq))
            )
        )
        # Silk damping scale (h/Mpc, Eq. 7)
        self._k_silk = (
            1.6 * Obh2 ** 0.52 * Omh2 ** 0.38
            * (1.0 + (0.43 * z_d * Obh2 ** 0.13 / 1.6) ** 0.84) ** -1
        )
        fb = Obh2 / Omh2
        # CDM growth suppression (Eqs. 15-16)
        a1 = (46.9 * Omh2) ** 0.670 * (1.0 + (32.1 * Omh2) ** -0.532)
        a2 = (12.0 * Omh2) ** 0.424 * (1.0 + (45.0 * Omh2) ** -0.582)
        self._alpha_c = a1 ** (-fb) * a2 ** (-fb ** 3)
        bb1 = 0.944 / (1.0 + (458.0 * Omh2) ** -0.708)
        bb2 = (0.395 * Omh2) ** -0.0266
        self._beta_c = 1.0 / (1.0 + bb1 * ((1.0 - fb) ** bb2 - 1.0))
        # Baryon parameters (Eqs. 15, 24)
        y = z_eq / (1.0 + z_d)
        G = y * (
            -6.0 * np.sqrt(1.0 + y)
            + (2.0 + 3.0 * y) * np.log((np.sqrt(1.0 + y) + 1.0) / (np.sqrt(1.0 + y) - 1.0))
        )
        self._alpha_b = 2.07 * k_eq * self._s_w * (1.0 + R_d) ** (-3.0 / 4.0) * G
        self._beta_b  = 0.5 + fb + (3.0 - 2.0 * fb) * np.sqrt((17.2 * Omh2) ** 2 + 1.0)
        self._beta_node = 8.41 * Omh2 ** 0.435
        self._k_eq = k_eq
        self._fb = fb
        self._fc = 1.0 - fb

    # ------------------------------------------------------------------
    # Transfer functions
    # ------------------------------------------------------------------

    def _transfer_nowiggle(self, k: np.ndarray) -> np.ndarray:
        """E&H 1998 Eqs. 29-31 (smooth, no BAO wiggles)."""
        Omh2, Theta = self._Omh2, self._Theta
        s, alpha_G = self._s_nw, self._alpha_Gamma
        Gamma_eff = Omh2 * (alpha_G + (1.0 - alpha_G) / (1.0 + (0.43 * k * s) ** 4))
        q = k * Theta ** 2 / Gamma_eff
        L0 = np.log(2.0 * np.e + 1.8 * q)
        C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
        return L0 / (L0 + C0 * q ** 2)

    def _T0(self, k: np.ndarray, alpha_c: float, beta_c: float) -> np.ndarray:
        """E&H Eq. 17: master CDM piece."""
        q = k / (13.41 * self._k_eq)
        C = 14.2 / alpha_c + 386.0 / (1.0 + 69.9 * q ** 1.08)
        L = np.log(np.e + 1.8 * beta_c * q)
        return L / (L + C * q ** 2)

    def _transfer_wiggle(self, k: np.ndarray) -> np.ndarray:
        """E&H 1998 Eqs. 17-21 (full fit with BAO wiggles)."""
        s, k_silk = self._s_w, self._k_silk
        fb, fc = self._fb, self._fc
        # CDM
        f  = 1.0 / (1.0 + (k * s / 5.4) ** 4)
        T_c = f * self._T0(k, 1.0, self._beta_c) + (1.0 - f) * self._T0(k, self._alpha_c, 1.0)
        # Baryons
        s_tilde = s / (1.0 + (self._beta_node / (k * s)) ** 3) ** (1.0 / 3.0)
        j0 = np.sinc(k * s_tilde / np.pi)  # sin(x)/x via np.sinc(x/pi)
        T_b = (
            self._T0(k, 1.0, 1.0) / (1.0 + (k * s / 5.2) ** 2)
            + self._alpha_b / (1.0 + (self._beta_b / (k * s)) ** 3)
              * np.exp(-(k / k_silk) ** 1.4)
        ) * j0
        return fb * T_b + fc * T_c

    def transfer(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        if self.wiggle:
            return self._transfer_wiggle(k)
        return self._transfer_nowiggle(k)


# ======================================================================
# Boltzmann solver stub / file loader
# ======================================================================

class BoltzmannSolver(PowerSpectrum):
    """Interface to an external Boltzmann solver (CAMB, CLASS) or a pre-computed P(k) file.

    At minimum, pass ``ps_file`` pointing to a two-column ASCII table with
    columns ``(k [h/Mpc], P(k) [(Mpc/h)^3])`` at z = 0.  Growth-factor
    rescaling to other redshifts uses the same analytic D(z) as the rest of
    BEoRN.

    CAMB/CLASS direct interfaces can be added later by overriding
    :meth:`_load_boltzmann`.

    Args:
        parameters: BEoRN Parameters object.
        solver:     'camb' or 'class' (not yet implemented; reserved for future use).
        ps_file:    Path to a pre-computed P(k) file (z=0).
        backend:    Compute backend ('numpy', 'torch', 'jax').
    """

    def __init__(
        self,
        parameters,
        solver: str = 'camb',
        ps_file: str | None = None,
        backend: str = 'numpy',
    ):
        super().__init__(parameters, method='boltzmann', backend=backend)
        self.solver = solver
        self.ps_file = ps_file
        self._Pk_interp: interp1d | None = None
        if ps_file is not None:
            self._load_from_file(ps_file)

    def _load_from_file(self, path: str):
        data = np.loadtxt(path)
        k_file, Pk_file = data[:, 0], data[:, 1]
        self._Pk_interp = interp1d(
            np.log(k_file), np.log(Pk_file),
            bounds_error=False, fill_value=-np.inf,
        )

    def transfer(self, k: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "BoltzmannSolver.transfer() is not available when using a P(k) file. "
            "Use .P(k, z) directly, or implement a CAMB/CLASS interface."
        )

    def P(self, k: np.ndarray, z: float = 0.0) -> np.ndarray:
        if self._Pk_interp is not None:
            k = np.asarray(k, dtype=float)
            Pk0 = np.exp(self._Pk_interp(np.log(k)))
            return Pk0 * self._growth_ratio(z) ** 2
        raise RuntimeError(
            "No P(k) source loaded.  Pass ps_file= or implement a direct solver call."
        )


# ======================================================================
# Factory
# ======================================================================

def get_power_spectrum(method: str, parameters, **kwargs) -> PowerSpectrum:
    """Return a PowerSpectrum instance for the given method.

    Args:
        method:     One of 'eisenstein_hu', 'eisenstein_hu_wiggle', 'boltzmann'.
        parameters: BEoRN Parameters object.
        **kwargs:   Passed to the subclass constructor.

    Returns:
        Configured PowerSpectrum instance.
    """
    if method == 'eisenstein_hu':
        return EisensteinHu(parameters, wiggle=False, **kwargs)
    if method == 'eisenstein_hu_wiggle':
        return EisensteinHu(parameters, wiggle=True, **kwargs)
    if method == 'boltzmann':
        return BoltzmannSolver(parameters, **kwargs)
    raise ValueError(
        f"Unknown power spectrum method {method!r}. "
        "Choose from 'eisenstein_hu', 'eisenstein_hu_wiggle', 'boltzmann'."
    )
