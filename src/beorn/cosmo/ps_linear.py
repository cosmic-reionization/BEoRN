"""Linear matter power spectrum models, shared by ``lpt``, ``mass_function``
and ``halo_model``.

Hierarchy
---------
PowerSpectrum  (ABC)
├── EisensteinHu            — E&H 1998 fitting function (default; no-wiggle, wiggle optional)
├── BoltzmannSolver         — z=0 table from a P(k) file; base for the solvers below
│   ├── CAMBPowerSpectrum   — CAMB  (``ps_method='camb'``)
│   └── CLASSPowerSpectrum  — CLASS (``ps_method='class'``, via classy)
├── DiscoEB                 — DISCO-EB JAX Boltzmann solver
└── TabulatedPowerSpectrum  — user-supplied (k, P) table

Factory ``get_power_spectrum(method, parameters, **ps_var)`` is the
recommended entry point; ``method=None`` selects Eisenstein & Hu and
``ps_var`` keywords are forwarded to the Boltzmann solver.

Units: k in h/Mpc, P(k) in (Mpc/h)^3.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import math
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

from .background import D
from .differentiable import (
    get_backend, device_of, as_const, as_array, trapz_static, growth_factor,
)
from ..constants import Tcmb0

__all__ = [
    'PowerSpectrum',
    'EisensteinHu',
    'BoltzmannSolver',
    'CAMBPowerSpectrum',
    'CLASSPowerSpectrum',
    'DiscoEB',
    'TabulatedPowerSpectrum',
    'get_power_spectrum',
    'transfer_eh_nowiggle',
    'sigma8_normalisation',
    'pk_eh_nowiggle',
]


# ======================================================================
# Differentiable, backend-generic E&H no-wiggle P(k)  (numpy/jax/torch)
#
# Pure-function counterparts of the EisensteinHu class below — same
# formulas, but every operation runs in the chosen backend, so the result
# is differentiable w.r.t. ALL cosmological parameters (Om, Ob, h0, ns,
# sigma_8, z) and device-resident (GPU-capable). The numpy class path is
# unchanged; these are the opt-in gpu/diff route (issue #42, Phase 1: G1).
# ======================================================================

def transfer_eh_nowiggle(k, Om, Ob, h0, backend='numpy', Theta=Tcmb0 / 2.7):
    """E&H 1998 no-wiggle transfer function T(k; Om, Ob, h0).

    Same formulas as ``EisensteinHu._transfer_nowiggle`` (Eqs. 26, 28-31).
    k in h/Mpc. Pass jax tracers / torch tensors for gradients; tensors on
    GPU stay on GPU.
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, k, Om, Ob, h0)
    k = as_array(k, name, xp, device)
    Om = as_array(Om, name, xp, device)
    Ob = as_array(Ob, name, xp, device)
    h0 = as_array(h0, name, xp, device)

    Omh2 = Om * h0 ** 2
    Obh2 = Ob * h0 ** 2
    fb = Obh2 / Omh2
    s = 44.5 * xp.log(9.83 / Omh2) / xp.sqrt(1.0 + 10.0 * Obh2 ** 0.75)
    alpha_G = (1.0
               - 0.328 * xp.log(431.0 * Omh2) * fb
               + 0.38 * xp.log(22.3 * Omh2) * fb ** 2)
    Gamma_eff = (Omh2 / h0) * (alpha_G + (1.0 - alpha_G)
                               / (1.0 + (0.43 * k * s) ** 4))
    q = k * Theta ** 2 / Gamma_eff
    L0 = xp.log(2.0 * math.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return L0 / (L0 + C0 * q ** 2)


def sigma8_normalisation(Om, Ob, h0, ns, sigma_8, backend='numpy',
                         n_k=1024, Theta=Tcmb0 / 2.7):
    """A_s such that sigma(R=8 Mpc/h, z=0) = sigma_8 (E&H no-wiggle).

    Fixed-node log-k trapezoid over k in [1e-4, 1e3] h/Mpc — replaces
    scipy.quad so gradients flow through every parameter. Agrees with
    ``PowerSpectrum._compute_A_s`` to ~1e-6 relative at n_k=1024.
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, Om, Ob, h0, ns, sigma_8)
    lnk_np = np.linspace(np.log(1e-4), np.log(1e3), n_k)
    k = as_const(np.exp(lnk_np), name, xp, device)
    ns = as_array(ns, name, xp, device)
    sigma_8 = as_array(sigma_8, name, xp, device)
    x = k * 8.0
    W = 3.0 * (xp.sin(x) - x * xp.cos(x)) / x ** 3
    T = transfer_eh_nowiggle(k, Om, Ob, h0, backend=backend, Theta=Theta)
    integrand = k ** (ns + 3.0) * T ** 2 * W ** 2 / (2.0 * math.pi ** 2)
    integral = trapz_static(integrand, lnk_np, name, xp)
    return sigma_8 ** 2 / integral


def pk_eh_nowiggle(k, z, Om, Ob, h0, ns, sigma_8, backend='numpy',
                   n_k=1024, n_nodes=512, Theta=Tcmb0 / 2.7):
    """Linear P(k, z) [(Mpc/h)^3], E&H no-wiggle, sigma_8-normalised.

    Pure and differentiable w.r.t. every cosmological argument
    (Om, Ob, h0, ns, sigma_8) and z, in numpy / jax / torch; GPU-capable
    (computation happens on the device of the input tensors).

    Example (jax)::

        import jax
        dP_dOm = jax.jacobian(
            lambda Om: pk_eh_nowiggle(k, 7.0, Om, 0.049, 0.673, 0.963, 0.811,
                                      backend='jax')
        )(0.315)
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, k, Om, Ob, h0, ns, sigma_8)
    k = as_array(k, name, xp, device)
    ns = as_array(ns, name, xp, device)
    A_s = sigma8_normalisation(Om, Ob, h0, ns, sigma_8, backend=backend,
                               n_k=n_k, Theta=Theta)
    T = transfer_eh_nowiggle(k, Om, Ob, h0, backend=backend, Theta=Theta)
    a = 1.0 / (1.0 + z)
    Dz = growth_factor(a, Om, backend=backend, n_nodes=n_nodes)
    return A_s * k ** ns * T ** 2 * Dz ** 2


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
        ns = self.parameters.cosmology.ns
        Dz = self._growth_ratio(z)
        if self.backend not in (None, 'numpy'):
            # backend path: keep k as a jax/torch array (device-resident);
            # A_s and Dz are float constants — see pk_eh_nowiggle for the
            # fully differentiable pure-function API.
            name, xp = get_backend(self.backend)
            k = as_array(k, name, xp)
            return self.A_s * k ** ns * self.transfer(k) ** 2 * Dz ** 2
        k = np.asarray(k, dtype=float)
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

    def _compute_A_s(self) -> float:
        if self.backend not in (None, 'numpy') and not self.wiggle:
            # fixed-node quadrature on the backend (same result as scipy.quad
            # to ~1e-6 relative); avoids per-point dispatch through quad
            c = self.parameters.cosmology
            return float(sigma8_normalisation(
                c.Om, c.Ob, c.h0, c.ns, c.sigma_8,
                backend=self.backend, Theta=self._Theta))
        return super()._compute_A_s()

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
        k_eq = 7.46e-2 * Omh2 * Theta ** -2  # Mpc^{-1} (physical comoving, not h/Mpc)
        b1 = 0.313 * Omh2 ** -0.419 * (1.0 + 0.607 * Omh2 ** 0.674)
        b2 = 0.238 * Omh2 ** 0.223
        z_d = (
            1291.0 * Omh2 ** 0.251 / (1.0 + 0.659 * Omh2 ** 0.828)
            * (1.0 + b1 * Obh2 ** b2)
        )
        # Photon-baryon ratio at z_eq and z_d (Eq. 6)
        R_eq = 31.5 * Obh2 * Theta ** -4 * (1e3 / z_eq)
        R_d  = 31.5 * Obh2 * Theta ** -4 * (1e3 / z_d)
        # Sound horizon at drag epoch (Mpc, Eq. 6 — E&H formula yields physical Mpc)
        self._s_w = (
            2.0 / (3.0 * k_eq) * np.sqrt(6.0 / R_eq)
            * np.log(
                (np.sqrt(1.0 + R_d) + np.sqrt(R_d + R_eq))
                / (1.0 + np.sqrt(R_eq))
            )
        )
        # Silk damping scale (Mpc^{-1}, E&H Eq. 7 fit)
        self._k_silk = (
            1.6 * Obh2 ** 0.52 * Omh2 ** 0.73
            * (1.0 + (10.4 * Omh2) ** (-0.95))
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
        # E&H Eq. 30: Γ_eff = Ωm h × {…}, not Ωm h²
        Gamma_eff = (Omh2 / self._h) * (alpha_G + (1.0 - alpha_G) / (1.0 + (0.43 * k * s) ** 4))
        q = k * Theta ** 2 / Gamma_eff
        L0 = np.log(2.0 * np.e + 1.8 * q)
        C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
        return L0 / (L0 + C0 * q ** 2)

    def _T0(self, k_mpc: np.ndarray, alpha_c: float, beta_c: float) -> np.ndarray:
        """E&H Eq. 17: master CDM piece. k_mpc must be in Mpc^{-1}."""
        q = k_mpc / (13.41 * self._k_eq)
        C = 14.2 / alpha_c + 386.0 / (1.0 + 69.9 * q ** 1.08)
        L = np.log(np.e + 1.8 * beta_c * q)
        return L / (L + C * q ** 2)

    def _transfer_wiggle(self, k: np.ndarray) -> np.ndarray:
        """E&H 1998 Eqs. 17-21 (full fit with BAO wiggles).

        The E&H wiggle formula quantities (k_eq, s, k_silk) are derived from
        physical Boltzmann-code integrals and have units of Mpc^{-1} / Mpc.
        Input k is in h/Mpc, so we convert to Mpc^{-1} here.
        """
        # k_eq [Mpc^{-1}], s_w [Mpc], k_silk [Mpc^{-1}] — convert input to match
        k = k * self._h  # h/Mpc → Mpc^{-1}
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
        if self.backend not in (None, 'numpy'):
            if self.wiggle:
                raise NotImplementedError(
                    "wiggle=True is numpy-only for now; use wiggle=False with "
                    f"backend='{self.backend}' (differentiable no-wiggle path)."
                )
            c = self.parameters.cosmology
            return transfer_eh_nowiggle(k, c.Om, c.Ob, c.h0,
                                        backend=self.backend, Theta=self._Theta)
        k = np.asarray(k, dtype=float)
        if self.wiggle:
            return self._transfer_wiggle(k)
        return self._transfer_nowiggle(k)


# ======================================================================
# DISCO-EB: differentiable JAX Boltzmann solver
# ======================================================================

class DiscoEB(PowerSpectrum):
    """Linear power spectrum via the DISCO-EB JAX Boltzmann solver.

    DISCO-EB (https://github.com/ohahn/DISCO-EB) solves the full
    linearised Boltzmann hierarchy in JAX, making P(k) differentiable
    w.r.t. all cosmological parameters.

    Install::

        pip install git+https://github.com/ohahn/DISCO-EB.git

    The background + perturbation ODE is solved once at ``__init__``
    time (z = 0).  JIT compilation on the first call takes ~1 min;
    subsequent calls take ~3 s on GPU or ~30 s on CPU.  Redshift
    scaling uses BEoRN's analytic growth factor D(z).

    Units: k in h/Mpc, P(k) in (Mpc/h)^3 — consistent with all other
    BEoRN power spectrum classes.

    Args:
        parameters: BEoRN Parameters object.
        num_k:      Number of k-modes passed to ``evolve_perturbations``
                    (default 512).
        kmax_hmpc:  Maximum wavenumber in h/Mpc (default 1.0).  DISCO-EB
                    solves the full Boltzmann hierarchy without the
                    tight-coupling approximation used by CLASS/CAMB, so
                    the photon-baryon ODE at k ≳ 1 Mpc⁻¹ requires
                    O(k × 2800) adaptive steps to resolve pre-recombination
                    oscillations.  1 h/Mpc (≈ 0.67 Mpc⁻¹) needs ~1900 steps
                    and is safe with the default ``max_steps=8192``.  To
                    reach 10 h/Mpc you would need ``max_steps ≥ 20 000``.
        thermo_module: Thermodynamics module — ``'RECFAST'`` (default)
                    or ``'MB95'``.
        rtol, atol: ODE solver tolerances (default 1e-3 each).
        max_steps:  Maximum ODE steps per k-mode (default 8192).  Rule of
                    thumb: max_steps ≳ 2800 × kmax_hmpc × h.
    """

    def __init__(
        self,
        parameters,
        num_k: int = 512,
        kmax_hmpc: float = 1.0,
        thermo_module: str = 'RECFAST',
        rtol: float = 1e-3,
        atol: float = 1e-3,
        max_steps: int = 8192,
        backend: str = 'numpy',
    ):
        super().__init__(parameters, method='disco_eb', backend=backend)
        self._num_k = num_k
        self._kmax_hmpc = kmax_hmpc
        self._thermo_module = thermo_module
        self._rtol = rtol
        self._atol = atol
        self._max_steps = max_steps
        self._Pk_interp_log = None
        self._precompute_discoeb()

    # ------------------------------------------------------------------

    def _build_param_dict(self) -> dict:
        cosmo = self.parameters.cosmology
        from ..constants import Tcmb0
        return {
            'Omegam'  : float(cosmo.Om),
            'Omegab'  : float(cosmo.Ob),
            'w_DE_0'  : -1.0,
            'w_DE_a'  : 0.0,
            'cs2_DE'  : 1.0,
            'Omegak'  : 0.0,
            'A_s'     : 2.1e-9,        # preliminary; rescaled below
            'n_s'     : float(cosmo.ns),
            'H0'      : float(cosmo.h0) * 100.0,   # km/s/Mpc
            'Tcmb'    : float(getattr(cosmo, 'T_cmb', Tcmb0)),
            'YHe'     : 0.248,
            'Neff'    : 2.046,
            'Nmnu'    : 0,
            'mnu'     : 0.0,
            'k_p'     : 0.05,          # pivot scale in 1/Mpc
        }

    def _precompute_discoeb(self):
        try:
            from discoeb.background import evolve_background
            from discoeb.perturbations import evolve_perturbations, get_power
        except ImportError as exc:
            raise ImportError(
                "disco-eb is required for DiscoEB.\n"
                "Install with: pip install git+https://github.com/ohahn/DISCO-EB.git"
            ) from exc

        import jax.numpy as jnp

        cosmo = self.parameters.cosmology
        h = float(cosmo.h0)

        param = self._build_param_dict()
        param = evolve_background(param=param, thermo_module=self._thermo_module)

        # k range in 1/Mpc (DISCO-EB convention); BEoRN uses h/Mpc
        # kmax is kept modest (~10 h/Mpc → ~6.7 Mpc⁻¹) to avoid
        # oscillatory high-k ODE modes that exhaust max_steps.
        aexp_out = jnp.array([1.0])   # z = 0
        y, kmodes, param = evolve_perturbations(
            param=param,
            kmin=1e-4 * h,
            kmax=self._kmax_hmpc * h,
            num_k=self._num_k,
            aexp_out=aexp_out,
            rtol=self._rtol,
            atol=self._atol,
            max_steps=self._max_steps,
        )

        # idx=4 → total matter power spectrum (CDM + baryons)
        Pk_1mpc = np.asarray(get_power(k=kmodes, y=y[:, 0, :], idx=4, param=param))
        k_1mpc  = np.asarray(kmodes)

        # Convert units: k [1/Mpc] → [h/Mpc],  P [Mpc³] → [(Mpc/h)³]
        k_hmpc  = k_1mpc / h
        Pk_hmpc = Pk_1mpc * h ** 3

        # Rescale A_s to match sigma_8 (P ∝ A_s, so P ∝ sigma_8²)
        sigma8_raw = self._compute_sigma8_from_Pk(k_hmpc, Pk_hmpc)
        Pk_hmpc *= (cosmo.sigma_8 / sigma8_raw) ** 2

        self._k_grid  = k_hmpc
        self._Pk_grid = Pk_hmpc
        self._Pk_interp_log = interp1d(
            np.log(k_hmpc), np.log(Pk_hmpc),
            kind='cubic', bounds_error=False, fill_value='extrapolate',
        )

    @staticmethod
    def _compute_sigma8_from_Pk(k_hmpc: np.ndarray, Pk_hmpc: np.ndarray) -> float:
        from scipy.integrate import quad as _quad
        log_Pk = interp1d(np.log(k_hmpc), np.log(Pk_hmpc),
                          kind='cubic', bounds_error=False, fill_value=-np.inf)

        def integrand(lnk: float) -> float:
            k = np.exp(lnk)
            x = k * 8.0
            W = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
            return k ** 3 * np.exp(log_Pk(lnk)) * W ** 2 / (2.0 * np.pi ** 2)

        result, _ = _quad(integrand, np.log(1e-4), np.log(1e2), limit=300)
        return float(np.sqrt(result))

    # ------------------------------------------------------------------

    def transfer(self, k: np.ndarray) -> np.ndarray:
        # Full transfer function not independently available from DISCO-EB output;
        # derive from P(k, z=0) = A_s k^n_s T²(k) D²(0) = A_s k^n_s T²(k)
        k = np.asarray(k, dtype=float)
        Pk0 = np.exp(self._Pk_interp_log(np.log(k)))
        ns  = self.parameters.cosmology.ns
        T2  = Pk0 / (self.A_s * k ** ns)
        return np.sqrt(np.maximum(T2, 0.0))

    def P(self, k: np.ndarray, z: float = 0.0) -> np.ndarray:
        k   = np.asarray(k, dtype=float)
        Pk0 = np.exp(self._Pk_interp_log(np.log(k)))
        Dz  = self._growth_ratio(z)
        return Pk0 * Dz ** 2


# ======================================================================
# Boltzmann solvers (CAMB, CLASS) and file loader
# ======================================================================

def _sigma8_of_table(k: np.ndarray, Pk: np.ndarray) -> float:
    """sigma(R = 8 Mpc/h) from a tabulated P(k) (log-k trapezoid)."""
    x = k * 8.0
    W = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
    integrand = k ** 3 * Pk * W ** 2 / (2.0 * np.pi ** 2)
    try:
        sigma2 = np.trapezoid(integrand, np.log(k))
    except AttributeError:
        sigma2 = np.trapz(integrand, np.log(k))
    return float(np.sqrt(sigma2))


class BoltzmannSolver(PowerSpectrum):
    """Linear P(k) tabulated at z = 0 by a Boltzmann solver or read from file.

    Subclasses fill the z = 0 table in :meth:`_compute_table`; this base
    class handles sigma_8 normalisation, interpolation and redshift scaling.
    Used directly, it loads ``ps_file`` — a two-column ASCII table
    ``(k [h/Mpc], P(k) [(Mpc/h)^3])`` at z = 0.

    Outside the tabulated k-range the spectrum is continued with the
    Eisenstein & Hu no-wiggle shape, matched in amplitude at the table
    edges, so sigma^2(M) integrals out to k ~ 1e3 h/Mpc stay well behaved
    even when the solver is only run to modest ``kmax``.

    Redshift dependence: ``P(k, z) = P(k, 0) D^2(z)/D^2(0)`` with BEoRN's
    growth factor, consistent with the mass function and LPT modules.

    Args:
        parameters:        BEoRN Parameters object.
        solver:            Kept for backwards compatibility. ``'camb'`` or
                           ``'class'`` without ``ps_file`` is better served by
                           ``get_power_spectrum('camb' | 'class', ...)``.
        ps_file:           Path to a pre-computed P(k) file (z = 0).
        normalise_sigma8:  Rescale the table so that sigma_8 matches
                           ``parameters.cosmology.sigma_8`` (default True for
                           solvers, False for files).
        backend:           Compute backend ('numpy', 'torch', 'jax').
    """

    def __init__(
        self,
        parameters,
        solver: str = 'file',
        ps_file: str | None = None,
        normalise_sigma8: bool | None = None,
        backend: str = 'numpy',
    ):
        super().__init__(parameters, method='boltzmann', backend=backend)
        self.solver = solver
        self.ps_file = ps_file
        self._Pk_interp: interp1d | None = None
        if normalise_sigma8 is None:
            normalise_sigma8 = ps_file is None
        self.normalise_sigma8 = normalise_sigma8

        if ps_file is not None:
            data = np.loadtxt(ps_file)
            k_tab, Pk_tab = data[:, 0], data[:, 1]
        else:
            k_tab, Pk_tab = self._compute_table()
        self._set_table(np.asarray(k_tab, float), np.asarray(Pk_tab, float))

    # ------------------------------------------------------------------

    def _compute_table(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(k [h/Mpc], P(k, z=0) [(Mpc/h)^3])``. Override in subclasses."""
        raise RuntimeError(
            "No P(k) source. Pass ps_file=, or use get_power_spectrum('camb' | 'class', ...)."
        )

    def _set_table(self, k: np.ndarray, Pk: np.ndarray) -> None:
        order = np.argsort(k)
        k, Pk = k[order], Pk[order]
        if self.normalise_sigma8:
            Pk = Pk * (self.parameters.cosmology.sigma_8 / _sigma8_of_table(k, Pk)) ** 2
        self.k_table, self.Pk_table = k, Pk
        self._Pk_interp = interp1d(np.log(k), np.log(Pk), kind='cubic')
        # E&H no-wiggle continuation beyond the table edges
        self._eh = EisensteinHu(self.parameters, wiggle=False)
        self._lo_ratio = Pk[0] / self._eh.P(k[0])
        self._hi_ratio = Pk[-1] / self._eh.P(k[-1])

    def _P0(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        kmin, kmax = self.k_table[0], self.k_table[-1]
        inside = (k >= kmin) & (k <= kmax)
        out = np.empty_like(k)
        out[inside] = np.exp(self._Pk_interp(np.log(k[inside])))
        lo, hi = k < kmin, k > kmax
        if lo.any():
            out[lo] = self._lo_ratio * self._eh.P(k[lo])
        if hi.any():
            out[hi] = self._hi_ratio * self._eh.P(k[hi])
        return out

    def transfer(self, k: np.ndarray) -> np.ndarray:
        """T(k) recovered from the table: ``sqrt(P(k, 0) / (A k^n_s))``.

        The amplitude ``A`` is fixed by T -> 1 at the lowest tabulated k.
        """
        k = np.asarray(k, dtype=float)
        ns = self.parameters.cosmology.ns
        k0 = self.k_table[0]
        A = self.Pk_table[0] / k0 ** ns
        return np.sqrt(self._P0(k) / (A * k ** ns))

    def P(self, k: np.ndarray, z: float = 0.0) -> np.ndarray:
        k_arr = np.asarray(k, dtype=float)
        Pk = self._P0(k_arr.ravel()).reshape(k_arr.shape) * self._growth_ratio(z) ** 2
        if self.backend not in (None, 'numpy'):
            name, xp = get_backend(self.backend)
            return as_array(Pk, name, xp)
        return Pk


class CAMBPowerSpectrum(BoltzmannSolver):
    """Linear matter P(k) from CAMB (``pip install camb``).

    BEoRN cosmology (``Om, Ob, h0, ns``) sets ``H0, ombh2, omch2, ns``.
    Every keyword in ``ps_var`` is forwarded to :func:`camb.set_params`
    and overrides these defaults — e.g. ``mnu=0.06``, ``tau=0.054``,
    ``As=2.1e-9``, ``halofit_version=...``, ``Want_CMB=False``.

    By default the result is rescaled to ``parameters.cosmology.sigma_8``;
    pass ``normalise_sigma8=False`` to keep CAMB's ``As`` normalisation.

    Args:
        parameters:        BEoRN Parameters object.
        kmax:              Largest tabulated k [h/Mpc] (default 100).
        kmin:              Smallest tabulated k [h/Mpc] (default 1e-4).
        nk:                Number of tabulated k points (default 512).
        normalise_sigma8:  See above (default True).
        backend:           Output array backend.
        **ps_var:          Forwarded to ``camb.set_params``.
    """

    def __init__(
        self,
        parameters,
        kmax: float = 100.0,
        kmin: float = 1e-4,
        nk: int = 512,
        normalise_sigma8: bool = True,
        backend: str = 'numpy',
        **ps_var,
    ):
        self.kmax, self.kmin, self.nk = kmax, kmin, nk
        self.ps_var = ps_var
        super().__init__(parameters, solver='camb',
                         normalise_sigma8=normalise_sigma8, backend=backend)
        self.method = 'camb'

    def _compute_table(self):
        try:
            import camb
        except ImportError as exc:
            raise ImportError("CAMB is required for ps_method='camb'. Install with: pip install camb") from exc

        c = self.parameters.cosmology
        h = float(c.h0)
        settings = dict(
            H0=100.0 * h,
            ombh2=float(c.Ob) * h ** 2,
            omch2=float(c.Om - c.Ob) * h ** 2,
            ns=float(c.ns),
            As=2.1e-9,
            mnu=0.0,
            omk=0.0,
            WantTransfer=True,
            WantCls=False,
            redshifts=[0.0],
            kmax=self.kmax * h * 1.1,   # CAMB kmax is in 1/Mpc
        )
        settings.update(self.ps_var)
        pars = camb.set_params(**settings)
        results = camb.get_results(pars)
        kh, _, pk = results.get_matter_power_spectrum(
            minkh=self.kmin, maxkh=self.kmax, npoints=self.nk)
        return kh, pk[0]


class CLASSPowerSpectrum(BoltzmannSolver):
    """Linear matter P(k) from CLASS via ``classy`` (``pip install classy``).

    BEoRN cosmology sets ``h, omega_b, omega_cdm, n_s``. Every keyword in
    ``ps_var`` is passed to ``classy.Class.set`` and overrides these
    defaults — e.g. ``N_ncdm=1, m_ncdm=0.06``, ``YHe=0.245``, or
    ``sigma8=...`` (in which case ``A_s`` is dropped).

    By default the result is rescaled to ``parameters.cosmology.sigma_8``;
    pass ``normalise_sigma8=False`` to keep CLASS's normalisation.

    Args:
        parameters:        BEoRN Parameters object.
        kmax:              Largest tabulated k [h/Mpc] (default 100).
        kmin:              Smallest tabulated k [h/Mpc] (default 1e-4).
        nk:                Number of tabulated k points (default 512).
        normalise_sigma8:  See above (default True).
        backend:           Output array backend.
        **ps_var:          Forwarded to ``classy.Class.set``.
    """

    def __init__(
        self,
        parameters,
        kmax: float = 100.0,
        kmin: float = 1e-4,
        nk: int = 512,
        normalise_sigma8: bool = True,
        backend: str = 'numpy',
        **ps_var,
    ):
        self.kmax, self.kmin, self.nk = kmax, kmin, nk
        self.ps_var = ps_var
        super().__init__(parameters, solver='class',
                         normalise_sigma8=normalise_sigma8, backend=backend)
        self.method = 'class'

    def _compute_table(self):
        try:
            from classy import Class
        except ImportError as exc:
            raise ImportError("classy is required for ps_method='class'. Install with: pip install classy") from exc

        c = self.parameters.cosmology
        h = float(c.h0)
        settings = {
            'output': 'mPk',
            'P_k_max_h/Mpc': self.kmax * 1.1,
            'z_max_pk': 0.0,
            'h': h,
            'omega_b': float(c.Ob) * h ** 2,
            'omega_cdm': float(c.Om - c.Ob) * h ** 2,
            'n_s': float(c.ns),
            'A_s': 2.1e-9,
        }
        settings.update(self.ps_var)
        if any(key in self.ps_var for key in ('sigma8', 'ln10^{10}A_s', 'ln_A_s_1e10')):
            settings.pop('A_s')

        cosmo = Class()
        cosmo.set(settings)
        cosmo.compute()
        try:
            kh = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.nk)
            pk = np.array([cosmo.pk_lin(ki * h, 0.0) for ki in kh]) * h ** 3
        finally:
            cosmo.struct_cleanup()
            cosmo.empty()
        return kh, pk


# ======================================================================
# Tabulated power spectrum
# ======================================================================

class TabulatedPowerSpectrum(PowerSpectrum):
    """Linear power spectrum interpolated from a user-supplied (k, P(k)) table.

    Typical usage: pass a P(k, z=0) array from colossus, CAMB, or any
    external code and use it directly inside BEoRN's HMF machinery.

    Args:
        parameters:  BEoRN Parameters object.
        k:           Wavenumbers in h/Mpc, 1-D, sorted ascending.
        Pk:          P(k) in (Mpc/h)^3 at redshift ``z_ref``.
        z_ref:       Redshift of the input table (default 0).  If non-zero
                     the table is divided by D²(z_ref)/D²(0) to give P(k,z=0).
        renormalize: If True (default) rescale the amplitude so that σ₈
                     computed from the interpolated table matches
                     ``parameters.cosmology.sigma_8``.  Set to False to
                     use the input normalization verbatim.
    """

    def __init__(
        self,
        parameters,
        k: np.ndarray,
        Pk: np.ndarray,
        z_ref: float = 0.0,
        renormalize: bool = True,
    ):
        super().__init__(parameters, method='tabulated')
        k  = np.asarray(k,  dtype=float)
        Pk = np.asarray(Pk, dtype=float)

        if z_ref != 0.0:
            Pk = Pk / self._growth_ratio(z_ref) ** 2

        if renormalize:
            Pk = Pk * self._sigma8_rescale(k, Pk)

        self._Pk0_interp = interp1d(
            np.log(k), np.log(Pk),
            kind='linear', bounds_error=False,
            fill_value='extrapolate',
        )

    def _sigma8_rescale(self, k: np.ndarray, Pk: np.ndarray) -> float:
        """Amplitude multiplier so that σ₈(k, Pk) matches the target."""
        target = self.parameters.cosmology.sigma_8
        x = k * 8.0
        W = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
        integrand = k ** 3 * Pk * W ** 2 / (2.0 * np.pi ** 2)
        try:
            sigma2 = np.trapezoid(integrand, np.log(k))
        except AttributeError:
            sigma2 = np.trapz(integrand, np.log(k))
        return target ** 2 / sigma2

    def transfer(self, k: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "TabulatedPowerSpectrum does not expose a transfer function. "
            "Use .P(k, z) directly."
        )

    def P(self, k: np.ndarray, z: float = 0.0) -> np.ndarray:
        """Interpolated P(k, z) in (Mpc/h)^3."""
        k = np.asarray(k, dtype=float)
        Pk0 = np.exp(self._Pk0_interp(np.log(k)))
        return Pk0 * self._growth_ratio(z) ** 2


# ======================================================================
# Factory
# ======================================================================

_METHOD_ALIASES = {
    None: 'eisenstein_hu',
    'eh': 'eisenstein_hu',
    'eisenstein_hu': 'eisenstein_hu',
    'eh_wiggle': 'eisenstein_hu_wiggle',
    'eisenstein_hu_wiggle': 'eisenstein_hu_wiggle',
    'camb': 'camb',
    'class': 'class',
    'classy': 'class',
    'boltzmann': 'boltzmann',
    'disco_eb': 'disco_eb',
    'tabulated': 'tabulated',
}


def get_power_spectrum(method: str | None, parameters, **ps_var) -> PowerSpectrum:
    """Return a linear :class:`PowerSpectrum` for the given method.

    Args:
        method:     Case-insensitive. ``None`` / ``'eh'`` / ``'eisenstein_hu'``
                    (default E&H no-wiggle), ``'eisenstein_hu_wiggle'``,
                    ``'camb'``, ``'class'``, ``'boltzmann'`` (file loader,
                    ``ps_file=``), ``'disco_eb'``, ``'tabulated'``.
        parameters: BEoRN Parameters object.
        **ps_var:   Forwarded to the class constructor. For ``'camb'`` /
                    ``'class'``, keywords that are not constructor arguments
                    (``kmax``, ``nk``, ``normalise_sigma8``, ``backend``) go
                    straight to the Boltzmann solver.

    Returns:
        Configured PowerSpectrum instance.

    Example::

        ps = get_power_spectrum('camb', param, mnu=0.06, kmax=50)
        P  = ps.P(k, z=8.0)
    """
    key = method.lower() if isinstance(method, str) else method
    if key not in _METHOD_ALIASES:
        raise ValueError(
            f"Unknown power spectrum method {method!r}. Choose from None, "
            "'eisenstein_hu', 'eisenstein_hu_wiggle', 'camb', 'class', "
            "'boltzmann', 'disco_eb', 'tabulated'."
        )
    key = _METHOD_ALIASES[key]
    if key == 'eisenstein_hu':
        return EisensteinHu(parameters, wiggle=False, **ps_var)
    if key == 'eisenstein_hu_wiggle':
        return EisensteinHu(parameters, wiggle=True, **ps_var)
    if key == 'camb':
        return CAMBPowerSpectrum(parameters, **ps_var)
    if key == 'class':
        return CLASSPowerSpectrum(parameters, **ps_var)
    if key == 'boltzmann':
        return BoltzmannSolver(parameters, **ps_var)
    if key == 'disco_eb':
        return DiscoEB(parameters, **ps_var)
    return TabulatedPowerSpectrum(parameters, **ps_var)
