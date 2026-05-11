"""Base class for halo mass function models.

Provides sigma^2(M, z) precomputed from an arbitrary linear power spectrum
and the shared physical constants used by all HMF implementations.
Supports pluggable window functions (top-hat, sharp-k, smooth-k).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

try:
    from numpy import trapezoid as _trapz
except ImportError:
    from numpy import trapz as _trapz

from ..cosmo import D
from ..constants import rhoc0
from ..structs import Parameters
from ..lpt.linear_power import get_power_spectrum, PowerSpectrum
from .window import Window, get_window


class MassFunction:
    """Abstract halo mass function base.

    Precomputes sigma^2(M) at z=0 from the linear power spectrum filtered
    by a chosen window function, then scales it with the growth factor for
    arbitrary redshift.  Concrete subclasses implement :meth:`dndlnm`.

    Args:
        parameters:  BEoRN :class:`~beorn.structs.Parameters` object.
        window:      Filter function — ``'tophat'`` (default), ``'sharp_k'``,
                     ``'smooth_k'``, or a :class:`~beorn.mass_function.window.Window`
                     instance.
        ps_method:   Power spectrum method (default ``'eisenstein_hu'``).
        **ps_kwargs: Forwarded to the power spectrum constructor.
    """

    def __init__(
        self,
        parameters: Parameters,
        window: str | Window = 'tophat',
        ps_method: str = 'eisenstein_hu',
        **ps_kwargs,
    ):
        self.parameters = parameters
        self.window_fn: Window = get_window(window)
        self.power_spectrum: PowerSpectrum = get_power_spectrum(
            ps_method, parameters, **ps_kwargs
        )
        self._precompute_sigma()

    # ------------------------------------------------------------------
    # Physical quantities
    # ------------------------------------------------------------------

    @property
    def rho_m(self) -> float:
        """Mean comoving matter density in M_sun (Mpc/h)^{-3}.

        rhoc0 = 2.775e11 h^2 Msun/Mpc^3.  1 Mpc = h0 Mpc/h
        => rho_m = Om * rhoc0 / h0  [Msun/(Mpc/h)^3].
        """
        return (
            self.parameters.cosmology.Om
            * rhoc0
            / self.parameters.cosmology.h0
        )

    def R_of_M(self, M: float | np.ndarray) -> float | np.ndarray:
        """Top-hat Lagrangian radius for mass M [Mpc/h] — same for all windows."""
        return (3.0 * M / (4.0 * np.pi * self.rho_m)) ** (1.0 / 3.0)

    def M_of_R(self, R: float | np.ndarray) -> float | np.ndarray:
        """Mass enclosed in a top-hat sphere of radius R [Mpc/h]."""
        return (4.0 / 3.0) * np.pi * R ** 3 * self.rho_m

    # ------------------------------------------------------------------
    # sigma^2(M, z)
    # ------------------------------------------------------------------

    def _precompute_sigma(self) -> None:
        """Vectorised sigma^2(M) integral at z=0, 1000 k-nodes, 200 M-grid."""
        N_k = 1000
        lnk = np.linspace(np.log(1e-4), np.log(1e3), N_k)
        k = np.exp(lnk)

        M_grid = np.logspace(6, 17, 200)
        R_grid = self.R_of_M(M_grid)

        kR = np.outer(k, R_grid)          # (N_k, 200)
        W = self.window_fn.W(kR)           # uses chosen window

        Pk = self.power_spectrum.P(k, z=0.0)
        integrand = k[:, None] ** 3 * Pk[:, None] * W ** 2 / (2.0 * np.pi ** 2)
        sigma2_grid = _trapz(integrand, lnk, axis=0)

        self._sigma2_interp = interp1d(
            np.log(M_grid),
            np.log(sigma2_grid),
            kind='cubic',
            fill_value='extrapolate',
        )

    def _D1(self, z: float) -> float:
        a = 1.0 / (1.0 + z)
        return D(a, self.parameters) / D(1.0, self.parameters)

    def sigma2(self, M: float | np.ndarray, z: float) -> float | np.ndarray:
        """sigma^2(M, z) = D1(z)^2 * sigma^2(M, z=0)."""
        M = np.asarray(M, dtype=float)
        s2_z0 = np.exp(self._sigma2_interp(np.log(M)))
        return self._D1(z) ** 2 * s2_z0

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def dndlnm(self, M: np.ndarray, z: float) -> np.ndarray:
        """dn/d ln M in (Mpc/h)^{-3}.

        Args:
            M: Halo masses in M_sun.
            z: Redshift.

        Returns:
            Array of same shape as M.
        """
        raise NotImplementedError
