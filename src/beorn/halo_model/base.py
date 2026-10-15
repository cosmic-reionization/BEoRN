"""Generic halo model for any set of tracers (issue #38).

For tracers X, Y with halo windows W_X(k|M), W_Y(k|M) (see :mod:`.tracers`)

    P_XY^1h(k) = \\int dlnM n(M) W_X(k|M) W_Y(k|M)
    P_XY^2h(k) = I_X(k) I_Y(k) P_lin(k),   I_X(k) = \\int dlnM n(M) b_X(M) W_X(k|M)

with n = dn/dlnM.  Mass integrals use the midpoint rule on log-mass bins
(``weights = n(M_i) dlnM_i``), matching how the painting pipeline sums
haloes in mass bins.  The integrators work in numpy / jax / torch.

Units: halo masses in M_sun/h (as BEoRN's radiation profiles and halo
catalogues), lengths in Mpc/h, k in h/Mpc.  The mass function
(:class:`~beorn.mass_function.HaloMassFunction`, which works in M_sun) is
evaluated at M / h internally.

Quadrature grid of a pair (X, Y): a tracer's own mass bins if it has them
(binned profiles), otherwise the model grid ``[M_min, M_max]``; if X and Y
have different own grids, the model grid is used.

Example::

    from beorn.halo_model import HaloModel, TopHatTracer, FunctionTracer

    hm = HaloModel(param)                             # matter tracer 'm' included
    hm.add_tracer(TopHatTracer('ion', R_bubble, M_edges))
    hm.add_tracer(FunctionTracer('g', my_window))
    out = hm.power_spectra(k, z=8.0)                  # all auto + cross spectra
    out['P_m_ion'], out['P_ion_ion_1h'], out['mean_ion']
    hm.power_spectrum(k, 8.0, 'ion')                  # one pair: P, P_1h, P_2h, P_lin
"""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np

from ..constants import rhoc0
from ..cosmo.differentiable import as_array, get_backend
from ..cosmo.ps_linear import PowerSpectrum, get_power_spectrum
from ..mass_function import HaloMassFunction
from ..structs import Parameters
from .bias import halo_bias
from .tracers import MatterTracer, Tracer, bins_from_edges, log_mass_bins

__all__ = ['HaloModel', 'log_mass_bins', 'bins_from_edges']


class HaloModel:
    """Halo model with linear P(k), n(M), b(M) and any number of tracers.

    Args:
        parameters: BEoRN :class:`~beorn.structs.Parameters`.
        tracers:    Tracers to register (list of :class:`~.tracers.Tracer`).
        include_matter: Register a :class:`~.tracers.MatterTracer` named ``'m'``
                    (NFW haloes) — default True.
        ps_method:  Linear P(k): ``None`` (Eisenstein & Hu, default),
                    ``'camb'`` or ``'class'``; see
                    :func:`beorn.cosmo.ps_linear.get_power_spectrum`.
        hmf_model:  Mass function model passed to
                    :class:`~beorn.mass_function.HaloMassFunction`.
        bias_model: Halo bias model (:func:`.bias.halo_bias`). Defaults to
                    the peak-background-split partner of ``hmf_model``.
        window:     Window function for sigma(M).
        delta_c:    Linear collapse threshold.
        M_min, M_max, n_mass: Default log-mass grid [M_sun/h].
        backend:    Array backend of the integrators.
        **ps_var:   Forwarded to the linear power spectrum / Boltzmann solver.
    """

    def __init__(
        self,
        parameters: Parameters,
        tracers: list[Tracer] | None = None,
        include_matter: bool = True,
        ps_method: str | None = None,
        hmf_model: str = 'sheth_tormen',
        bias_model: str | None = None,
        window: str = 'tophat',
        delta_c: float = 1.686,
        M_min: float = 1e4,
        M_max: float = 1e17,
        n_mass: int = 200,
        backend: str = 'numpy',
        **ps_var,
    ):
        self.parameters = parameters
        self.backend = backend
        self.delta_c = delta_c
        self.linear_power: PowerSpectrum = get_power_spectrum(ps_method, parameters, **ps_var)
        self.hmf = HaloMassFunction(parameters, model=hmf_model, window=window,
                                    delta_c=delta_c, power_spectrum=self.linear_power)
        if bias_model is None:
            bias_model = 'press_schechter' if hmf_model in ('press_schechter', 'ps') else 'sheth_tormen'
        self.bias_model = bias_model
        self.M_grid, self.dlnM_grid = log_mass_bins(M_min, M_max, n_mass)

        self.tracers: dict[str, Tracer] = {}
        if include_matter:
            self.add_tracer(MatterTracer('m'))
        for tracer in tracers or []:
            self.add_tracer(tracer)

    # ------------------------------------------------------------------
    # Tracers
    # ------------------------------------------------------------------

    def add_tracer(self, tracer: Tracer) -> Tracer:
        """Register a tracer (replaces one with the same name)."""
        tracer.attach(self)
        self.tracers[tracer.name] = tracer
        return tracer

    def _tracer(self, tracer) -> Tracer:
        return self.tracers[tracer] if isinstance(tracer, str) else tracer

    def _grid(self, z: float, X, Y=None):
        """``(M, dlnM, key)``; ``key`` names whose grid is used (tracer name or 'model')."""
        cands = []
        for T in (X, Y):
            if T is None:
                continue
            T = self._tracer(T)
            g = T.mass_grid(z)
            if g is not None:
                cands.append((g[0], g[1], T.name))
        if not cands:
            return self.M_grid, self.dlnM_grid, 'model'
        if len(cands) == 2 and not (cands[0][0].shape == cands[1][0].shape
                                    and np.allclose(cands[0][0], cands[1][0])):
            return self.M_grid, self.dlnM_grid, 'model'
        return cands[0]

    def grid_for(self, z: float, X, Y=None) -> tuple[np.ndarray, np.ndarray]:
        """Quadrature grid ``(M, dlnM)`` for tracer X, or for the pair (X, Y)."""
        return self._grid(z, X, Y)[:2]

    # ------------------------------------------------------------------
    # Ingredients
    # ------------------------------------------------------------------

    @property
    def rho_m(self) -> float:
        """Mean comoving matter density [(M_sun/h) (Mpc/h)^-3]."""
        return self.parameters.cosmology.Om * rhoc0

    def plin(self, k, z: float):
        """Linear matter power spectrum [(Mpc/h)^3]."""
        return self.linear_power.P(np.asarray(k, dtype=float), z)

    def dndlnm(self, M, z: float) -> np.ndarray:
        """Halo mass function dn/dlnM [(Mpc/h)^-3] for M in M_sun/h."""
        M_phys = np.asarray(M, dtype=float) / self.parameters.cosmology.h0
        return np.asarray(self.hmf.dndlnm(M_phys, z))

    def bias(self, M, z: float) -> np.ndarray:
        """Linear halo bias b(M, z) for M in M_sun/h."""
        M_phys = np.asarray(M, dtype=float) / self.parameters.cosmology.h0
        sigma = np.sqrt(self.hmf.sigma2(M_phys, z))
        return halo_bias(sigma, self.bias_model, self.delta_c)

    def mass_weights(self, M, dlnM, z: float) -> tuple[np.ndarray, np.ndarray]:
        """Quadrature weights ``n(M) dlnM`` and halo bias ``b(M)`` on a mass grid."""
        return self.dndlnm(M, z) * np.asarray(dlnM), self.bias(M, z)

    # ------------------------------------------------------------------
    # Integrators (backend-generic)
    # ------------------------------------------------------------------

    def one_halo(self, weights, W_X, W_Y=None):
        """P^1h(k) = sum_M w(M) W_X(k|M) W_Y(k|M); W arrays are (Nk, NM)."""
        name, xp = get_backend(self.backend)
        W_X = as_array(W_X, name, xp)
        W_Y = W_X if W_Y is None else as_array(W_Y, name, xp)
        return (W_X * W_Y * as_array(weights, name, xp)[None, :]).sum(-1)

    def two_halo_amplitude(self, weights, bias, W_X):
        """I_X(k) = sum_M w(M) b(M) W_X(k|M)."""
        name, xp = get_backend(self.backend)
        wb = as_array(weights, name, xp) * as_array(bias, name, xp)
        return (as_array(W_X, name, xp) * wb[None, :]).sum(-1)

    def two_halo(self, I_X, I_Y, P_lin):
        """P^2h(k) = I_X(k) I_Y(k) P_lin(k)."""
        name, xp = get_backend(self.backend)
        return as_array(I_X, name, xp) * as_array(I_Y, name, xp) * as_array(P_lin, name, xp)

    # ------------------------------------------------------------------
    # Power spectra
    # ------------------------------------------------------------------

    def power_spectra(self, k, z: float, tracers=None, pairs=None) -> dict:
        """All auto- and cross-spectra of the registered tracers at one redshift.

        Args:
            k:       (Nk,) wavenumbers [h/Mpc].
            z:       Redshift.
            tracers: Names to include (default: all registered).
            pairs:   Explicit list of ``(X, Y)`` pairs (default: all
                     combinations with replacement of ``tracers``).

        Returns:
            dict with ``k``, ``z``, ``P_lin``, ``P_<X>_<Y>``, ``P_<X>_<Y>_1h``,
            ``P_<X>_<Y>_2h`` [(field units)^2 (Mpc/h)^3] and ``mean_<X>``.
        """
        k = np.asarray(k, dtype=float)
        names = list(self.tracers) if tracers is None else list(tracers)
        if pairs is None:
            pairs = list(combinations_with_replacement(names, 2))
        names = list(dict.fromkeys(names + [n for pair in pairs for n in pair]))

        windows = {}

        def W(name, M, grid_key):
            if (name, grid_key) not in windows:
                windows[(name, grid_key)] = self.tracers[name].window(k, M, z)
            return windows[(name, grid_key)]

        P_lin = self.plin(k, z)
        out = {'k': k, 'z': z, 'P_lin': P_lin}

        # two-halo amplitudes on each tracer's own grid
        amp = {}
        for name in names:
            tr = self.tracers[name]
            M, dlnM, gk = self._grid(z, name)
            w = self.dndlnm(M, z) * dlnM
            amp[name] = tr.two_halo_amplitude(k, z, M, w, tr.bias(M, z), W(name, M, gk))
            out[f'mean_{name}'] = tr.mean(z)

        for X, Y in pairs:
            M, dlnM, gk = self._grid(z, X, Y)
            w = self.dndlnm(M, z) * dlnM
            P1 = self.one_halo(w, W(X, M, gk), W(Y, M, gk))
            P2 = self.two_halo(amp[X], amp[Y], P_lin)
            out[f'P_{X}_{Y}_1h'], out[f'P_{X}_{Y}_2h'] = P1, P2
            out[f'P_{X}_{Y}'] = P1 + P2
        return out

    def power_spectrum(self, k, z: float, X: str = 'm', Y: str | None = None) -> dict:
        """Single (cross-)spectrum of tracers X and Y (default: matter auto).

        Returns:
            dict with ``k, z, P_lin, P_1h, P_2h, P``.
        """
        Y = X if Y is None else Y
        out = self.power_spectra(k, z, pairs=[(X, Y)])
        key = f'P_{X}_{Y}'
        return {'k': out['k'], 'z': z, 'P_lin': out['P_lin'],
                'P_1h': out[key + '_1h'], 'P_2h': out[key + '_2h'], 'P': out[key]}

    # ------------------------------------------------------------------
    # Higher-order statistics
    # ------------------------------------------------------------------

    def bispectrum(self, *args, **kwargs):
        """Halo-model bispectrum (1h + 2h + 3h). Planned — issue #38, Phase B."""
        raise NotImplementedError("The halo-model bispectrum is not implemented yet (issue #38, Phase B).")
