"""Tracers: fields sourced by haloes, described by their halo window W(k|M, z).

A tracer X is a field whose value is the sum of identical profiles
rho_X(r|M, z) centred on every halo.  The halo model only needs the
Fourier transform of that profile,

    W_X(k|M, z) = 4 pi \\int r^2 rho_X(r|M, z) j0(kr) dr     [field units x (Mpc/h)^3],

with halo masses M in M_sun/h, so ``P_XY`` comes out in (field units)^2 (Mpc/h)^3 and the mean field is
``<X> = \\int dlnM n(M) W_X(0|M)``.  Subclass :class:`Tracer` (implement
:meth:`Tracer.window`) to add a new field, or use one of the ready-made
tracers:

* :class:`MatterTracer`   — matter overdensity, NFW haloes (W -> M / rho_m)
* :class:`ProfileTracer`  — tabulated radial profiles per mass bin (e.g. BEoRN radiation profiles)
* :class:`TopHatTracer`   — uniform spheres of radius R(M) (e.g. ionised bubbles)
* :class:`FunctionTracer` — any user function ``W(k, M, z)``

Binned tracers (profile / top-hat) follow the painting pipeline: every
halo in a mass bin carries that bin's profile, and their own mass bins
are used as the quadrature grid.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .profiles import nfw_ft, profile_ft, tophat_ft


def log_mass_bins(M_min: float, M_max: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Log-spaced bin centres and widths ``(M_c, dlnM)`` between two edges."""
    edges = np.geomspace(M_min, M_max, n + 1)
    return np.sqrt(edges[1:] * edges[:-1]), np.diff(np.log(edges))


def bins_from_edges(M_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Geometric bin centres and log widths ``(M_c, dlnM)`` from mass bin edges."""
    M_edges = np.asarray(M_edges, dtype=float)
    return np.sqrt(M_edges[1:] * M_edges[:-1]), np.diff(np.log(M_edges))


class Tracer:
    """Base class for a halo-model tracer.

    Subclasses implement :meth:`window`; they may override :meth:`mass_grid`
    (own quadrature nodes), :meth:`bias` (tracer-specific bias, e.g. bubble
    bias), :meth:`two_halo_amplitude` (e.g. missing-mass correction) and
    :meth:`mean`.  ``self.model`` is set when the tracer is added to a
    :class:`~beorn.halo_model.HaloModel`.

    Args:
        name: Key used in the output dictionaries (``P_<X>_<Y>``).
    """

    def __init__(self, name: str):
        self.name = name
        self.model = None

    def attach(self, model) -> None:
        """Called by :meth:`HaloModel.add_tracer`."""
        self.model = model

    def clear_cache(self) -> None:
        """Drop cached windows (call after the underlying profiles change)."""

    def window(self, k, M, z: float) -> np.ndarray:
        """W(k|M, z), shape (Nk, NM). Must accept any mass (zero outside the tracer's range)."""
        raise NotImplementedError

    def mass_grid(self, z: float):
        """Own quadrature nodes ``(M, dlnM)``, or ``None`` to use the model's grid."""
        return None

    def bias(self, M, z: float) -> np.ndarray:
        """Large-scale bias of the sources; defaults to the model's halo bias."""
        return self.model.bias(M, z)

    def two_halo_amplitude(self, k, z: float, M, weights, bias, W) -> np.ndarray:
        """I_X(k) = sum_M w b W.  Override to add corrections."""
        return self.model.two_halo_amplitude(weights, bias, W)

    def mean(self, z: float) -> float:
        """Mean field <X> = sum_M n(M) dlnM W(k -> 0|M)."""
        M, dlnM = self.model.grid_for(z, self)
        w = self.model.dndlnm(M, z) * dlnM
        return float(np.sum(w * self.window(np.array([1e-8]), M, z)[0]))


# ──────────────────────────────────────────────────────────────────────
# Matter
# ──────────────────────────────────────────────────────────────────────

def concentration_duffy08(M, z, h0):
    """Duffy et al. (2008) c(M, z), full sample, Delta = 200 x mean density.

    M in M_sun/h (the fit's pivot mass is 2e12 M_sun/h); ``h0`` is unused and
    kept so all concentration relations share the signature ``c(M, z, h0)``.
    """
    return 10.14 * (np.asarray(M) / 2e12) ** -0.081 * (1.0 + z) ** -1.01


class MatterTracer(Tracer):
    """Matter overdensity with NFW haloes: W_m = (M / rho_m) u_NFW(k|M).

    The two-halo amplitude includes the mass outside the model's mass grid
    (the "missing-mass" correction: ``1 - \\int n b M / rho_m`` placed in the
    lowest bin), so ``P^2h -> P_lin`` for k -> 0, and the mean is 1.

    Args:
        name:          Tracer key (default ``'m'``).
        Delta:         Halo overdensity w.r.t. the mean matter density.
        concentration: Callable ``c(M, z, h0)`` (default Duffy et al. 2008).
    """

    def __init__(self, name: str = 'm', Delta: float = 200.0,
                 concentration: Callable = concentration_duffy08):
        super().__init__(name)
        self.Delta = Delta
        self.concentration = concentration

    def r_halo(self, M) -> np.ndarray:
        """Halo radius [Mpc/h] for overdensity Delta w.r.t. the mean matter density."""
        return (3.0 * np.asarray(M) / (4.0 * np.pi * self.Delta * self.model.rho_m)) ** (1.0 / 3.0)

    def u_nfw(self, k, M, z: float) -> np.ndarray:
        """Normalised NFW profile FT u(k|M), shape (Nk, NM)."""
        c = self.concentration(M, z, self.model.parameters.cosmology.h0)
        return nfw_ft(k, self.r_halo(M) / c, c)

    def window(self, k, M, z: float) -> np.ndarray:
        return np.asarray(M)[None, :] / self.model.rho_m * self.u_nfw(k, M, z)

    def two_halo_amplitude(self, k, z, M, weights, bias, W):
        I_m = self.model.two_halo_amplitude(weights, bias, W)
        missing = 1.0 - float(np.sum(np.asarray(weights) * np.asarray(bias) * np.asarray(M) / self.model.rho_m))
        return I_m + missing * self.u_nfw(k, np.asarray(M)[:1], z)[:, 0]

    def mean(self, z: float) -> float:
        return 1.0


# ──────────────────────────────────────────────────────────────────────
# Binned (tabulated per mass bin) tracers
# ──────────────────────────────────────────────────────────────────────

class BinnedTracer(Tracer):
    """Tracer tabulated on mass bins; every halo in a bin carries the bin's profile.

    Subclasses implement :meth:`binned_window` (W on the bin centres) and
    :meth:`mass_edges`.  :meth:`window` maps any mass to its bin (zero
    outside the edges) and :meth:`mass_grid` returns the bin centres.
    Binned windows are cached per (k, z), so evaluating them on several mass
    grids costs one Fourier transform.
    """

    _max_cache = 8

    def clear_cache(self) -> None:
        self._cache = {}

    def mass_edges(self, z: float) -> np.ndarray:
        raise NotImplementedError

    def binned_window(self, k, z: float) -> np.ndarray:
        """W on the bins, shape (Nk, Nbins)."""
        raise NotImplementedError

    def mass_grid(self, z: float):
        return bins_from_edges(self.mass_edges(z))

    def window(self, k, M, z: float) -> np.ndarray:
        edges = self.mass_edges(z)
        M = np.asarray(M, dtype=float)
        k = np.asarray(k, dtype=float)
        key = (k.tobytes(), float(z))
        cache = self.__dict__.setdefault('_cache', {})
        if key not in cache:
            if len(cache) >= self._max_cache:
                cache.pop(next(iter(cache)))
            cache[key] = np.asarray(self.binned_window(k, z))
        W_bins = cache[key]
        centres, _ = bins_from_edges(edges)
        if M.shape == centres.shape and np.allclose(M, centres, rtol=1e-12):
            return W_bins
        idx = np.searchsorted(edges, M, side='right') - 1
        inside = (idx >= 0) & (idx < W_bins.shape[1])
        return W_bins[:, np.clip(idx, 0, W_bins.shape[1] - 1)] * inside[None, :]


def _as_z_function(value):
    return value if callable(value) else (lambda z: value)


class ProfileTracer(BinnedTracer):
    """Tabulated radial profiles, Fourier transformed with :func:`.profiles.profile_ft`.

    Args:
        name:    Tracer key.
        r:       Comoving radii [Mpc/h], shape (Nr,), or a function ``r(z)``.
        rho:     Profiles, shape (Nr, Nbins), or a function ``rho(z)``.
        M_edges: Mass bin edges [M_sun/h], shape (Nbins+1,), or a function ``M_edges(z)``.

    Example::

        tr = ProfileTracer('heat', r_grid, lambda z: rho_heat_at(z), lambda z: edges_at(z))
    """

    def __init__(self, name: str, r, rho, M_edges):
        super().__init__(name)
        self._r, self._rho, self._edges = map(_as_z_function, (r, rho, M_edges))

    def mass_edges(self, z):
        return np.asarray(self._edges(z), dtype=float)

    def binned_window(self, k, z):
        return profile_ft(k, np.asarray(self._r(z), dtype=float), np.asarray(self._rho(z), dtype=float))


class TopHatTracer(BinnedTracer):
    """Uniform unit-height spheres of radius R(M): W = 4/3 pi R^3 W_TH(kR).

    Args:
        name:    Tracer key.
        R:       Comoving radii [Mpc/h] per bin, shape (Nbins,), or a function ``R(z)``.
        M_edges: Mass bin edges [M_sun/h], or a function ``M_edges(z)``.
    """

    def __init__(self, name: str, R, M_edges):
        super().__init__(name)
        self._R, self._edges = map(_as_z_function, (R, M_edges))

    def mass_edges(self, z):
        return np.asarray(self._edges(z), dtype=float)

    def binned_window(self, k, z):
        return tophat_ft(k, np.asarray(self._R(z), dtype=float))


class FunctionTracer(Tracer):
    """Tracer defined by a user function ``window(k, M, z) -> (Nk, NM)``.

    Args:
        name:     Tracer key.
        window:   Callable returning W(k|M, z).
        M_edges:  Optional mass bin edges (array or function of z) to use as
                  this tracer's own quadrature grid.
        bias:     Optional callable ``b(M, z)`` replacing the halo bias.

    Example (HOD-like galaxies with N(M) = (M / 1e12)^0.8 above 1e11 M_sun/h)::

        gal = FunctionTracer('g', lambda k, M, z: np.where(M > 1e11, (M / 1e12) ** 0.8, 0.0)[None, :]
                                                 * np.ones((len(k), 1)))
    """

    def __init__(self, name: str, window: Callable, M_edges=None, bias: Callable | None = None):
        super().__init__(name)
        self._window = window
        self._edges = None if M_edges is None else _as_z_function(M_edges)
        self._bias = bias

    def window(self, k, M, z):
        return np.asarray(self._window(np.asarray(k, dtype=float), np.asarray(M, dtype=float), z))

    def mass_grid(self, z):
        return None if self._edges is None else bins_from_edges(np.asarray(self._edges(z), dtype=float))

    def bias(self, M, z):
        return super().bias(M, z) if self._bias is None else np.asarray(self._bias(M, z))
