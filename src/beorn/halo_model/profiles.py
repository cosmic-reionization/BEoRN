"""Fourier transforms of spherically symmetric halo profiles.

All transforms follow

    W(k) = 4 pi \\int r^2 rho(r) j0(k r) dr,      j0(x) = sin(x) / x,

with comoving r [Mpc/h] and k [h/Mpc], so W(k -> 0) is the volume
integral of the profile.  Tabulated profiles are transformed with a
linear map ``W = K @ rho`` whose kernel ``K`` depends only on the (static)
k and r grids, so the transform is differentiable w.r.t. the profile
values in numpy / jax / torch.
"""
from __future__ import annotations

import numpy as np
from scipy.special import sici

from ..cosmo.differentiable import get_backend, as_array, as_const


# ──────────────────────────────────────────────────────────────────────
# Analytic windows
# ──────────────────────────────────────────────────────────────────────

def tophat_window(x, backend='numpy'):
    """Normalised top-hat window 3 (sin x - x cos x) / x^3 (-> 1 for x -> 0)."""
    name, xp = get_backend(backend)
    x = as_array(x, name, xp)
    small = x < 1e-3
    xs = xp.where(small, xp.ones_like(x), x)
    full = 3.0 * (xp.sin(xs) - xs * xp.cos(xs)) / xs ** 3
    return xp.where(small, 1.0 - x ** 2 / 10.0, full)


def tophat_ft(k, R, backend='numpy'):
    """FT of a unit-height sphere of radius R: (4 pi R^3 / 3) W_TH(kR).

    Args:
        k: (Nk,) wavenumbers [h/Mpc].
        R: (NM,) radii [Mpc/h].

    Returns:
        (Nk, NM) array in (Mpc/h)^3.
    """
    name, xp = get_backend(backend)
    k = as_array(k, name, xp)
    R = as_array(R, name, xp)
    return 4.0 * np.pi / 3.0 * R[None, :] ** 3 * tophat_window(k[:, None] * R[None, :], backend)


def nfw_ft(k, r_s, c):
    """Normalised NFW Fourier transform u(k|M) (Cooray & Sheth 2002, Eq. 81).

    Args:
        k:   (Nk,) wavenumbers [h/Mpc].
        r_s: (NM,) scale radii [Mpc/h].
        c:   (NM,) concentrations.

    Returns:
        (Nk, NM) array, u -> 1 for k -> 0.
    """
    k = np.asarray(k, dtype=float)[:, None]
    r_s = np.asarray(r_s, dtype=float)[None, :]
    c = np.asarray(c, dtype=float)[None, :]
    x = k * r_s
    si1, ci1 = sici((1.0 + c) * x)
    si0, ci0 = sici(x)
    m_c = np.log(1.0 + c) - c / (1.0 + c)
    u = (np.sin(x) * (si1 - si0)
         - np.sin(c * x) / ((1.0 + c) * x)
         + np.cos(x) * (ci1 - ci0)) / m_c
    return u


# ──────────────────────────────────────────────────────────────────────
# Tabulated profiles
# ──────────────────────────────────────────────────────────────────────

def _trapz_weights(r: np.ndarray) -> np.ndarray:
    dr = np.diff(r)
    w = np.zeros_like(r)
    w[:-1] += 0.5 * dr
    w[1:] += 0.5 * dr
    return w


def profile_ft_kernel(k, r, include_core: bool = True, k_switch: float = 0.3) -> np.ndarray:
    """Kernel K (Nk, Nr) such that ``W(k) = K @ rho(r)``.

    For well-sampled modes (``k * max(dr) < k_switch``) a trapezoid rule on
    ``4 pi r^2 j0(kr) rho`` is used.  Otherwise a Filon-type rule treats
    ``g(r) = r rho(r)`` as piecewise linear between nodes and integrates
    ``(4 pi / k) g(r) sin(kr)`` exactly on each segment, so modes that
    oscillate many times per radial cell (large k at large r on a log grid)
    are still accurate.

    Args:
        k: (Nk,) wavenumbers [h/Mpc].
        r: (Nr,) radial nodes [Mpc/h], strictly increasing.
        include_core: add the sphere ``r < r[0]`` with ``rho = rho(r[0])``.
        k_switch: switch-over value of ``k * max(dr)``.

    Returns:
        (Nk, Nr) numpy array.
    """
    k = np.asarray(k, dtype=float)
    r = np.asarray(r, dtype=float)
    h = np.diff(r)

    # trapezoid on 4 pi r^2 j0(kr)
    kr = np.outer(k, r)
    j0 = np.sinc(kr / np.pi)
    K_trap = 4.0 * np.pi * r[None, :] ** 2 * j0 * _trapz_weights(r)[None, :]

    # Filon: \int g sin(kr) dr, g piecewise linear; weights act on g = r rho
    kk = np.where(k > 0, k, 1.0)[:, None]
    S = np.sin(kk * r[None, :])
    C = np.cos(kk * r[None, :])
    D = (S[:, 1:] - S[:, :-1]) / (h[None, :] * kk ** 2)       # (Nk, Nr-1)
    wg = np.zeros((k.size, r.size))
    wg[:, 1:] += D
    wg[:, :-1] -= D
    wg[:, 0] += C[:, 0] / kk[:, 0]
    wg[:, -1] -= C[:, -1] / kk[:, 0]
    K_filon = 4.0 * np.pi / kk * wg * r[None, :]                # act on rho

    use_filon = (k * h.max() >= k_switch)[:, None]
    K = np.where(use_filon, K_filon, K_trap)

    if include_core:
        r0 = r[0]
        K[:, 0] += 4.0 * np.pi / 3.0 * r0 ** 3 * tophat_window(k * r0)
    return K


def profile_ft(k, r, rho, backend='numpy', **kernel_kwargs):
    """Fourier transform of tabulated radial profiles.

    Args:
        k:   (Nk,) wavenumbers [h/Mpc].
        r:   (Nr,) comoving radial nodes [Mpc/h].
        rho: (Nr, ...) profile values; leading axis is radial (the layout
             of :class:`~beorn.structs.RadiationProfiles`).
        backend: 'numpy', 'jax' or 'torch'.

    Returns:
        (Nk, ...) array.
    """
    name, xp = get_backend(backend)
    K = profile_ft_kernel(k, r, **kernel_kwargs)
    rho = as_array(rho, name, xp)
    rest = tuple(rho.shape[1:])
    out = as_const(K, name, xp) @ rho.reshape(rho.shape[0], -1)
    return out.reshape((K.shape[0],) + rest)
