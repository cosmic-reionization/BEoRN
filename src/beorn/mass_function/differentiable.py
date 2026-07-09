"""Backend-generic, differentiable sigma^2(M) and halo mass function.

Pure-function counterparts of :class:`.base.MassFunction` /
:class:`.models.HaloMassFunction` for the opt-in gpu/diff path (issue #42,
Phase 1: G5; closes the sigma^2 gap of issue #39). The numpy classes are
unchanged and remain the default.

What differs from the class implementation:

- sigma^2(M) is a direct fixed-node trapezoid in the chosen backend — no
  scipy ``interp1d`` table, so gradients flow through the cosmology
  (Om, Ob, h0, ns, sigma_8) via the E&H no-wiggle P(k) and the growth factor.
- dln(sigma)/dlnM is computed **analytically** from the derivative of the
  top-hat window (dW/dx), replacing the 1% finite difference of the classes.
- Everything runs on the device of the input tensors (CUDA / MPS). The
  arrays are tiny, so GPU brings no throughput win here — the point is
  device-residency, so an HMF evaluated inside a GPU pipeline (e.g. the
  CHMF over N^3 cells) never forces a host round-trip.

Conventions match :class:`.base.MassFunction`: M in Msun, rho_m in
Msun (Mpc/h)^-3, k in h/Mpc, dn/dlnM in (Mpc/h)^-3; top-hat window.

Note: with ``A=None`` the self-consistent normalisation A(p) uses the Gamma
function on a *float* p, so gradients w.r.t. p ignore the A(p) dependence —
pass an explicit ``A`` (or treat A as a free parameter) when differentiating
w.r.t. p.
"""
from __future__ import annotations

import math
import numpy as np

from ..constants import rhoc0
from ..cosmo.differentiable import (
    get_backend, device_of, as_array, as_const, trapz_static, growth_factor,
)
from ..lpt.linear_power import transfer_eh_nowiggle, sigma8_normalisation
from .models import _f_nu, _normalise_A


def _tophat_W(x, xp):
    return 3.0 * (xp.sin(x) - x * xp.cos(x)) / x ** 3


def _tophat_dW(x, xp):
    """dW/dx for the top-hat window."""
    return 3.0 * xp.sin(x) / x ** 2 - 9.0 * (xp.sin(x) - x * xp.cos(x)) / x ** 4


def sigma2_M(M, z, Om, Ob, h0, ns, sigma_8, backend='numpy',
             n_k=1000, n_nodes=512, return_dln_dlnM=False):
    """sigma^2(M, z) for a top-hat window, E&H no-wiggle P(k).

    Differentiable w.r.t. every cosmological argument in jax/torch;
    device-resident. With ``return_dln_dlnM=True`` also returns the analytic
    dln(sigma)/dlnM (z-independent).
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, M, Om, Ob, h0, ns, sigma_8)
    M = as_array(M, name, xp, device)
    Om = as_array(Om, name, xp, device)
    Ob = as_array(Ob, name, xp, device)
    h0 = as_array(h0, name, xp, device)
    ns = as_array(ns, name, xp, device)
    sigma_8 = as_array(sigma_8, name, xp, device)

    rho_m = Om * rhoc0 / h0                       # Msun (Mpc/h)^-3
    M1 = M.reshape(-1)
    R = (3.0 * M1 / (4.0 * math.pi * rho_m)) ** (1.0 / 3.0)   # Mpc/h

    lnk_np = np.linspace(np.log(1e-4), np.log(1e3), n_k)
    k = as_const(np.exp(lnk_np), name, xp, device)

    A_s = sigma8_normalisation(Om, Ob, h0, ns, sigma_8, backend=backend)
    T = transfer_eh_nowiggle(k, Om, Ob, h0, backend=backend)
    Pk0 = A_s * k ** ns * T ** 2                  # P(k, z=0)

    kR = k.reshape(-1, 1) * R.reshape(1, -1)      # (n_k, n_M)
    W = _tophat_W(kR, xp)
    pref = (k ** 3 * Pk0 / (2.0 * math.pi ** 2)).reshape(-1, 1)

    s2_0 = trapz_static(pref * W ** 2, lnk_np, name, xp, axis=0)

    a = 1.0 / (1.0 + z)
    D1 = growth_factor(a, Om, backend=backend, n_nodes=n_nodes)
    s2_z = (D1 ** 2 * s2_0).reshape(M.shape)

    if not return_dln_dlnM:
        return s2_z

    # d sigma^2 / d lnM = int dlnk pref * 2 W dW/dx * (kR/3)   (dlnR/dlnM = 1/3)
    dW = _tophat_dW(kR, xp)
    ds2_dlnM = trapz_static(pref * 2.0 * W * dW * kR / 3.0, lnk_np, name, xp,
                            axis=0)
    dln_sigma_dlnM = (0.5 * ds2_dlnM / s2_0).reshape(M.shape)
    return s2_z, dln_sigma_dlnM


def dndlnm(M, z, Om, Ob, h0, ns, sigma_8,
           p=0.3, q=0.707, A=None, delta_c=1.686,
           backend='numpy', n_k=1000, n_nodes=512):
    """dn/dlnM [(Mpc/h)^-3] — fully differentiable parametric HMF.

    f(nu) = A sqrt(2 q nu / pi) (1 + (q nu)^{-p}) exp(-q nu / 2), with
    nu = delta_c^2 / sigma^2(M, z). Defaults are Sheth-Tormen; (p=0, q=1,
    A=0.5) gives Press-Schechter.

    Differentiable w.r.t. (Om, Ob, h0, ns, sigma_8, delta_c, p, q, A) and z,
    in numpy / jax / torch; GPU-capable via tensor inputs.

    Example (jax)::

        import jax
        dn_ds8 = jax.grad(
            lambda s8: dndlnm(M, 7.0, 0.315, 0.049, 0.673, 0.963, s8,
                              backend='jax').sum()
        )(0.811)
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, M, Om, Ob, h0, ns, sigma_8, delta_c, p, q, A)
    M_b = as_array(M, name, xp, device)
    Om_b = as_array(Om, name, xp, device)
    h0_b = as_array(h0, name, xp, device)

    s2, dln = sigma2_M(M, z, Om, Ob, h0, ns, sigma_8, backend=backend,
                       n_k=n_k, n_nodes=n_nodes, return_dln_dlnM=True)
    sigma = xp.sqrt(s2)
    dc = as_array(delta_c, name, xp, device)
    nu = (dc / sigma) ** 2

    if A is None:
        A = _normalise_A(float(p))   # float p only — see module docstring
    f = _f_nu(nu, p, q, A, backend=name)

    rho_m = Om_b * rhoc0 / h0_b
    return (rho_m / M_b) * xp.abs(dln) * f
