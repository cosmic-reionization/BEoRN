"""
Here we compute the Lyman_alpha and collisional coupling coefficient (x_al and x_coll).
"""

import numpy as np
import importlib.util
from pathlib import Path

from scipy.interpolate import splrep,splev
from .constants import Tstar, A10, rhoc0, M_sun, cm_per_Mpc, m_H, nu_al, nu_LL, m_p_in_Msun
from .cosmo import T_cmb
from .structs import Parameters


def kappa_coll():
    """
    Eq.10 in arXiv:1109.6012
    Used only in x_coll.
    Reads in tables for the scattering rates between H-H and H-e.

    Parameters
    ----------
    None

    Returns
    ----------
    Rate coefficient for spin de-excitation in collisions with H and e [cm^3/s]
    """

    names = 'T, kappa'
    path_to_file = Path(importlib.util.find_spec('beorn').origin).parent / 'input_data' / 'kappa_eH.dat'
    eH = np.genfromtxt(path_to_file, usecols=(0, 1), comments='#', dtype=float, names=names)

    names = 'T, kappa'
    path_to_file = Path(importlib.util.find_spec('beorn').origin).parent / 'input_data' / 'kappa_HH.dat'
    HH = np.genfromtxt(path_to_file, usecols=(0, 1), comments='#', dtype=float, names=names)

    return HH, eH


# Splines computed once at import time — kappa_coll() reads two .dat files and
# fits splines; doing this per x_coll() call is pure waste.
_kappa_HH_raw, _kappa_eH_raw = kappa_coll()
_kappa_eH_tck = splrep(_kappa_eH_raw['T'], _kappa_eH_raw['kappa'])
_kappa_HH_tck = splrep(_kappa_HH_raw['T'], _kappa_HH_raw['kappa'])


def x_coll(z, Tk, xHI, rho_b):
    """
    Parameters
    ----------
    z     : redshift
    Tk    : Gas kinetic temperature profile [K]
    xHI   : Hydrogen neutral fraction
    rho_b : baryon density in nbr of [H atoms /pcm**3] (physical cm)

    Returns
    ----------
    Collisional coupling coefficient. [Dimensionless]
    """
    # nH and e- densities
    n_HI  = rho_b * xHI
    n_HII = rho_b * (1-xHI) # [1/cm^3]

    # prefac (Eq.10 in arXiv:1109.6012)
    Tcmb = T_cmb(z)
    prefac = Tstar / A10 / Tcmb  # [s]

    kappa_eH = splev(Tk, _kappa_eH_tck, ext=3)  # [cm^3/s]
    kappa_HH = splev(Tk, _kappa_HH_tck, ext=3)

    x_HH = prefac * kappa_HH * n_HI
    x_eH = prefac * kappa_eH * n_HII
    return x_HH  + x_eH

def x_coll_coef(z,param):
    """
    Coefficient to turn rho/rho_mean into a baryon density in nbr of H atoms per physical cm**3 [1/pcm**3]
    """
    _Om, Ob, h0 = param.cosmo.Om, param.cosmo.Ob, param.cosmo.h0
    coef = rhoc0 * h0 ** 2 * Ob * (1 + z) ** 3 * M_sun / cm_per_Mpc ** 3 / m_H
    return coef


def S_alpha(z, Tk, xHI):
    """
    Parameters
    ----------
    z     : redshift
    Tk    : Gas kinetic temperature [K]
    xHI   : Hydrogen neutral fraction

    Returns
    ----------
    Suppression factor S_alpha. [dDmensionless]
    Following method in astro-ph/0608032
    """
    # Eq.43
    tau_GP = 3.0e5 * xHI * ((1 + z) / 7.0) ** 1.5
    # gamma = 1 / tau_GP

    # Eq. 55
    S_al = np.exp(-0.803 * Tk ** (-2 / 3) * (1e-6 * tau_GP) ** (1 / 3))

    return S_al


def eps_lyal(nu, parameters: Parameters):
    """
    Lyman-a spectral energy distribution (power-law). See eq.8 in BEoRN paper.

    Parameters
    ----------
    nu : Frequency [Hz].
    param : BEoRN dictionnary containing model parameters.

    Returns
    -------
    float. [photons.yr-1.Hz-1.SFR-1], SFR being the Star Formation Rate in Msol/h/yr
    """

    h0    = parameters.cosmology.h0
    N_al  = parameters.source.n_lyman_alpha_photons  #9690 number of lya photons per protons (baryons) in stars
    alS = parameters.source.lyman_alpha_power_law

    nu_min_norm  = nu_al
    nu_max_norm  = nu_LL

    Anorm = (1-alS)/(nu_max_norm**(1-alS) - nu_min_norm**(1-alS))
    def Inu(nu):
        return Anorm * nu**(-alS)

    eps_alpha = Inu(nu)*N_al/(m_p_in_Msun * h0)

    return eps_alpha


# ══════════════════════════════════════════════════════════════════════════════
# Differentiable counterparts (issue #42, Phase 2: G9) — numpy / jax / torch
# ══════════════════════════════════════════════════════════════════════════════
#
# Pure functions of explicit arguments; they complement (never replace) the
# numpy functions above.  The kappa(T) spin-de-excitation tables stay fixed
# numpy constants; the differentiable path interpolates them *linearly* in the
# backend (splev above is a cubic spline — the two agree to <~1e-3 relative,
# limited by table density).

def _interp_clamped(x, xp_nodes, fp_nodes, name, xp):
    """Linear interpolation with boundary clamping (splev ext=3 behaviour).

    ``xp_nodes``/``fp_nodes`` are static numpy tables; ``x`` may carry grads.
    """
    if name == 'torch':
        device = x.device if xp.is_tensor(x) else None
        xn = xp.as_tensor(np.asarray(xp_nodes, dtype=float), device=device)
        fn = xp.as_tensor(np.asarray(fp_nodes, dtype=float), device=device)
        if xn.dtype != x.dtype:
            xn = xn.to(x.dtype)
            fn = fn.to(x.dtype)
        xq = xp.clamp(x, xn[0], xn[-1])
        idx = xp.searchsorted(xn, xq.reshape(-1).contiguous(), right=True)
        idx = xp.clamp(idx, 1, len(xn) - 1)
        x0, x1 = xn[idx - 1], xn[idx]
        f0, f1 = fn[idx - 1], fn[idx]
        w = (xq.reshape(-1) - x0) / (x1 - x0)
        return (f0 + w * (f1 - f0)).reshape(x.shape)
    if name == 'jax':
        return xp.interp(x, xp.asarray(xp_nodes), xp.asarray(fp_nodes))
    return np.interp(x, xp_nodes, fp_nodes)


def x_coll_diff(z, Tk, xHI, rho_b, backend='numpy'):
    """Collisional coupling x_coll — differentiable counterpart of :func:`x_coll`.

    Same physics; kappa_HH/kappa_eH come from linear backend interpolation of
    the fixed tables (vs cubic splev in the numpy original — agreement ~1e-3).
    Differentiable w.r.t. Tk, xHI and rho_b.

    Args:
        z:       Redshift (static float).
        Tk:      Gas kinetic temperature [K] (backend array, may carry grads).
        xHI:     Neutral fraction (backend array).
        rho_b:   Baryon density [H atoms / pcm^3] (backend array or scalar).
        backend: 'numpy' (default), 'jax' or 'torch'.
    """
    from .cosmo.differentiable import get_backend
    name, xp = get_backend(backend)

    n_HI = rho_b * xHI
    n_HII = rho_b * (1 - xHI)
    prefac = Tstar / A10 / T_cmb(z)

    # log-log interpolation: kappa(T) spans many decades and is close to a
    # power law between nodes, so linear-in-log agrees with the cubic spline
    # of the numpy path far better than linear-in-linear.
    logT = xp.log(Tk)
    kappa_eH = xp.exp(_interp_clamped(
        logT, np.log(_kappa_eH_raw['T']), np.log(_kappa_eH_raw['kappa']),
        name, xp))
    kappa_HH = xp.exp(_interp_clamped(
        logT, np.log(_kappa_HH_raw['T']), np.log(_kappa_HH_raw['kappa']),
        name, xp))
    return prefac * (kappa_HH * n_HI + kappa_eH * n_HII)


def s_alpha_diff(z, Tk, xHI, backend='numpy'):
    """Lyman-alpha suppression S_alpha — differentiable counterpart of
    :func:`S_alpha` (identical formula, backend exp/power ops).

    The cube root is where-guarded: d(x^{1/3})/dx diverges at x = 0, so fully
    ionized cells (xHI = 0) would inject NaNs into the backward pass.
    """
    from .cosmo.differentiable import get_backend
    _, xp = get_backend(backend)
    tau_GP = 3.0e5 * xHI * ((1 + z) / 7.0) ** 1.5
    pos = tau_GP > 0
    tau_safe = xp.where(pos, tau_GP, xp.ones_like(tau_GP))
    cbrt = xp.where(pos, (1e-6 * tau_safe) ** (1.0 / 3.0),
                    xp.zeros_like(tau_safe))
    return xp.exp(-0.803 * Tk ** (-2.0 / 3.0) * cbrt)


def dtb_diff(z, Tk, x_tot, delta_b, xHII, factor, backend='numpy'):
    """Brightness temperature dTb [mK] — differentiable counterpart of
    :meth:`GridDerivedPropertiesMixin._compute_dTb` (identical algebra).

    Args:
        z:       Redshift (static float).
        Tk:      Kinetic temperature grid [K].
        x_tot:   Total coupling x_alpha + x_coll.
        delta_b: Baryon overdensity grid.
        xHII:    Ionized fraction grid.
        factor:  ``beorn.cosmo.dTb_factor(parameters)`` — static scalar.
        backend: Accepted for API symmetry; the expression is pure broadcast
                 algebra, so it works on any backend array unchanged.
    """
    del backend
    return (factor * np.sqrt(1 + z) * (1 - T_cmb(z) / Tk)
            * (1 - xHII) * x_tot / (1 + x_tot) * (1 + delta_b))
