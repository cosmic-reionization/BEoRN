"""Halo Mass Function models for BEoRN.

All models share a common :class:`MassFunction` base that precomputes
sigma^2(M, z) from BEoRN's linear power spectrum.  The recommended
entry point is :class:`HaloMassFunction`.

Quick start::

    from beorn.mass_function import HaloMassFunction
    from beorn.structs import Parameters
    import numpy as np

    param = Parameters()
    hmf = HaloMassFunction(param)           # Sheth-Tormen, top-hat, numpy
    M   = np.logspace(8, 14, 60)
    n   = hmf.dndlnm(M, z=7.0)             # (Mpc/h)^{-3}

    # Named runners — single sigma^2 precomputation, multiple models
    n_ps = hmf.run_press_schechter(M, z=7.0)
    n_st = hmf.run_sheth_tormen(M, z=7.0)

    # Window functions
    hmf_sk = HaloMassFunction(param, window='sharp_k')

    # JAX gradient of n(>Mmin) w.r.t. delta_c
    import jax
    hmf_jax = HaloMassFunction(param, backend='jax')
    dlnM = float(np.log(M[1] / M[0]))
    dn_dc = jax.grad(
        lambda dc: hmf_jax.dndlnm(M, z=7.0, delta_c=dc).sum() * dlnM
    )(1.686)

Low-level / legacy classes (preserved for backwards compatibility)::

    from beorn.mass_function import PressSchechter, ShethTormen

Differentiability
-----------------
``HaloMassFunction`` with ``backend='jax'`` or ``backend='torch'`` evaluates
f(nu) and dn/dlnM in the chosen framework, enabling ``jax.grad`` /
``torch.autograd`` w.r.t. ``delta_c``, ``p``, ``q``, and ``A``.

Full cosmological-parameter gradients (theta → P(k) → sigma^2 → n) require
a JAX-native power spectrum and growth factor; that extension is tracked in
GitHub issue #39.
"""
from .base import MassFunction as MassFunction
from .models import (
    HaloMassFunction as HaloMassFunction,
    ParametricHMF as ParametricHMF,
    PressSchechter as PressSchechter,
    ShethTormen as ShethTormen,
)
from .window import (
    TopHatWindow as TopHatWindow,
    SharpKWindow as SharpKWindow,
    SmoothKWindow as SmoothKWindow,
    get_window as get_window,
)

__all__ = [
    'MassFunction',
    'HaloMassFunction',
    'ParametricHMF',
    'PressSchechter',
    'ShethTormen',
    'TopHatWindow',
    'SharpKWindow',
    'SmoothKWindow',
    'get_window',
]
