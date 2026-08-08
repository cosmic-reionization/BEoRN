"""Backend-generic, differentiable counterparts of :mod:`.astro`.

Pure functions of explicit astro parameters — no ``Parameters`` object — so
they compose with ``jax.grad``/``jax.jit`` and torch autograd, following the
same ``backend='numpy'|'jax'|'torch'`` convention as :mod:`.cosmo.differentiable`
and :mod:`.precomputation.differentiable`. ``backend='numpy'`` is **not**
differentiable — plain NumPy has no autodiff; it's the default only so the
same code path can be checked numerically against the production functions
below without requiring jax/torch. Gradients require ``backend='jax'`` or
``'torch'``.

They complement (never replace) :mod:`.astro`'s numpy functions, and are
ported 1:1 from them — not a redesign:

- :func:`f_star_halo_diff`  — counterpart of :func:`.astro.f_star_Halo`
- :func:`f_esc_diff`        — counterpart of :func:`.astro.f_esc`
"""
from __future__ import annotations

from .cosmo.differentiable import get_backend, device_of, as_array

__all__ = [
    'f_star_halo_diff',
    'f_esc_diff',
]


def _s_fct(Mh, Mt, g3, g4, xp):
    """Small-scale efficiency modifier — counterpart of :func:`.astro.S_fct`."""
    return (1.0 + (Mt / Mh) ** g3) ** g4


def f_star_halo_diff(Mh, f_st, Mp, g1, g2, Mt, g3, g4, halo_mass_min,
                      backend='numpy'):
    """Star-formation efficiency f_star(M) — differentiable counterpart of
    :func:`.astro.f_star_Halo` (double power law × :func:`.astro.S_fct`,
    arXiv:2305.15466 Eq. 5).

    Differentiable w.r.t. ``Mh`` and every shape parameter (``f_st``, ``Mp``,
    ``g1``, ``g2``, ``Mt``, ``g3``, ``g4``) when ``backend='jax'``/``'torch'``.
    ``halo_mass_min`` is treated as a static cutoff (not differentiated
    through), matching how it's used in production — a hard mass floor, not
    a smooth model parameter.

    Args:
        Mh: Halo mass, Msun/h (backend array; may carry gradients).
        f_st, Mp, g1, g2, Mt, g3, g4: Double-power-law shape parameters
            (scalars; may carry gradients).
        halo_mass_min: Static mass floor, Msun/h — ``f_star`` is zero below
            this (numpy float, not differentiated).
        backend: 'numpy' (default, not differentiable), 'jax' or 'torch'.

    Returns:
        f_star(Mh) in [0, 1], same shape as ``Mh``.
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, Mh, f_st, Mp, g1, g2, Mt, g3, g4)
    Mh = as_array(Mh, name, xp, device)
    f_st = as_array(f_st, name, xp, device)
    Mp = as_array(Mp, name, xp, device)
    g1 = as_array(g1, name, xp, device)
    g2 = as_array(g2, name, xp, device)
    Mt = as_array(Mt, name, xp, device)
    g3 = as_array(g3, name, xp, device)
    g4 = as_array(g4, name, xp, device)

    fstar = 2.0 * f_st / ((Mh / Mp) ** g1 + (Mh / Mp) ** g2) * _s_fct(Mh, Mt, g3, g4, xp)
    one = xp.ones_like(fstar)
    fstar = xp.where(fstar < one, fstar, one)
    zero = xp.zeros_like(fstar)
    return xp.where(Mh < halo_mass_min, zero, fstar)


def f_esc_diff(Mh, f0_esc, Mp_esc, pl_esc, backend='numpy'):
    """Escape fraction f_esc(M) — differentiable counterpart of
    :func:`.astro.f_esc` (power law in halo mass).

    Differentiable w.r.t. ``Mh``, ``f0_esc``, ``Mp_esc``, ``pl_esc`` when
    ``backend='jax'``/``'torch'``.

    Args:
        Mh: Halo mass, Msun/h (backend array; may carry gradients).
        f0_esc, Mp_esc, pl_esc: Power-law shape parameters (scalars; may
            carry gradients).
        backend: 'numpy' (default, not differentiable), 'jax' or 'torch'.

    Returns:
        f_esc(Mh), clipped to a maximum of 1, same shape as ``Mh``.
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, Mh, f0_esc, Mp_esc, pl_esc)
    Mh = as_array(Mh, name, xp, device)
    f0_esc = as_array(f0_esc, name, xp, device)
    Mp_esc = as_array(Mp_esc, name, xp, device)
    pl_esc = as_array(pl_esc, name, xp, device)

    fesc = f0_esc * (Mp_esc / Mh) ** pl_esc
    one = xp.ones_like(fesc)
    return xp.where(fesc < one, fesc, one)
