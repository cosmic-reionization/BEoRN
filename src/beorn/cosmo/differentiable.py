"""Backend-generic, differentiable counterparts of :mod:`.background`.

Pure functions of explicit cosmological parameters — no Parameters object, no
mutable state — so they compose with ``jax.jit`` / ``jax.grad`` and torch
autograd, and run on whatever device their tensor inputs live on (CUDA, MPS).

They complement (never replace) the numpy functions in :mod:`.background`:
the numpy defaults stay unchanged; these are the opt-in gpu/diff path
(issue #42, Phase 1: G2). The shared backend helpers here are also used by
the differentiable P(k) functions in :mod:`beorn.lpt.linear_power`.

- :func:`growth_factor`  D(a; Om) normalised to D(1)=1 — counterpart of
  :func:`.background.D`; differentiable w.r.t. a and Om
- :func:`growth_rate`    f1 = dlnD/dlna via autodiff (jax/torch) or FD (numpy)
- :func:`hubble_E`       E(z) = H(z)/H0 — counterpart of :func:`.background.E`

Note on accuracy: the growth integral uses ``n_nodes`` fixed nodes from a'→0
(the numpy :func:`.background.D` starts at a'=0.001 with 100 nodes); the two
agree to ~1e-4 relative, limited by the numpy discretisation.
"""
from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Backend helpers (shared with beorn.lpt.linear_power differentiable functions)
# ──────────────────────────────────────────────────────────────────────────────

def get_backend(backend: str):
    """Resolve a backend name to (name, array module)."""
    if backend is None or backend == 'numpy':
        return 'numpy', np
    name = str(backend).lower()
    if name == 'numpy':
        return name, np
    if name == 'jax':
        import jax.numpy as jnp
        return name, jnp
    if name == 'torch':
        import torch
        return name, torch
    raise ValueError(f"backend must be 'numpy', 'jax' or 'torch'; got {backend!r}")


def device_of(name, xp, *args):
    """Device of the first torch tensor among args (None otherwise)."""
    if name == 'torch':
        for a in args:
            if xp.is_tensor(a):
                return a.device
    return None


def _torch_dtype(xp, device):
    """float64, except float32 on MPS (Apple GPU has no float64)."""
    if device is not None and str(device).startswith('mps'):
        return xp.float32
    return xp.float64


def as_const(x, name, xp, device=None):
    """Convert a static numpy constant to the backend (never carries grads)."""
    if name == 'torch':
        return xp.as_tensor(np.asarray(x, dtype=float),
                            dtype=_torch_dtype(xp, device), device=device)
    if name == 'jax':
        return xp.asarray(np.asarray(x, dtype=float))
    return np.asarray(x, dtype=float)


def as_array(x, name, xp, device=None):
    """Convert an input that may carry gradients, preserving tracers/tensors.

    Torch tensors on a different device are moved to *device* (``.to`` keeps
    the autograd graph); float64 is cast to float32 when the target is MPS
    (Apple GPU has no float64).
    """
    if name == 'torch':
        if xp.is_tensor(x):
            if device is not None and x.device != device:
                if str(device).startswith('mps') and x.dtype == xp.float64:
                    return x.to(device=device, dtype=xp.float32)
                return x.to(device=device)
            return x
        return xp.as_tensor(np.asarray(x, dtype=float),
                            dtype=_torch_dtype(xp, device), device=device)
    if name == 'jax':
        return xp.asarray(x)
    return np.asarray(x, dtype=float)


def trapz_static(y, x_np, name, xp, axis=0):
    """Trapezoid rule with static numpy nodes ``x_np`` along ``axis`` of y."""
    dx = as_const(np.diff(x_np), name, xp,
                  device=y.device if name == 'torch' else None)
    sl1 = [slice(None)] * y.ndim
    sl0 = [slice(None)] * y.ndim
    sl1[axis] = slice(1, None)
    sl0[axis] = slice(None, -1)
    shape = [1] * y.ndim
    shape[axis] = -1
    return ((y[tuple(sl1)] + y[tuple(sl0)]) * 0.5
            * dx.reshape(shape)).sum(axis)


# ──────────────────────────────────────────────────────────────────────────────
# Growth factor and rate (differentiable counterparts of background.D / E)
# ──────────────────────────────────────────────────────────────────────────────

def _growth_non_normalised(a, Om, name, xp, t_np):
    """5/2 Om E(a) ∫_0^a da' / (a' E(a'))^3 with a' = a·t, t static."""
    device = device_of(name, xp, a, Om)
    a = as_array(a, name, xp, device)
    Om = as_array(Om, name, xp, device)
    t = as_const(t_np, name, xp, device)
    a1 = a.reshape(-1)
    ap = t.reshape(-1, 1) * a1.reshape(1, -1)          # (n_nodes, n_a)
    E_ap = xp.sqrt(Om * ap ** -3 + 1.0 - Om)
    integrand = 1.0 / (ap * E_ap) ** 3
    w = a1 * trapz_static(integrand, t_np, name, xp, axis=0)
    E_a = xp.sqrt(Om * a1 ** -3 + 1.0 - Om)
    out = 2.5 * Om * E_a * w
    return out.reshape(a.shape) if a.shape else out.reshape(())


def growth_factor(a, Om, backend='numpy', n_nodes=512):
    """Linear growth factor D(a; Om) for flat LCDM, normalised to D(1) = 1.

    Differentiable w.r.t. both ``a`` and ``Om`` (pass jax tracers or torch
    tensors with ``requires_grad``); runs on the device of its tensor inputs.
    """
    name, xp = get_backend(backend)
    t_np = np.linspace(0.0, 1.0, n_nodes + 1)[1:]
    one = as_const(1.0, name, xp, device_of(name, xp, a, Om))
    return (_growth_non_normalised(a, Om, name, xp, t_np)
            / _growth_non_normalised(one, Om, name, xp, t_np))


def growth_rate(a, Om, backend='numpy', n_nodes=512, eps=1e-4):
    """Growth rate f1 = dlnD/dlna.

    jax/torch: exact autodiff through the growth integral. numpy: central
    finite difference (mirrors ``LPTBase._f1``).
    """
    name, xp = get_backend(backend)
    if name == 'jax':
        import jax

        def lnD(lna):
            return xp.log(growth_factor(xp.exp(lna), Om, backend, n_nodes))

        return jax.grad(lnD)(xp.log(xp.asarray(float(a))))
    if name == 'torch':
        a_t = xp.as_tensor(float(a), dtype=xp.float64).requires_grad_(True)
        lnD = xp.log(growth_factor(a_t, Om, backend, n_nodes))
        (g,) = xp.autograd.grad(lnD, a_t)
        return (g * a_t).detach()
    Dp = growth_factor(a * (1 + eps), Om, backend, n_nodes)
    Dm = growth_factor(a * (1 - eps), Om, backend, n_nodes)
    return (np.log(Dp) - np.log(Dm)) / (2 * eps)


def hubble_E(z, Om, backend='numpy'):
    """Dimensionless Hubble rate E(z) = H(z)/H0 for flat LCDM.

    Differentiable counterpart of :func:`.background.E` (which takes the
    scale factor); here z is the redshift, matching :func:`.background.hubble`.
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, z, Om)
    z = as_array(z, name, xp, device)
    Om = as_array(Om, name, xp, device)
    return xp.sqrt(Om * (1.0 + z) ** 3 + 1.0 - Om)


def hubble_per_yr(z, Om, h0, backend='numpy'):
    """Hubble parameter [yr⁻¹], flat LCDM.

    Differentiable counterpart of :func:`.background.hubble_per_yr` — same
    ``h0 * 100 * sec_per_year/km_per_Mpc`` unit conversion, built on
    :func:`hubble_E` instead of :func:`.background.dark_energy_density_factor`,
    so (like :func:`hubble_E`) this is flat-LCDM only — the production
    version's CPL ``w0``/``wa`` dark-energy support isn't carried over here,
    the same pre-existing limitation :func:`.precomputation.differentiable.bubble_radius_diff`
    already has via its own use of :func:`hubble_E`.
    """
    from ..constants import sec_per_year, km_per_Mpc

    name, xp = get_backend(backend)
    device = device_of(name, xp, z, Om, h0)
    h0 = as_array(h0, name, xp, device)
    return h0 * 100.0 * sec_per_year / km_per_Mpc * hubble_E(z, Om, backend=backend)
