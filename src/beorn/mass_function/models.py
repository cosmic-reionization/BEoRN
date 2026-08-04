"""HMF models: parametric f(nu) family with multi-backend support.

All parametric models share the unified first-crossing distribution

    f(nu) = A * sqrt(2 q nu / pi) * (1 + (q nu)^{-p}) * exp(-q nu / 2)

where nu = delta_c^2 / sigma^2(M, z)  (squared peak height).

When p = 0 the factor (q*nu)^{-p} = 1, giving correction = 2 — this
reproduces Press-Schechter exactly without a special branch.

Named models
------------
Press & Schechter (1974)  — PS:  A=0.5,    p=0,   q=1
Sheth & Tormen (2002)     — ST:  A≈0.3222, p=0.3, q=0.707
Ellipsoidal collapse      — EC:  A≈0.3222, p=0.3, q=1

Backend support
---------------
``'numpy'``  — default, returns numpy arrays.
``'jax'``    — f(nu) and dndlnm computed with JAX; supports jax.grad /
               jax.jit.  sigma^2(M) is pre-computed in numpy (constant);
               gradients flow through delta_c, p, q, A.
``'torch'``  — same idea with torch.autograd.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import gamma

from ..structs import Parameters
from .base import MassFunction


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_A(p: float) -> float:
    """Self-consistent normalisation: int f(nu) d nu = 1."""
    return 1.0 / (1.0 + 2.0 ** (-p) * gamma(0.5 - p) / math.pi ** 0.5)


def _to_backend(x, backend: str):
    """Convert a numpy scalar/array to the chosen backend's array type."""
    x_np = np.asarray(x, dtype=float)
    if backend == 'numpy':
        return x_np
    if backend == 'jax':
        import jax.numpy as jnp
        return jnp.asarray(x_np)
    if backend == 'torch':
        import torch
        return torch.as_tensor(x_np, dtype=torch.float64)
    raise ValueError(f"Unknown backend '{backend}'. Choose from 'numpy', 'jax', 'torch'.")


def _f_nu(nu, p: float, q: float, A: float, backend: str = 'numpy'):
    """Compute f(nu) = A sqrt(2 q nu / pi) (1 + (q nu)^{-p}) exp(-q nu / 2).

    The ``nu`` argument may be a numpy array, JAX array, or torch tensor;
    ``p``, ``q``, ``A`` are Python floats.  When ``backend`` is ``'jax'``
    or ``'torch'``, the result is a differentiable tensor of the same type.

    Note: (q*nu)^{-p} = 1 when p=0 (0^0 = 1 convention), so the formula
    is valid for all p ≥ 0 without a special branch.
    """
    if backend == 'jax':
        import jax.numpy as xp
    elif backend == 'torch':
        import torch as xp
    else:
        xp = np

    term = xp.sqrt(2.0 * q * nu / math.pi)
    correction = 1.0 + (q * nu) ** (-p)
    return A * term * correction * xp.exp(-0.5 * q * nu)


# ──────────────────────────────────────────────────────────────────────────────
# Legacy classes (kept for backwards compatibility)
# ──────────────────────────────────────────────────────────────────────────────

class ParametricHMF(MassFunction):
    """HMF with the generalised Press-Schechter first-crossing distribution.

    Args:
        parameters: BEoRN :class:`~beorn.structs.Parameters` object.
        p, q:       Shape parameters of f(nu).
        A:          Normalisation.  If ``None`` (default) the self-consistent
                    value ``1 / (1 + 2^{-p} Gamma(0.5-p)/sqrt(pi))`` is used.
        delta_c:    Linear collapse threshold (default 1.686).
        **kwargs:   Forwarded to :class:`~beorn.mass_function.base.MassFunction`.
    """

    def __init__(
        self,
        parameters: Parameters,
        p: float,
        q: float,
        A: float | None = None,
        delta_c: float = 1.686,
        **kwargs,
    ):
        super().__init__(parameters, **kwargs)
        self.p = p
        self.q = q
        self.A = A if A is not None else _normalise_A(p)
        self.delta_c = delta_c

    def f_nu(self, nu: np.ndarray) -> np.ndarray:
        return _f_nu(nu, self.p, self.q, self.A, backend='numpy')

    def dndlnm(self, M: np.ndarray, z: float) -> np.ndarray:
        M = np.atleast_1d(np.asarray(M, dtype=float))
        eps = 0.01
        s2_p = self.sigma2(M * (1.0 + eps), z)
        s2_m = self.sigma2(M * (1.0 - eps), z)
        sigma_M = np.sqrt(self.sigma2(M, z))

        dln_sigma_dlnM = (
            np.log(np.sqrt(s2_p)) - np.log(np.sqrt(s2_m))
        ) / (2.0 * eps)

        nu = (self.delta_c / sigma_M) ** 2
        f = self.f_nu(nu)
        return (self.rho_m / M) * np.abs(dln_sigma_dlnM) * f


class PressSchechter(ParametricHMF):
    """Press-Schechter (1974) spherical collapse: (A, p, q) = (0.5, 0, 1)."""

    def __init__(self, parameters: Parameters, delta_c: float = 1.686, **kwargs):
        super().__init__(parameters, p=0.0, q=1.0, A=0.5, delta_c=delta_c, **kwargs)


class ShethTormen(ParametricHMF):
    """Sheth & Tormen (2002) ellipsoidal collapse: (A≈0.3222, p=0.3, q=0.707)."""

    def __init__(self, parameters: Parameters, delta_c: float = 1.686, **kwargs):
        super().__init__(parameters, p=0.3, q=0.707, delta_c=delta_c, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Unified HaloMassFunction
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_PARAMS: dict[str, tuple[float, float, float | None]] = {
    # name            : (p,    q,     A or None=self-consistent)
    'press_schechter' : (0.0,  1.0,   0.5),
    'ps'              : (0.0,  1.0,   0.5),
    'sheth_tormen'    : (0.3,  0.707, None),
    'st'              : (0.3,  0.707, None),
    'ellipsoidal'     : (0.3,  1.0,   None),
}


class HaloMassFunction(MassFunction):
    """Unified parametric HMF with window-function and backend selection.

    Sigma^2(M) is pre-computed from BEoRN's linear power spectrum using the
    chosen window function.  The first-crossing distribution f(nu) and the
    full dn/dlnM are evaluated in the selected backend, enabling JAX / torch
    autodiff through ``delta_c``, ``p``, ``q``, and ``A``.

    Args:
        parameters: BEoRN :class:`~beorn.structs.Parameters` object.
        model:      Named model or ``'custom'``.  Accepted values:
                    ``'sheth_tormen'``/``'st'`` (default),
                    ``'press_schechter'``/``'ps'``,
                    ``'ellipsoidal'``, ``'custom'``.
        window:     Filter function — ``'tophat'`` (default), ``'sharp_k'``,
                    ``'smooth_k'``, or a
                    :class:`~beorn.mass_function.window.Window` instance.
        backend:    Compute backend — ``'numpy'``, ``'jax'``, ``'torch'``.
                    ``None`` (default) reads
                    ``parameters.simulation.backend.resolve('hmf')`` (itself
                    ``'numpy'`` unless overridden — see
                    :class:`~beorn.structs.BackendParameters`). Unlike
                    :class:`~beorn.lpt.LPTBase`'s own ``backend``, this one
                    genuinely changes the returned array type — ``'jax'``/
                    ``'torch'`` give differentiable, device-resident output.
        p, q:       Shape parameters; override model defaults or required for
                    ``model='custom'``.
        A:          Normalisation; ``None`` → self-consistent value.
        delta_c:    Linear collapse threshold (default 1.686).
        **kwargs:   Forwarded to :class:`~beorn.mass_function.base.MassFunction`
                    (e.g. ``ps_method``, ``window`` kwargs like ``beta``).

    Examples::

        from beorn.mass_function import HaloMassFunction
        from beorn.structs import Parameters
        import numpy as np

        param = Parameters()
        hmf = HaloMassFunction(param)                   # ST, tophat, numpy
        M   = np.logspace(8, 14, 60)
        n   = hmf.dndlnm(M, z=7.0)                     # (Mpc/h)^{-3}

        # Named runners — same sigma^2 precomputation, different (p, q)
        n_ps = hmf.run_press_schechter(M, z=7.0)
        n_st = hmf.run_sheth_tormen(M, z=7.0)

        # JAX gradient of n(>Mmin) w.r.t. delta_c
        import jax
        hmf_jax = HaloMassFunction(param, backend='jax')
        dlnM = float(np.log(M[1] / M[0]))
        dn_dc = jax.grad(
            lambda dc: hmf_jax.dndlnm(M, z=7.0, delta_c=dc).sum() * dlnM
        )(1.686)
    """

    def __init__(
        self,
        parameters: Parameters,
        model: str = 'sheth_tormen',
        window: str = 'tophat',
        backend: str | None = None,
        p: float | None = None,
        q: float | None = None,
        A: float | None = None,
        delta_c: float = 1.686,
        **kwargs,
    ):
        super().__init__(parameters, window=window, **kwargs)
        self.backend = (
            backend if backend is not None
            else parameters.simulation.backend.resolve('hmf')
        )
        self.delta_c = delta_c

        if model == 'custom':
            if p is None or q is None:
                raise ValueError("Supply p and q when model='custom'.")
            _p, _q, _A_default = p, q, None
        elif model in _MODEL_PARAMS:
            _p, _q, _A_default = _MODEL_PARAMS[model]
        else:
            raise ValueError(
                f"Unknown model '{model}'. "
                f"Choose from {list(_MODEL_PARAMS)} or 'custom'."
            )

        self.model = model
        self.p     = _p   if p is None else p
        self.q     = _q   if q is None else q
        _A_norm    = _A_default if _A_default is not None else _normalise_A(self.p)
        self.A     = _A_norm if A is None else A

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def f_nu(self, nu, p: float | None = None, q: float | None = None,
             A: float | None = None):
        """First-crossing distribution f(nu) in the chosen backend.

        ``nu`` may be a Python/numpy scalar, JAX array, or torch tensor;
        the result is the same type.  ``p``, ``q``, ``A`` default to the
        values set at construction time.
        """
        return _f_nu(
            nu,
            p if p is not None else self.p,
            q if q is not None else self.q,
            A if A is not None else self.A,
            self.backend,
        )

    def dndlnm(
        self,
        M,
        z: float,
        p: float | None = None,
        q: float | None = None,
        A: float | None = None,
        delta_c=None,
    ):
        """dn/d ln M = (rho_m / M) |d ln sigma / d ln M| f(nu).

        Args:
            M:        Halo masses in M_sun (numpy array or any array-like).
            z:        Redshift.
            p, q, A:  Override model (p, q, A) for this call — useful with
                      named runners or when sweeping parameters.
            delta_c:  Override collapse threshold.  Accepts a JAX/torch
                      scalar for gradient computation.

        Returns:
            dn/d ln M in (Mpc/h)^{-3}, in the chosen backend's array type.
        """
        # sigma^2 always computed in numpy (from the pre-interpolated grid)
        M_np = np.atleast_1d(np.asarray(M, dtype=float))
        eps = 0.01
        s2_p = self.sigma2(M_np * (1.0 + eps), z)
        s2_m = self.sigma2(M_np * (1.0 - eps), z)
        sigma_M_np = np.sqrt(self.sigma2(M_np, z))
        dln_np = np.abs(
            (np.log(np.sqrt(s2_p)) - np.log(np.sqrt(s2_m))) / (2.0 * eps)
        )

        # Convert fixed arrays to backend (constants — no gradient through these)
        sigma_M_b = _to_backend(sigma_M_np, self.backend)
        dln_b     = _to_backend(dln_np,     self.backend)
        M_b       = _to_backend(M_np,       self.backend)
        rho_m_b   = _to_backend(self.rho_m, self.backend)

        # delta_c may be a JAX/torch tracer — do NOT call _to_backend on it
        dc = (
            _to_backend(self.delta_c, self.backend)
            if delta_c is None
            else delta_c
        )

        nu = (dc / sigma_M_b) ** 2
        f  = self.f_nu(nu, p=p, q=q, A=A)
        return (rho_m_b / M_b) * dln_b * f

    # ------------------------------------------------------------------
    # Named runners (convenience wrappers around dndlnm)
    # ------------------------------------------------------------------

    def run_press_schechter(self, M, z: float):
        """dn/dlnM with Press-Schechter (p=0, q=1, A=0.5)."""
        return self.dndlnm(M, z, p=0.0, q=1.0, A=0.5)

    def run_sheth_tormen(self, M, z: float):
        """dn/dlnM with Sheth-Tormen (p=0.3, q=0.707)."""
        return self.dndlnm(M, z, p=0.3, q=0.707, A=_normalise_A(0.3))

    def run_ellipsoidal(self, M, z: float):
        """dn/dlnM with ellipsoidal collapse (p=0.3, q=1)."""
        return self.dndlnm(M, z, p=0.3, q=1.0, A=_normalise_A(0.3))
