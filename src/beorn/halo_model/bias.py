"""Linear halo bias b(M, z) as a function of sigma(M, z)."""
from __future__ import annotations

import numpy as np

from ..cosmo.differentiable import get_backend, as_array


def halo_bias(sigma, model: str = 'sheth_tormen', delta_c: float = 1.686,
              backend: str = 'numpy'):
    """Large-scale halo bias.

    Args:
        sigma:   sigma(M, z) (array).
        model:   ``'press_schechter'`` / ``'ps'`` (Mo & White 1996),
                 ``'sheth_tormen'`` / ``'st'`` (Sheth, Mo & Tormen 2001,
                 peak-background split of the ST mass function), or
                 ``'tinker10'`` (Tinker et al. 2010, Delta = 200 mean).
        delta_c: Linear collapse threshold.
        backend: 'numpy', 'jax' or 'torch'.

    Returns:
        b, same shape as ``sigma``.
    """
    name, xp = get_backend(backend)
    nu = delta_c / as_array(sigma, name, xp)
    key = model.lower()
    if key in ('press_schechter', 'ps'):
        return 1.0 + (nu ** 2 - 1.0) / delta_c
    if key in ('sheth_tormen', 'st'):
        a, p = 0.707, 0.3
        anu2 = a * nu ** 2
        return 1.0 + (anu2 - 1.0) / delta_c + 2.0 * p / (delta_c * (1.0 + anu2 ** p))
    if key == 'tinker10':
        y = np.log10(200.0)
        e = np.exp(-(4.0 / y) ** 4)
        A, a = 1.0 + 0.24 * y * e, 0.44 * y - 0.88
        B, b = 0.183, 1.5
        C, c = 0.019 + 0.107 * y + 0.19 * e, 2.4
        return 1.0 - A * nu ** a / (nu ** a + delta_c ** a) + B * nu ** b + C * nu ** c
    raise ValueError(f"Unknown bias model {model!r}; use 'press_schechter', 'sheth_tormen' or 'tinker10'.")
