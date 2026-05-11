"""Window (filter) functions for sigma^2(M) smoothing.

All windows take x = k*R as argument and return W(x) in [0, 1].
M–R relation is always the top-hat sphere M = (4π/3) ρ_m R³ for all
windows — only the variance integral changes.

Available windows
-----------------
``'tophat'``   — real-space top-hat:  W(x) = 3 j_1(x) / x
``'sharp_k'``  — Fourier step cutoff: W(x) = Θ(1 − x)
``'smooth_k'`` — smooth k-space:      W(x) = (1 + x^β)^{-1}  (β=4 default)
"""
from __future__ import annotations

import numpy as np


class Window:
    """Abstract base for filter functions used in sigma^2(M) integrals."""

    def W(self, kR: np.ndarray) -> np.ndarray:
        """Window function W(kR), values in [0, 1].

        Args:
            kR: Dimensionless product k·R, any shape.

        Returns:
            W(kR), same shape.
        """
        raise NotImplementedError


class TopHatWindow(Window):
    """Real-space top-hat filter: W(x) = 3 j₁(x) / x = 3(sin x − x cos x)/x³."""

    def W(self, kR: np.ndarray) -> np.ndarray:
        x = np.asarray(kR, dtype=float)
        return np.where(
            x < 1e-3,
            1.0 - x ** 2 / 10.0 + x ** 4 / 280.0,
            3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3,
        )


class SharpKWindow(Window):
    """Fourier-space sharp cutoff: W(x) = Θ(1 − x)."""

    def W(self, kR: np.ndarray) -> np.ndarray:
        x = np.asarray(kR, dtype=float)
        return np.where(x <= 1.0, 1.0, 0.0)


class SmoothKWindow(Window):
    """Smooth k-space filter: W(x) = (1 + x^β)^{−1}.

    Args:
        beta: Steepness of the cutoff (default 4).
    """

    def __init__(self, beta: float = 4.0):
        self.beta = beta

    def W(self, kR: np.ndarray) -> np.ndarray:
        x = np.asarray(kR, dtype=float)
        return 1.0 / (1.0 + x ** self.beta)


_WINDOW_REGISTRY: dict[str, type] = {
    'tophat':   TopHatWindow,
    'sharp_k':  SharpKWindow,
    'smooth_k': SmoothKWindow,
}


def get_window(name: str | Window, **kwargs) -> Window:
    """Return a :class:`Window` instance by name or pass one through unchanged.

    Args:
        name:    ``'tophat'``, ``'sharp_k'``, ``'smooth_k'``, or a
                 :class:`Window` instance (returned as-is).
        **kwargs: Forwarded to the :class:`Window` constructor
                  (e.g. ``beta=4`` for :class:`SmoothKWindow`).

    Raises:
        ValueError: If *name* is an unknown string.
    """
    if isinstance(name, Window):
        return name
    if name not in _WINDOW_REGISTRY:
        raise ValueError(
            f"Unknown window '{name}'. "
            f"Choose from {list(_WINDOW_REGISTRY)}."
        )
    return _WINDOW_REGISTRY[name](**kwargs)
