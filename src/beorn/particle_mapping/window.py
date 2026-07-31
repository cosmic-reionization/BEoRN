"""Mass-assignment window functions and P(k)-only deconvolution (issue #48).

Painting particles onto a mesh via NGP/CIC/TSC/PCS convolves the true field
with a scheme-specific real-space kernel; in Fourier space this multiplies
the field by a window

    W(k) = prod_i sinc(k_i * dx / 2)^p

where ``dx`` is the cell size and ``p`` is the kernel order (NGP=1, CIC=2,
TSC=3, PCS=4 — matching the number of stencil points per axis in
:mod:`beorn.particle_mapping.numpy_backend`). This suppresses P(k) near
k_Nyquist as a pure numerical artefact, not physics.

:func:`deconvolve_mas` undoes this for power-spectrum purposes only — dividing
by a near-zero window amplifies noise near k_Nyquist, so the returned field is
not safe to use for real-space statistics (see its docstring).
"""
from __future__ import annotations

import numpy as np

from .core import _infer_functional_backend

_MAS_ORDER = {'NGP': 1, 'CIC': 2, 'TSC': 3, 'PCS': 4}


def k_nyquist(L: float, N: int) -> float:
    """Nyquist wavenumber ``pi * N / L`` for an ``N``-cell grid of side ``L``.

    Matches ``tools21cm.power_spectrum._resolve_k_limit``'s
    ``k_limit='nyquist'`` convention exactly, so this always agrees with what
    ``t2c.power_spectrum_1d(..., k_limit='nyquist')`` uses internally.

    Args:
        L: Box side length.
        N: Number of cells per side.

    Returns:
        k_Nyquist in the same angular-wavenumber units as the box (e.g.
        Mpc/h if ``L`` is in Mpc/h).
    """
    return np.pi * N / L


def _mas_window(L: float, N: int, mass_assignment: str) -> np.ndarray:
    """W(k) on the ``rfftn`` grid for an ``(N,N,N)`` box of side ``L``."""
    scheme = mass_assignment.upper()
    if scheme not in _MAS_ORDER:
        raise ValueError(
            f"Unknown mass_assignment {mass_assignment!r}. "
            f"Choose from {tuple(_MAS_ORDER)}."
        )
    p = _MAS_ORDER[scheme]
    dx = L / N

    def _sinc1d(k):
        # np.sinc(x) = sin(pi*x)/(pi*x) is normalized; the physics formula
        # wants the unnormalized sinc(k*dx/2) = sin(y)/y, i.e. np.sinc(y/pi).
        return np.sinc(k * dx / (2.0 * np.pi))

    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=dx)

    Wx = _sinc1d(kx) ** p
    Wy = _sinc1d(ky) ** p
    Wz = _sinc1d(kz) ** p
    return Wx[:, None, None] * Wy[None, :, None] * Wz[None, None, :]


def deconvolve_mas(field, L: float, mass_assignment: str):
    """Undo the mass-assignment window in Fourier space (Sefusatti et al. 2016).

    Divides the field's FFT by ``W(k)`` (see module docstring) and inverse-FFTs
    back to real space. This is a **P(k)-only fix**: the returned array is not
    safe to use for real-space statistics (persistence homology, Minkowski
    functionals, void-finding, ...) — dividing by a near-zero window amplifies
    noise near k_Nyquist by construction, especially for higher-order schemes
    (TSC/PCS) and near k-space cube corners where all three axes approach
    k_Nyquist at once. It exists purely as an intermediate to hand to a power
    spectrum estimator (e.g. :func:`beorn.power_spectrum.power_spectrum_1d`),
    which re-FFTs it anyway — the round-trip through real space is exact
    (up to floating-point noise), so this is not wasted or unsafe *for that
    purpose*.

    Differentiable / device-resident for jax and torch (functional contract,
    issue #42's G4): a jax array or torch tensor in gives a jax array or
    torch tensor out, gradient graph intact, no numpy round-trip. Backend is
    inferred from ``field``'s type — numpy input (or a plain Python
    sequence) always takes the numpy path.

    Args:
        field: Either a real-space field, shape ``(N, N, N)``, or an
            already-``rfftn``'d field, complex, shape ``(N, N, N//2+1)``
            (dispatched on "is this complex").
        L: Box side length (same units as ``field``'s implicit grid).
        mass_assignment: ``'NGP'``, ``'CIC'``, ``'TSC'``, or ``'PCS'`` — must
            match the scheme originally used to paint ``field``.

    Returns:
        Real-space array, shape ``(N, N, N)``, same array family as ``field``.
    """
    backend = _infer_functional_backend(field)
    N = field.shape[0]
    W = _mas_window(L, N, mass_assignment)  # plain numpy — static, real, small

    if backend == 'jax':
        import jax.numpy as jnp
        field_k = field if jnp.iscomplexobj(field) else jnp.fft.rfftn(field)
        Wb = jnp.asarray(W, dtype=field_k.real.dtype)
        return jnp.fft.irfftn(field_k / Wb, s=(N, N, N))

    if backend == 'torch':
        import torch
        field_k = field if torch.is_complex(field) else torch.fft.rfftn(field)
        Wb = torch.as_tensor(W, dtype=field_k.real.dtype, device=field_k.device)
        return torch.fft.irfftn(field_k / Wb, s=(N, N, N))

    field = np.asarray(field)
    field_k = field if np.iscomplexobj(field) else np.fft.rfftn(field)
    return np.fft.irfftn(field_k / W, s=(N, N, N))
