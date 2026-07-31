"""Block-average and Fourier upsampling (issue #48).

``coarsen_field`` generalizes ``BaseNbodyLoader._coarsen_density``
(``beorn.load_input_data.nbody_base``) into a public utility, since
fine-paint-then-downsample (paint onto a finer mesh, then coarsen back to the
analysis grid) needs the same block-average as the N-body loader's
``degrade_resolution``.

``upsample_field_fourier`` is the other half of that pipeline: painting the
*same* particles onto a finer mesh and then coarsening back down is a no-op
(each particle's mass stays within its own coarse block regardless of the
mesh resolution it's painted onto, so block-averaging just reproduces the
original coarse field to machine precision — verified while implementing
this). Recovering real sub-cell structure requires *more* particles sampling
the same underlying (band-limited, smooth) displacement field at a finer
Lagrangian grid — i.e. Fourier-interpolating ``psi_x``/``psi_y``/``psi_z``
onto a finer real-space grid before building fine-grid positions from it.
"""
import numpy as np

from .core import _infer_functional_backend


def _resample_1d(x, num: int, axis: int, backend: str):
    """Band-limited upsampling along one axis via rfft zero-padding.

    Equivalent to ``scipy.signal.resample`` for the upsampling case (verified
    numerically, exact match including the even-N Nyquist-bin correction —
    the only edge case that matters here since we always paint x onto a
    *finer* grid), but implemented directly on each backend's own FFT so it
    stays differentiable / device-resident for jax and torch, unlike
    ``scipy.signal.resample`` (which silently converts jax/torch inputs to
    plain numpy internally).
    """
    n_x = x.shape[axis]
    s_fac = n_x / num  # < 1 for upsampling
    idx = [slice(None)] * x.ndim
    idx[axis] = n_x // 2
    idx = tuple(idx)

    if backend == 'jax':
        import jax.numpy as jnp
        X = jnp.fft.rfft(x, axis=axis)
        if n_x % 2 == 0:
            X = X.at[idx].multiply(0.5)
        return jnp.fft.irfft(X / s_fac, n=num, axis=axis)

    if backend == 'torch':
        import torch
        X = torch.fft.rfft(x, dim=axis)
        if n_x % 2 == 0:
            X[idx] = X[idx] * 0.5
        return torch.fft.irfft(X / s_fac, n=num, dim=axis)

    X = np.fft.rfft(x, axis=axis)
    if n_x % 2 == 0:
        X[idx] *= 0.5
    return np.fft.irfft(X / s_fac, n=num, axis=axis)


def upsample_field_fourier(field, factor: int):
    """Band-limited Fourier upsampling of a periodic 3D field.

    Applies FFT-based (bandlimited) interpolation along each axis in turn.
    Adds no new small-scale power beyond what ``field`` already contains — it
    evaluates the *same* smooth, band-limited function at more sample points,
    it does not extrapolate new k-modes.

    Differentiable / device-resident for jax and torch (functional contract,
    issue #42's G4): a jax array or torch tensor in gives a jax array or
    torch tensor out, gradient graph intact, no numpy round-trip. Backend is
    inferred from ``field``'s type.

    Args:
        field: Input 3D array, shape ``(N, N, N)``, periodic.
        factor: Upsampling factor.

    Returns:
        Array of shape ``(N*factor, N*factor, N*factor)``, same array family
        as ``field``.
    """
    backend = _infer_functional_backend(field)
    N_fine = field.shape[0] * factor
    out = _resample_1d(field, N_fine, 0, backend)
    out = _resample_1d(out, N_fine, 1, backend)
    out = _resample_1d(out, N_fine, 2, backend)
    return out


def coarsen_field(field: np.ndarray, factor: int) -> np.ndarray:
    """Block-average a 3D field by an integer factor.

    Each ``factor**3`` block of cells is averaged into one output cell. This
    conserves the mean: ``coarsen_field(field, factor).mean() ==
    field.mean()``.

    Works unmodified on jax arrays and torch tensors — ``.reshape()``/
    ``.mean(axis=...)`` are supported identically across numpy/jax/torch, so
    no backend dispatch is needed here (verified numerically).

    Args:
        field: Input 3D array, shape ``(N, N, N)``.
        factor: Downsampling factor. ``N`` must be divisible by ``factor``.

    Returns:
        Coarsened array, shape ``(N//factor, N//factor, N//factor)``.

    Raises:
        ValueError: If any axis of ``field`` is not divisible by ``factor``.
    """
    N = field.shape[0]
    if any(s % factor != 0 for s in field.shape):
        raise ValueError(
            f"Cannot coarsen by factor {factor}: "
            f"grid shape {field.shape} is not divisible on all axes."
        )
    Nc = N // factor
    return (
        field.reshape(Nc, factor, Nc, factor, Nc, factor)
             .mean(axis=(1, 3, 5))
    )
