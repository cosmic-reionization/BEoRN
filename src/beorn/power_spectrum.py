"""Power-spectrum estimation, thinly wrapping ``tools21cm`` (issue #48).

``tools21cm.power_spectrum.power_spectrum_1d`` has no mass-assignment
deconvolution and doesn't report k_Nyquist, so every P(k) computed from a
painted field (NGP/CIC/TSC/PCS) silently carries the mass-assignment window's
high-k suppression. :func:`power_spectrum_1d` adds an opt-in deconvolution
step (:func:`beorn.particle_mapping.deconvolve_mas`) and always reports
k_Nyquist alongside the usual ``tools21cm`` return value.
"""
import numpy as np
import tools21cm as t2c

from .particle_mapping import deconvolve_mas, k_nyquist as _k_nyquist
from .particle_mapping.core import _infer_functional_backend


def power_spectrum_1d(
    field,
    box_dims,
    mass_assignment: str | None = None,
    deconvolve: bool = False,
    **kwargs,
):
    """1D power spectrum, optionally deconvolving the mass-assignment window.

    Thin wrapper around ``tools21cm.power_spectrum.power_spectrum_1d`` — all
    of its options (``kbins``, ``binning``, ``window``, ``k_limit``,
    ``return_n_modes``, ``backend``, ...) are forwarded unchanged via
    ``**kwargs``. ``tools21cm.power_spectrum_1d`` already supports
    ``backend='jax'/'torch'`` for an end-to-end differentiable, GPU-capable
    estimator; this wrapper auto-infers that backend from ``field``'s type
    (unless the caller passes ``backend=`` explicitly) so jax/torch fields
    stay on-device and differentiable through both the (optional)
    deconvolution and the binning, with no numpy round-trip.

    Args:
        field: Real-space field, shape ``(N, N, N)`` — numpy array, jax
            array, or torch tensor.
        box_dims: Box side length (scalar, cubic box) or per-axis lengths, as
            accepted by ``tools21cm``.
        mass_assignment: ``'NGP'``, ``'CIC'``, ``'TSC'``, or ``'PCS'`` — the
            scheme ``field`` was painted with. Required if ``deconvolve=True``.
        deconvolve: If ``True``, apply
            :func:`beorn.particle_mapping.deconvolve_mas` to ``field`` before
            binning (P(k)-only fix for the mass-assignment window — see that
            function's docstring for why the deconvolved field itself
            shouldn't be reused for real-space statistics).
        **kwargs: Forwarded to ``tools21cm.power_spectrum.power_spectrum_1d``.

    Returns:
        Whatever ``tools21cm.power_spectrum.power_spectrum_1d`` returns
        (``(Pk, bins)`` or ``(Pk, bins, n_modes)`` if ``return_n_modes=True``),
        with k_Nyquist appended as the last element.
    """
    N = field.shape[0]
    L = box_dims if np.isscalar(box_dims) else box_dims[0]

    if deconvolve:
        if mass_assignment is None:
            raise ValueError(
                "deconvolve=True requires mass_assignment "
                "('NGP', 'CIC', 'TSC', or 'PCS')."
            )
        field = deconvolve_mas(field, L, mass_assignment)

    kwargs.setdefault('backend', _infer_functional_backend(field))
    result = t2c.power_spectrum.power_spectrum_1d(field, box_dims=box_dims, **kwargs)
    return (*result, _k_nyquist(L, N))
