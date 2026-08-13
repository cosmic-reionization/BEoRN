"""Backend-generic, differentiable LPT pipeline (issue #42, Phase 1: G3).

Pure functions of explicit cosmological parameters — no Parameters object, no
mutable state — composable under ``jax.jit`` / ``jax.grad`` and torch
autograd, and device-resident end to end: no ``to_numpy`` mid-chain, k-grids
built as broadcastable backend constants, FFTs on the device of the inputs.

They complement (never replace) the :class:`~beorn.lpt.lpt.LPTBase` classes:
the numpy class defaults stay byte-identical; these are the opt-in gpu/diff
path.  The chain mirrors the classes stage by stage::

    noise (static)                          rng, outside the graph
      └─ lpt_ics(noise, L, θ)               δ(k) — grads w.r.t. θ via P(k)
           ├─ lpt_displacement(δk, L, z, Om)   Ψ (1LPT / 2LPT)
           ├─ lpt_velocity(δk, L, z, Om)       v in km/s
           ├─ lpt_linear_density(δk, L, z, Om) IRFFT[D₁ W(kR) δ(k)]
           └─ lpt_density(δk, L, z, Om)        Ψ → positions → paint_mesh → δ(x)

θ = (Om, Ob, h0, ns, sigma_8).  Random noise is an explicit *input* so the
stochasticity stays out of the differentiated graph (reparameterisation).

Growth factors come from :func:`beorn.cosmo.differentiable.growth_factor`
(fixed-node backend integral) — they differ from the classes' legacy
``beorn.cosmo.D`` at the ~1e-4 level (documented quadrature difference), so
pure-vs-class parity checks should use rtol ≳ 1e-3.

Precision: jax runs float32 unless ``jax.config.update('jax_enable_x64',
True)`` is set — enable it for gradient checks.  Torch follows the dtype of
its inputs (float64 on CPU/CUDA; use float32 on MPS).
"""
from __future__ import annotations

import numpy as np

from ..cosmo.differentiable import (
    get_backend, device_of, as_array, as_const, growth_factor, growth_rate,
    hubble_E,
)
from .linear_power import pk_eh_nowiggle

__all__ = [
    'lpt_ics',
    'lpt_displacement',
    'lpt_velocity',
    'lpt_linear_density',
    'lpt_density',
]


# ──────────────────────────────────────────────────────────────────────────────
# Backend helpers (complex-aware; cosmo.differentiable.as_array is real-only)
# ──────────────────────────────────────────────────────────────────────────────

def _as_complex(x, name, xp, device=None):
    """Convert *x* to a backend complex array, preserving tracers/tensors."""
    if name == 'torch':
        if xp.is_tensor(x):
            return x if device is None else x.to(device)
        return xp.as_tensor(np.asarray(x, dtype=np.complex128), device=device)
    if name == 'jax':
        return xp.asarray(x)
    return np.asarray(x)


def _rfftn(x, name, xp):
    if name == 'torch':
        return xp.fft.rfftn(x)
    if name == 'jax':
        return xp.fft.rfftn(x)
    return np.fft.rfftn(x)


def _irfftn(xk, shape, name, xp):
    if name == 'torch':
        return xp.fft.irfftn(xk, s=shape)
    if name == 'jax':
        return xp.fft.irfftn(xk, s=shape).real
    return np.fft.irfftn(xk, s=shape)


def _k_grids(N, L, name, xp, device=None):
    """Broadcastable k-component grids as static backend constants.

    Returns:
        kx (N,1,1), ky (1,N,1), kz (1,1,N//2+1) in h/Mpc — broadcast shapes,
        never materialised at (N, N, N//2+1) (the O3 memory pattern).
    """
    dk = 2.0 * np.pi / L
    kx = (np.fft.fftfreq(N, d=1.0 / N) * dk)[:, None, None]
    ky = (np.fft.fftfreq(N, d=1.0 / N) * dk)[None, :, None]
    kz = (np.fft.rfftfreq(N, d=1.0 / N) * dk)[None, None, :]
    return (as_const(kx, name, xp, device),
            as_const(ky, name, xp, device),
            as_const(kz, name, xp, device))


def _scalar(x, name, xp, device):
    """Move a 0-dim backend value onto *device* without severing gradients."""
    return as_array(x, name, xp, device)


# ──────────────────────────────────────────────────────────────────────────────
# Initial conditions
# ──────────────────────────────────────────────────────────────────────────────

def lpt_ics(noise, L, Om, Ob, h0, ns, sigma_8, backend='numpy', fixed=True,
            n_nodes=512):
    """δ(k) at z = 0 from a white-noise cube — differentiable w.r.t. cosmology.

    Counterpart of :meth:`LPTBase.generate_initial_conditions`; the noise is
    an explicit argument (reparameterisation) so gradients w.r.t.
    (Om, Ob, h0, ns, sigma_8) flow through P(k) while the realisation stays
    fixed.

    Args:
        noise:   Real white-noise cube, shape (N, N, N) — numpy array or
                 backend tensor.  Draw it once outside the graph, e.g. with
                 ``np.random.default_rng(seed).standard_normal((N,)*3)``.
        L:       Box side length in Mpc/h.
        Om, Ob, h0, ns, sigma_8: cosmological parameters (scalars or 0-dim
                 tensors carrying gradients).
        backend: ``'numpy'`` (default), ``'jax'`` or ``'torch'``.
        fixed:   Fixed-amplitude ICs (|δ_k| = sqrt(P(k)), random phases only),
                 matching the class default.
        n_nodes: Quadrature nodes for the σ₈ normalisation integral.

    Returns:
        delta_k — complex backend array of shape (N, N, N//2+1), device of
        the tensor inputs.
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, noise, Om, Ob, h0, ns, sigma_8)
    noise = as_array(noise, name, xp, device)
    N = noise.shape[0]
    V = float(L) ** 3

    noise_k = _rfftn(noise, name, xp)

    if fixed:
        abs_nk = xp.abs(noise_k)
        target = float(np.sqrt(float(N) ** 3))
        safe = xp.where(abs_nk > 0, abs_nk, xp.ones_like(abs_nk))
        noise_k = xp.where(abs_nk > 0, noise_k / safe * target,
                           xp.zeros_like(noise_k))

    kx, ky, kz = _k_grids(N, L, name, xp, device)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k_safe = xp.sqrt(xp.where(k2 > 0, k2, xp.ones_like(k2)))

    Pk = pk_eh_nowiggle(k_safe, 0.0, Om, Ob, h0, ns, sigma_8,
                        backend=backend, n_nodes=n_nodes)
    amplitude = xp.sqrt(Pk * float(N) ** 3 / V)
    amplitude = xp.where(k2 > 0, amplitude, xp.zeros_like(amplitude))

    return noise_k * amplitude


# ──────────────────────────────────────────────────────────────────────────────
# Displacement / velocity
# ──────────────────────────────────────────────────────────────────────────────

def _combined_source_k(delta_k, N, L, z, Om, order, name, xp, device, n_nodes,
                       velocity=False):
    """Σₙ cₙ·sourceₙ(k) · 1/k², with cₙ the displacement (or velocity) growth
    coefficients — the k-space field whose i k/k² gradient is Ψ (or v)."""
    kx, ky, kz = _k_grids(N, L, name, xp, device)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    ik2 = xp.where(k2 > 0, 1.0 / xp.where(k2 > 0, k2, xp.ones_like(k2)),
                   xp.zeros_like(k2))

    a = 1.0 / (1.0 + z)
    D1 = _scalar(growth_factor(a, Om, backend=name, n_nodes=n_nodes),
                 name, xp, device)

    if velocity:
        # v⁽ⁿ⁾ = a·H/h·fₙ·Dₙ·Ψ̂⁽ⁿ⁾ with fₙ ≈ n f₁ (EdS, as in the classes).
        # H/h = 100·E(z) km/s/Mpc.
        f1 = _scalar(growth_rate(a, Om, backend=name, n_nodes=n_nodes),
                     name, xp, device)
        Ez = _scalar(hubble_E(z, Om, backend=name), name, xp, device)
        base = a * 100.0 * Ez * f1
        c1 = base * D1
    else:
        c1 = D1

    ck = c1 * delta_k

    if order >= 2:
        # φ₁ second derivatives → 2LPT source (z-independent)
        def _d1(ki, kj):
            return _irfftn(-ki * kj * ik2 * delta_k, (N, N, N), name, xp)

        dxx, dyy, dzz = _d1(kx, kx), _d1(ky, ky), _d1(kz, kz)
        dxy, dxz, dyz = _d1(kx, ky), _d1(kx, kz), _d1(ky, kz)
        source2 = (dxx * dyy - dxy ** 2
                   + dxx * dzz - dxz ** 2
                   + dyy * dzz - dyz ** 2)
        D2 = -3.0 / 7.0 * D1 ** 2
        c2 = base * 2.0 * D2 if velocity else D2
        ck = ck + c2 * _rfftn(source2, name, xp)

    if order >= 3:
        raise NotImplementedError(
            "The differentiable path supports order 1 (Zel'dovich) and 2 "
            "(2LPT); use ThirdOrderLPT for 3LPT fields (numpy/GPU, "
            "non-differentiable).")

    return ik2 * ck, kx, ky, kz


def lpt_displacement(delta_k, L, z, Om, backend='numpy', order=1, n_nodes=512):
    """LPT displacement (Ψx, Ψy, Ψz) at redshift z — differentiable, device-resident.

    Counterpart of :meth:`LPTBase.get_displacement` for orders 1 and 2.
    Gradients flow to the cosmology through ``delta_k`` (see :func:`lpt_ics`)
    and through the growth factor w.r.t. ``Om``.

    Args:
        delta_k: δ(k) at z = 0, shape (N, N, N//2+1) — output of
                 :func:`lpt_ics` (or a class ``delta_k`` moved to the device).
        L:       Box side length in Mpc/h.
        z:       Redshift (static float).
        Om:      Matter density parameter (scalar or 0-dim tensor).
        backend: ``'numpy'`` (default), ``'jax'`` or ``'torch'``.
        order:   1 → Zel'dovich, 2 → 2LPT.

    Returns:
        (psi_x, psi_y, psi_z) — backend arrays of shape (N, N, N), Mpc/h.
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, delta_k, Om)
    delta_k = _as_complex(delta_k, name, xp, device)
    N = delta_k.shape[0]

    ck, kx, ky, kz = _combined_source_k(delta_k, N, L, z, Om, order,
                                        name, xp, device, n_nodes)
    return (_irfftn(1j * kx * ck, (N, N, N), name, xp),
            _irfftn(1j * ky * ck, (N, N, N), name, xp),
            _irfftn(1j * kz * ck, (N, N, N), name, xp))


def lpt_velocity(delta_k, L, z, Om, backend='numpy', order=1, n_nodes=512):
    """Peculiar velocity (vx, vy, vz) at redshift z in km/s — differentiable.

    Counterpart of :meth:`LPTBase.get_velocity` for orders 1 and 2, with the
    same EdS approximation fₙ ≈ n·f₁.  f₁ comes from
    :func:`beorn.cosmo.differentiable.growth_rate`: exact autodiff of the
    growth integral under jax (nested autodiff — grads w.r.t. Om flow through
    f₁); under torch f₁ is evaluated with a detached graph and enters as a
    constant w.r.t. Om.

    Args / returns: as :func:`lpt_displacement`, units km/s.
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, delta_k, Om)
    delta_k = _as_complex(delta_k, name, xp, device)
    N = delta_k.shape[0]

    ck, kx, ky, kz = _combined_source_k(delta_k, N, L, z, Om, order,
                                        name, xp, device, n_nodes,
                                        velocity=True)
    return (_irfftn(1j * kx * ck, (N, N, N), name, xp),
            _irfftn(1j * ky * ck, (N, N, N), name, xp),
            _irfftn(1j * kz * ck, (N, N, N), name, xp))


# ──────────────────────────────────────────────────────────────────────────────
# Density fields
# ──────────────────────────────────────────────────────────────────────────────

def lpt_linear_density(delta_k, L, z, Om, backend='numpy', R_tophat=None,
                       n_nodes=512):
    """Linear overdensity δ(x) = IRFFT[D₁(z) W(kR) δ(k)] — differentiable.

    Counterpart of :meth:`LPTBase.get_linear_density` (the CHMF conditioning
    field): a clean Gaussian field, optionally top-hat smoothed so that
    Var[δ] = σ²(R_tophat, z).

    Args:
        delta_k:  δ(k) at z = 0, shape (N, N, N//2+1).
        L:        Box side length in Mpc/h.
        z:        Redshift (static float).
        Om:       Matter density parameter (scalar or 0-dim tensor).
        backend:  ``'numpy'`` (default), ``'jax'`` or ``'torch'``.
        R_tophat: Optional top-hat window radius in Mpc/h.

    Returns:
        delta — real backend array of shape (N, N, N).
    """
    name, xp = get_backend(backend)
    device = device_of(name, xp, delta_k, Om)
    delta_k = _as_complex(delta_k, name, xp, device)
    N = delta_k.shape[0]

    a = 1.0 / (1.0 + z)
    D1 = _scalar(growth_factor(a, Om, backend=name, n_nodes=n_nodes),
                 name, xp, device)
    dkz = D1 * delta_k

    if R_tophat is not None:
        kx, ky, kz = _k_grids(N, L, name, xp, device)
        k2 = kx ** 2 + ky ** 2 + kz ** 2
        kR = xp.sqrt(xp.where(k2 > 0, k2, xp.ones_like(k2))) * R_tophat
        W = xp.where(
            kR < 1e-3,
            1.0 - kR ** 2 / 10.0 + kR ** 4 / 280.0,
            3.0 * (xp.sin(kR) - kR * xp.cos(kR)) / kR ** 3,
        )
        W = xp.where(k2 > 0, W, xp.ones_like(W))  # preserve the DC mode
        dkz = dkz * W

    return _irfftn(dkz, (N, N, N), name, xp)


def lpt_density(delta_k, L, z, Om, backend='numpy', order=1,
                mass_assignment='CIC', n_nodes=512):
    """Matter overdensity δ(x) via displacement + differentiable painting.

    Counterpart of :meth:`LPTBase.get_density`: displaces a uniform particle
    grid by Ψ(z) and paints with :func:`beorn.particle_mapping.paint_mesh`
    (functional, device-resident — G4), so gradients flow from the painted
    field back through the particle positions to the cosmology.

    Args: as :func:`lpt_displacement`, plus ``mass_assignment`` (default CIC).
    Paints with ``deconvolve=False`` (unlike :func:`~beorn.particle_mapping.paint_mesh`'s
    own default) to match :meth:`LPTBase.get_density`'s default of the same name.

    Returns:
        delta — backend array of shape (N, N, N), mean-zero overdensity.
    """
    from ..particle_mapping import paint_mesh

    name, xp = get_backend(backend)
    device = device_of(name, xp, delta_k, Om)
    psi_x, psi_y, psi_z = lpt_displacement(delta_k, L, z, Om, backend=backend,
                                           order=order, n_nodes=n_nodes)
    N = psi_x.shape[0]
    cell = float(L) / N
    q1d = (np.arange(N) + 0.5) * cell
    qx = as_const(q1d[:, None, None], name, xp, device)
    qy = as_const(q1d[None, :, None], name, xp, device)
    qz = as_const(q1d[None, None, :], name, xp, device)

    x = (qx + psi_x) % L
    y = (qy + psi_y) % L
    z_pos = (qz + psi_z) % L

    if name == 'torch':
        positions = xp.stack(
            [x.reshape(-1), y.reshape(-1), z_pos.reshape(-1)], dim=-1)
    else:
        positions = xp.stack(
            [x.reshape(-1), y.reshape(-1), z_pos.reshape(-1)], axis=-1)

    mesh = paint_mesh(positions, None, N, L,
                     mass_assignment=mass_assignment, backend=name,
                     deconvolve=False)
    return mesh / mesh.mean() - 1.0


def eulerian_field_diff(lagrangian_field, delta_k, L, z, Om, backend='numpy',
                        order=1, mass_assignment='CIC', n_nodes=512):
    """Repaint a field defined on the regular Lagrangian grid (e.g.
    :func:`~beorn.lpt.chmf.halo_field_diff`'s own output) onto Eulerian
    positions via LPT displacement — the differentiable-tier counterpart of
    the exact tier's gather-kernel position correction (issue: Eulerian
    halo positions).

    No new numerical kernel is needed here (unlike the exact tier's
    :func:`~beorn.particle_mapping.interpolate_field_at_positions`):
    ``halo_field_diff``/``excursion_set_field_diff`` already produce fields
    on the regular grid, not scattered discrete positions, so this reuses
    the exact displaced-position/painting pattern already proven in
    :func:`lpt_density` — just swapping ``lpt_density``'s "weights = matter
    mass" for "weights = the Lagrangian field's own per-cell values".

    Args:
        lagrangian_field: Backend array, shape ``(N, N, N)`` — a field
            defined on the regular Lagrangian grid (e.g.
            :func:`~beorn.lpt.chmf.halo_field_diff`'s ``field`` return).
        delta_k: δ(k) at z = 0, shape ``(N, N, N//2+1)`` — same input
            :func:`lpt_displacement` takes (must be the SAME realization
            ``lagrangian_field`` was itself conditioned on, for the
            displacement to correlate with what moved).
        L:       Box side length in Mpc/h.
        z:       Redshift (static float).
        Om:      Matter density parameter (scalar or 0-dim tensor).
        backend: ``'numpy'`` (default), ``'jax'`` or ``'torch'``.
        order:   1 → Zel'dovich, 2 → 2LPT (matches
                 :meth:`~beorn.lpt.LPTBase`'s own displacement order
                 convention — pass ``order=2`` for consistency with the
                 exact tier's default 2LPT solver).
        mass_assignment: ``'NGP'``, ``'CIC'`` (default), ``'TSC'`` or
                 ``'PCS'`` — forwarded to
                 :func:`~beorn.particle_mapping.paint_mesh`.

    Returns:
        Backend array, shape ``(N, N, N)`` — ``lagrangian_field``'s total
        (``sum w``, not normalised) repainted at Eulerian positions.
    """
    from ..particle_mapping import paint_mesh
    from ..cosmo.differentiable import get_backend, device_of, as_const

    name, xp = get_backend(backend)
    device = device_of(name, xp, delta_k, Om)
    psi_x, psi_y, psi_z = lpt_displacement(delta_k, L, z, Om, backend=backend,
                                           order=order, n_nodes=n_nodes)
    N = psi_x.shape[0]
    cell = float(L) / N
    q1d = (np.arange(N) + 0.5) * cell
    qx = as_const(q1d[:, None, None], name, xp, device)
    qy = as_const(q1d[None, :, None], name, xp, device)
    qz = as_const(q1d[None, None, :], name, xp, device)

    x = (qx + psi_x) % L
    y = (qy + psi_y) % L
    z_pos = (qz + psi_z) % L

    if name == 'torch':
        positions = xp.stack(
            [x.reshape(-1), y.reshape(-1), z_pos.reshape(-1)], dim=-1)
    else:
        positions = xp.stack(
            [x.reshape(-1), y.reshape(-1), z_pos.reshape(-1)], axis=-1)

    weights = lagrangian_field.reshape(-1)
    return paint_mesh(positions, weights, N, L, mass_assignment=mass_assignment,
                     backend=name, deconvolve=False)
