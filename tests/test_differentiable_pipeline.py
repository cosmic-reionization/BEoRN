"""Phase 1 (issue #42) differentiable-pipeline tests: G3/G4/G6 + exit test.

The exit test differentiates the power spectrum *measured from the painted
LPT density field* w.r.t. cosmological parameters, in jax and torch, and
compares against central finite differences — the acceptance criterion for
Phase 1 of the GPU/differentiability audit.

Pure-vs-class parity tolerances are set by the documented ~1e-4 relative
difference between the classes' legacy growth/normalisation quadratures and
the fixed-node differentiable ones — not by float precision.
"""
import warnings

import numpy as np
import pytest

from beorn.structs import Parameters
from beorn.lpt import (
    ZeldovichApproximation, SecondOrderLPT,
    lpt_ics, lpt_displacement, lpt_velocity, lpt_linear_density, lpt_density,
    CHMF, CHMFSampler, halo_field_diff,
)
from beorn.particle_mapping import paint_mesh

jax = pytest.importorskip('jax', reason='differentiable pipeline tests need jax')
import jax.numpy as jnp  # noqa: E402

jax.config.update('jax_enable_x64', True)

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    import tools21cm as t2c
    import inspect
    _T2C_BACKEND = 'backend' in str(
        inspect.signature(t2c.power_spectrum.apply_window))
except Exception:
    _T2C_BACKEND = False

N, L, Z = 16, 100.0, 7.0
SEED = 42


@pytest.fixture(scope='module')
def param():
    p = Parameters()
    p.simulation.Ncell = N
    p.simulation.Lbox = L
    p.simulation.use_hunits = True  # L is a plain Mpc/h constant shared with the "pure" reference functions below (issue #49)
    p.source.halo_mass_min = 1e9
    return p


@pytest.fixture(scope='module')
def theta(param):
    c = param.cosmology
    return dict(Om=c.Om, Ob=c.Ob, h0=c.h0, ns=c.ns, sigma_8=c.sigma_8)


@pytest.fixture(scope='module')
def noise():
    return np.random.default_rng(SEED).standard_normal((N,) * 3)


# ─────────────────────────────────────────────────────────────────────────────
# G3 — pure functions match the classes (numpy backend)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('cls,order', [(ZeldovichApproximation, 1),
                                       (SecondOrderLPT, 2)])
def test_pure_matches_class(param, theta, cls, order):
    solver = cls(param, verbose=False, seed=SEED)
    solver.generate_initial_conditions()
    noise_cls = solver._backend.random_normal((N,) * 3, seed=SEED)

    dk = lpt_ics(noise_cls, L, **theta)
    nz = np.abs(solver.delta_k) > 0
    assert np.allclose(dk[nz], solver.delta_k[nz], rtol=1e-3)

    for pure, ref in [
        (lpt_displacement(solver.delta_k, L, Z, theta['Om'], order=order),
         solver.get_displacement(Z)),
        (lpt_velocity(solver.delta_k, L, Z, theta['Om'], order=order),
         solver.get_velocity(Z)),
    ]:
        for p_c, r_c in zip(pure, ref):
            scale = np.abs(r_c).max()
            assert np.max(np.abs(p_c - r_c)) < 1e-3 * scale

    dl = lpt_linear_density(solver.delta_k, L, Z, theta['Om'], R_tophat=2.0)
    ref = solver.get_linear_density(Z, R_tophat=2.0)
    assert np.max(np.abs(dl - ref)) < 1e-3 * np.abs(ref).max()

    de = lpt_density(solver.delta_k, L, Z, theta['Om'], order=order)
    ref = solver.get_density(Z)
    assert np.max(np.abs(de - ref)) < 1e-3 * (1.0 + np.abs(ref).max())


def test_class_linear_density_backend_path(param):
    """The device path of get_linear_density equals the legacy numpy path."""
    grf = np.random.default_rng(11).standard_normal((N,) * 3)
    cell = L / N
    R = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * cell

    za_n = ZeldovichApproximation(param, verbose=False, seed=SEED)
    za_j = ZeldovichApproximation(param, verbose=False, seed=SEED,
                                  backend='jax')
    za_n.generate_initial_conditions(grf=grf)
    za_j.generate_initial_conditions(grf=grf)

    ref = za_n.get_linear_density(Z, R_tophat=R)
    dev = za_j.get_linear_density(Z, R_tophat=R)
    assert dev.dtype == np.float32
    np.testing.assert_allclose(dev, ref, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# G4 — functional paint contract
# ─────────────────────────────────────────────────────────────────────────────

def test_paint_mesh_backends_agree():
    pos = (np.random.default_rng(2).random((500, 3)) * L).astype(np.float32)
    w = np.random.default_rng(3).random(500).astype(np.float32)

    ref = paint_mesh(pos, w, N, L, backend='numpy')
    assert isinstance(ref, np.ndarray)
    assert ref.sum() == pytest.approx(w.sum(), rel=1e-5)

    mj = paint_mesh(jnp.asarray(pos), jnp.asarray(w), N, L)  # auto → jax
    assert not isinstance(mj, np.ndarray)
    np.testing.assert_allclose(np.asarray(mj), ref, atol=1e-4)

    if _TORCH:
        mt = paint_mesh(torch.as_tensor(pos), torch.as_tensor(w), N, L)
        assert isinstance(mt, torch.Tensor)
        np.testing.assert_allclose(mt.numpy(), ref, atol=1e-4)


@pytest.mark.skipif(not _TORCH, reason='torch not installed')
def test_paint_mesh_torch_gradients():
    pos = torch.rand(200, 3, dtype=torch.float64) * L
    pos.requires_grad_(True)
    mesh = paint_mesh(pos, None, N, L)
    (mesh ** 2).sum().backward()
    assert pos.grad is not None and torch.isfinite(pos.grad).all()


# ─────────────────────────────────────────────────────────────────────────────
# G6 — expected-number painting + reparameterised shot noise
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def chmf_setup(param):
    cell = L / N
    R = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * cell
    za = ZeldovichApproximation(param, verbose=False, seed=7)
    delta = za.get_linear_density(Z, R_tophat=R).astype(np.float64)
    sampler = CHMFSampler(param, chmf=CHMF(param))
    return delta, sampler, sampler.chmf.M_of_R(R), cell ** 3


def test_halo_field_diff_expectation(param, theta, chmf_setup):
    delta, sampler, M_env, V_cell = chmf_setup
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Mc_ref, lam = sampler.expected_counts(delta, Z, n_mass_bins=20)

    field, Mc = halo_field_diff(delta, M_env, Z, **theta,
                                cell_volume=V_cell, M_min=1e9,
                                n_mass_bins=20, weights='counts')
    np.testing.assert_allclose(Mc, Mc_ref, rtol=1e-10)
    # tolerance: interp-table σ² (class) vs direct-integral σ² (diff path)
    assert float(np.sum(field)) == pytest.approx(lam.sum(), rel=2e-3)

    # eps = 0 reduces exactly to the expectation
    eps0 = np.zeros((20,) + delta.shape)
    f0, _ = halo_field_diff(delta, M_env, Z, **theta,
                            cell_volume=V_cell, M_min=1e9,
                            n_mass_bins=20, weights='counts', eps=eps0)
    np.testing.assert_allclose(np.asarray(f0), np.asarray(field))


def test_halo_field_diff_stochastic_gradient(param, theta, chmf_setup):
    delta, _, M_env, V_cell = chmf_setup
    eps = np.random.default_rng(3).standard_normal((20,) + delta.shape)

    def total_mass(s8):
        f, _ = halo_field_diff(jnp.asarray(delta), M_env, Z,
                               theta['Om'], theta['Ob'], theta['h0'],
                               theta['ns'], s8, cell_volume=V_cell,
                               M_min=1e9, n_mass_bins=20, weights='mass',
                               eps=eps, backend='jax')
        return f.sum()

    g = jax.grad(total_mass)(theta['sigma_8'])
    h = 1e-4
    fd = (total_mass(theta['sigma_8'] + h)
          - total_mass(theta['sigma_8'] - h)) / (2 * h)
    assert float(g) == pytest.approx(float(fd), rel=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 exit test — grad of the measured P(k) w.r.t. cosmology
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _T2C_BACKEND,
                    reason='tools21cm without differentiable backend '
                           '(needs tools21cm > 2.4.2 / PR #81)')
@pytest.mark.parametrize('pname', ['sigma_8', 'ns', 'Om', 'h0'])
def test_exit_grad_jax(theta, noise, pname):
    """d P̂(k) / dθ on the painted ZA density field, jax vs central FD."""
    noise_j = jnp.asarray(noise)

    def S(val):
        th = dict(theta)
        th[pname] = val
        dk = lpt_ics(noise_j, L, th['Om'], th['Ob'], th['h0'], th['ns'],
                     th['sigma_8'], backend='jax')
        d = lpt_density(dk, L, Z, th['Om'], backend='jax')
        ps, _ = t2c.power_spectrum_1d(d, kbins=4, box_dims=L, backend='jax')
        return ps.sum()

    x0 = theta[pname]
    g = jax.grad(S)(x0)
    h = 1e-4 * abs(x0)
    fd = (S(x0 + h) - S(x0 - h)) / (2 * h)
    assert float(g) == pytest.approx(float(fd), rel=1e-4)


@pytest.mark.skipif(not (_T2C_BACKEND and _TORCH),
                    reason='needs torch + tools21cm differentiable backend')
def test_exit_grad_torch_matches_jax(theta, noise):
    """Torch autograd of the same pipeline agrees with jax."""
    noise_j = jnp.asarray(noise)

    def S_jax(s8):
        dk = lpt_ics(noise_j, L, theta['Om'], theta['Ob'], theta['h0'],
                     theta['ns'], s8, backend='jax')
        d = lpt_density(dk, L, Z, theta['Om'], backend='jax')
        return t2c.power_spectrum_1d(d, kbins=4, box_dims=L,
                                     backend='jax')[0].sum()

    g_jax = float(jax.grad(S_jax)(theta['sigma_8']))

    s8 = torch.tensor(theta['sigma_8'], dtype=torch.float64,
                      requires_grad=True)
    dk = lpt_ics(torch.as_tensor(noise), L, theta['Om'], theta['Ob'],
                 theta['h0'], theta['ns'], s8, backend='torch')
    d = lpt_density(dk, L, Z, theta['Om'], backend='torch')
    ps, _ = t2c.power_spectrum_1d(d, kbins=4, box_dims=L, backend='torch')
    ps.sum().backward()

    assert float(s8.grad) == pytest.approx(g_jax, rel=1e-10)
