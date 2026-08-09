"""Unit tests for the differentiable excursion-set surrogate
(:func:`beorn.lpt.excursion_set_field_diff`) -- the smooth counterpart of
:class:`beorn.lpt.ExcursionSetFinder`, mirroring
``spreading_method='exact'|'diffusion'``'s own exact-vs-surrogate test split.
"""
import numpy as np
import pytest

from beorn.structs import Parameters
from beorn.lpt import ZeldovichApproximation, CHMF, ExcursionSetFinder, excursion_set_field_diff

jax = pytest.importorskip('jax', reason='differentiable excursion-set tests need jax')
import jax.numpy as jnp  # noqa: E402

jax.config.update('jax_enable_x64', True)

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

N, L = 32, 200.0
Z_LOW = 0.5


@pytest.fixture(scope='module')
def params():
    p = Parameters()
    p.simulation.Ncell = N
    p.simulation.Lbox = L
    p.simulation.use_hunits = True
    return p


@pytest.fixture(scope='module')
def theta(params):
    c = params.cosmology
    return dict(Om=c.Om, Ob=c.Ob, h0=c.h0, ns=c.ns, sigma_8=c.sigma_8)


@pytest.fixture(scope='module')
def chmf(params):
    return CHMF(params)


@pytest.fixture(scope='module')
def delta(params):
    za = ZeldovichApproximation(params, verbose=False, seed=13)
    return za.get_linear_density(Z_LOW).astype(np.float64)


@pytest.fixture(scope='module')
def M_env(params, chmf):
    return chmf.rho_m * (params.Lbox_hunits / N) ** 3


# ── Basics ────────────────────────────────────────────────────────────────

def test_field_shape_and_nonnegative(params, theta, chmf, delta, M_env):
    field = excursion_set_field_diff(
        delta, params.Lbox_hunits, M_env, Z_LOW, **theta,
        n_scales=12, T=0.1, backend='numpy',
    )
    assert field.shape == delta.shape
    assert np.all(np.asarray(field) >= 0.0)


def test_field_rejects_m_max_below_m_split(params, theta, M_env):
    with pytest.raises(ValueError, match="must exceed"):
        excursion_set_field_diff(
            np.zeros((N, N, N)), params.Lbox_hunits, M_env, Z_LOW, **theta,
            M_max=M_env / 2.0, backend='numpy',
        )


def test_field_zero_for_empty_field(params, theta, M_env):
    """A field with no overdensity anywhere is far below any barrier at
    every scale, so the soft-crossing weight -- and hence the deterministic
    mass field -- should be negligible everywhere."""
    field = excursion_set_field_diff(
        np.zeros((N, N, N)), params.Lbox_hunits, M_env, Z_LOW, **theta,
        n_scales=12, T=0.05, backend='numpy',
    )
    total_frac = float(field.sum()) / (Om_rho_m(theta, params) * params.Lbox_hunits ** 3)
    assert total_frac < 1e-6


def Om_rho_m(theta, params):
    from beorn.constants import rhoc0
    return theta['Om'] * rhoc0 / theta['h0']


# ── Softness-limit convergence vs. analytic Press-Schechter ────────────────

def test_softness_limit_order_of_magnitude_matches_analytic_ps(params, theta, chmf, delta, M_env):
    """As T shrinks toward a sharp crossing, the total deterministically-
    collapsed mass fraction should land within an order of magnitude of the
    analytic PS collapsed-mass fraction above M_split -- the same loose,
    single-realization sanity bound used by the exact tier's own
    completeness test (large sample variance for the rare massive tail)."""
    field = excursion_set_field_diff(
        delta, params.Lbox_hunits, M_env, Z_LOW, **theta,
        n_scales=24, T=0.02, backend='numpy',
    )
    measured_frac = float(field.sum()) / (chmf.rho_m * params.Lbox_hunits ** 3)

    M_test = np.logspace(np.log10(M_env), np.log10(M_env * 50), 500)
    dndlnm = chmf.hmf_ps(M_test, Z_LOW)
    analytic_frac = np.trapezoid(dndlnm * M_test, np.log(M_test)) / chmf.rho_m

    assert measured_frac > 0
    assert 0.1 < measured_frac / analytic_frac < 10.0


# ── Cross-method conservation: 'exact' vs 'soft' ────────────────────────────

def test_exact_and_soft_totals_are_within_an_order_of_magnitude(params, theta, chmf, delta, M_env):
    """Same field, same M_split: the exact and soft tiers are independent
    algorithms (hard connected-component merge vs. a continuous per-cell
    survival-product relaxation with no merge step -- see module
    docstring), so they are not expected to agree closely, but both are
    estimating the same physical collapsed-mass fraction and should not
    differ by orders of magnitude."""
    finder = ExcursionSetFinder(chmf)
    cat, _ = finder.find(delta, Z_LOW, M_split=M_env, n_scales=24)
    exact_frac = cat.masses.sum() / (chmf.rho_m * params.Lbox_hunits ** 3)

    field = excursion_set_field_diff(
        delta, params.Lbox_hunits, M_env, Z_LOW, **theta,
        n_scales=24, T=0.05, backend='numpy',
    )
    soft_frac = float(field.sum()) / (chmf.rho_m * params.Lbox_hunits ** 3)

    assert exact_frac > 0 and soft_frac > 0
    assert 0.1 < soft_frac / exact_frac < 10.0


# ── Differentiability ───────────────────────────────────────────────────────

def test_jax_grad_matches_central_fd(params, theta, M_env):
    """jax.grad w.r.t. sigma_8 through the full multi-scale walk matches
    central finite differences.

    Uses ``chmf_recipe='MovingBarrier'`` -- the default ``'BarkanaLoeb2004'``
    barrier is the constant ``delta_c`` (see :meth:`CHMF.barrier`), entirely
    independent of ``sigma_8``, so its gradient is correctly (not buggily)
    zero; ``'MovingBarrier'``'s barrier depends on ``sigma_8`` through
    ``sigma2_M``, giving a genuine nonzero-gradient test case."""
    delta_small = np.random.default_rng(1).standard_normal((8, 8, 8)) * 0.3

    def total_mass(s8):
        field = excursion_set_field_diff(
            jnp.asarray(delta_small), params.Lbox_hunits, M_env, Z_LOW,
            theta['Om'], theta['Ob'], theta['h0'], theta['ns'], s8,
            n_scales=8, T=0.1, backend='jax', chmf_recipe='MovingBarrier',
        )
        return field.sum()

    g = jax.grad(total_mass)(theta['sigma_8'])
    h = 1e-4
    fd = (total_mass(theta['sigma_8'] + h) - total_mass(theta['sigma_8'] - h)) / (2 * h)
    assert float(g) == pytest.approx(float(fd), rel=1e-3, abs=1e-6)


@pytest.mark.skipif(not _TORCH, reason='torch not installed')
def test_torch_grad_matches_central_fd(params, theta, M_env):
    """torch autograd (.backward()) w.r.t. sigma_8 through the full
    multi-scale walk matches central finite differences, and gradients also
    flow through the conditioning field ``delta`` itself.

    Regression test for a real bug: the MovingBarrier recipe's barrier
    formula computed ``xp.sqrt(a_mb)`` on ``a_mb`` (a plain Python float,
    the fixed Jenkins et al. 2001 constant) -- ``torch.sqrt`` requires a
    Tensor argument (unlike ``jnp.sqrt``/``np.sqrt``, which auto-convert),
    so this raised ``TypeError`` for every torch + ``chmf_recipe=
    'MovingBarrier'`` call. Fixed by precomputing ``math.sqrt(a_mb)`` as a
    plain float (the same fix applied to
    :func:`beorn.lpt.chmf.conditional_dndlnm_diff`, which has the identical
    pre-existing bug -- see ``test_differentiable_pipeline.py``'s own
    torch regression test)."""
    delta_small = np.random.default_rng(1).standard_normal((8, 8, 8)) * 0.3

    def total_mass(s8, delta):
        field = excursion_set_field_diff(
            delta, params.Lbox_hunits, M_env, Z_LOW,
            theta['Om'], theta['Ob'], theta['h0'], theta['ns'], s8,
            n_scales=8, T=0.1, backend='torch', chmf_recipe='MovingBarrier',
        )
        return field.sum()

    delta_t = torch.tensor(delta_small, dtype=torch.float64)
    s8 = torch.tensor(theta['sigma_8'], dtype=torch.float64, requires_grad=True)
    out = total_mass(s8, delta_t)
    out.backward()
    g = s8.grad.item()

    h = 1e-4
    with torch.no_grad():
        fp = total_mass(torch.tensor(theta['sigma_8'] + h, dtype=torch.float64), delta_t).item()
        fm = total_mass(torch.tensor(theta['sigma_8'] - h, dtype=torch.float64), delta_t).item()
    fd = (fp - fm) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-3)

    delta_grad_t = torch.tensor(delta_small, dtype=torch.float64, requires_grad=True)
    total_mass(s8.detach(), delta_grad_t).backward()
    assert torch.any(delta_grad_t.grad != 0.0)


def test_backend_agreement_movingbarrier_numpy_jax_torch(params, theta, M_env):
    """Same regression as above, cross-checked as backend agreement
    (numpy/jax never hit the torch-only bug, so this pins the MovingBarrier
    branch -- not just the default BarkanaLoeb2004 branch the other
    backend-agreement tests exercise -- to also agree across backends)."""
    delta_small = np.random.default_rng(4).standard_normal((8, 8, 8)) * 0.3
    kwargs = dict(Lbox=params.Lbox_hunits, M_split=M_env, z=Z_LOW, **theta,
                  n_scales=8, T=0.1, chmf_recipe='MovingBarrier')

    field_np = excursion_set_field_diff(delta_small, backend='numpy', **kwargs)
    field_jax = excursion_set_field_diff(jnp.asarray(delta_small), backend='jax', **kwargs)
    np.testing.assert_allclose(np.asarray(field_np), np.asarray(field_jax), rtol=1e-6)

    if _TORCH:
        field_torch = excursion_set_field_diff(
            torch.as_tensor(delta_small, dtype=torch.float64), backend='torch', **kwargs)
        np.testing.assert_allclose(np.asarray(field_np), field_torch.numpy(), rtol=1e-6)


def test_backend_agreement_numpy_jax(params, theta, M_env):
    delta_small = np.random.default_rng(2).standard_normal((8, 8, 8)) * 0.3

    field_np = excursion_set_field_diff(
        delta_small, params.Lbox_hunits, M_env, Z_LOW, **theta,
        n_scales=8, T=0.1, backend='numpy',
    )
    field_jax = excursion_set_field_diff(
        jnp.asarray(delta_small), params.Lbox_hunits, M_env, Z_LOW, **theta,
        n_scales=8, T=0.1, backend='jax',
    )
    np.testing.assert_allclose(np.asarray(field_np), np.asarray(field_jax), rtol=1e-6)


@pytest.mark.skipif(not _TORCH, reason='torch not installed')
def test_backend_agreement_numpy_torch(params, theta, M_env):
    delta_small = np.random.default_rng(2).standard_normal((8, 8, 8)) * 0.3

    field_np = excursion_set_field_diff(
        delta_small, params.Lbox_hunits, M_env, Z_LOW, **theta,
        n_scales=8, T=0.1, backend='numpy',
    )
    field_torch = excursion_set_field_diff(
        torch.as_tensor(delta_small, dtype=torch.float64), params.Lbox_hunits,
        M_env, Z_LOW, **theta, n_scales=8, T=0.1, backend='torch',
    )
    np.testing.assert_allclose(np.asarray(field_np), field_torch.numpy(), rtol=1e-6)


def test_rejects_unknown_chmf_recipe(params, theta, M_env):
    with pytest.raises(ValueError, match="Unknown chmf_recipe"):
        excursion_set_field_diff(
            np.zeros((N, N, N)), params.Lbox_hunits, M_env, Z_LOW, **theta,
            n_scales=4, chmf_recipe='bogus', backend='numpy',
        )
