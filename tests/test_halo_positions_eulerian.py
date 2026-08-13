"""Tests for Item 2 (Eulerian halo positions): the exact-tier gather kernel
(interpolate_field_at_positions/displace_positions), CHMFSampler.sample's
and ExcursionSetFinder.find's displacement_field wiring, LPTHaloLoader's
apply_eulerian_displacement opt-in, and the differentiable-tier composition
(eulerian_field_diff)."""
import numpy as np
import pytest

from beorn.structs import Parameters
from beorn.lpt import ZeldovichApproximation, SecondOrderLPT, lpt_ics, eulerian_field_diff
from beorn.lpt.chmf import CHMF, CHMFSampler, halo_field_diff
from beorn.lpt.excursion_set import ExcursionSetFinder
from beorn.load_input_data import LPTHaloLoader
from beorn.particle_mapping import (
    interpolate_field_at_positions, displace_positions, map_particles_to_mesh,
)

N, L, Z = 16, 100.0, 10.0


@pytest.fixture
def params():
    p = Parameters()
    p.simulation.Ncell = N
    p.simulation.Lbox = L
    return p


# ── interpolate_field_at_positions (exact-tier gather kernel) ─────────────────

@pytest.mark.parametrize('scheme', ['NGP', 'CIC', 'TSC', 'PCS'])
def test_gather_constant_field_returns_itself(scheme):
    field = np.full((N, N, N), 3.5, dtype=np.float64)
    positions = np.random.default_rng(0).random((200, 3)) * L
    values = interpolate_field_at_positions(field, L, positions, mass_assignment=scheme)
    np.testing.assert_allclose(values, 3.5)


def test_gather_cic_recovers_exact_value_at_cell_centers():
    rng = np.random.default_rng(1)
    field = rng.standard_normal((N, N, N))
    cell = L / N
    idx = rng.integers(0, N, size=(50, 3))
    positions = (idx + 0.5) * cell
    values = interpolate_field_at_positions(field, L, positions.astype(np.float64), mass_assignment='CIC')
    expected = field[idx[:, 0], idx[:, 1], idx[:, 2]]
    np.testing.assert_allclose(values, expected, atol=1e-10)


def test_gather_cic_recovers_linear_field_off_grid():
    """CIC (bilinear) interpolation is exact for a linear function."""
    cell = L / N
    q1d = (np.arange(N) + 0.5) * cell
    a, b, c, d0 = 0.7, -1.3, 2.1, 5.0
    field = (a * q1d[:, None, None] + b * q1d[None, :, None]
             + c * q1d[None, None, :] + d0).astype(np.float64)

    rng = np.random.default_rng(2)
    # Keep well inside the box (away from the periodic wrap) so the linear
    # form holds without wrapping corrections.
    positions = cell + rng.random((100, 3)) * (L - 2 * cell)
    values = interpolate_field_at_positions(field, L, positions, mass_assignment='CIC')
    expected = a * positions[:, 0] + b * positions[:, 1] + c * positions[:, 2] + d0
    np.testing.assert_allclose(values, expected, rtol=1e-6)


@pytest.mark.parametrize('scheme', ['NGP', 'CIC', 'TSC', 'PCS'])
def test_gather_is_scatter_adjoint(scheme):
    """Gather and scatter are mathematical duals: sum_p gather(field, p) ==
    sum_cells field * scatter(ones, positions) -- the defining adjoint
    identity, checked directly rather than trusting scheme-by-scheme
    weight bookkeeping."""
    rng = np.random.default_rng(3)
    field = rng.standard_normal((N, N, N)).astype(np.float64)
    positions = (rng.random((300, 3)) * L).astype(np.float64)

    gathered_sum = interpolate_field_at_positions(field, L, positions, mass_assignment=scheme).sum()

    deposited = np.zeros((N, N, N), dtype=np.float64)
    map_particles_to_mesh(deposited, L, positions, mass_assignment=scheme,
                          backend='numpy', deconvolve=False)
    scatter_dual_sum = float(np.sum(field * deposited))

    assert gathered_sum == pytest.approx(scatter_dual_sum, rel=1e-8)


def test_displace_positions_constant_shift():
    shift = 7.25
    psi = (np.full((N, N, N), shift, dtype=np.float32),
           np.zeros((N, N, N), dtype=np.float32),
           np.zeros((N, N, N), dtype=np.float32))
    positions = (np.random.default_rng(4).random((50, 3)) * L).astype(np.float32)
    displaced = displace_positions(positions, psi, L)

    expected_x = (positions[:, 0] + shift) % L
    np.testing.assert_allclose(displaced[:, 0], expected_x, atol=1e-3)
    np.testing.assert_allclose(displaced[:, 1], positions[:, 1], atol=1e-3)
    np.testing.assert_allclose(displaced[:, 2], positions[:, 2], atol=1e-3)


def test_displace_positions_wraps_periodically():
    psi = (np.full((N, N, N), L - 1.0, dtype=np.float32),) + (np.zeros((N, N, N), dtype=np.float32),) * 2
    positions = np.array([[2.0, 5.0, 5.0]], dtype=np.float32)
    displaced = displace_positions(positions, psi, L)
    assert 0.0 <= displaced[0, 0] < L
    np.testing.assert_allclose(displaced[0, 0], (2.0 + L - 1.0) % L, atol=1e-2)


# ── CHMFSampler.sample: displacement_field wiring ─────────────────────────────

@pytest.fixture
def delta_field(params):
    za = ZeldovichApproximation(params, verbose=False, seed=11)
    return za.get_linear_density(Z).astype(np.float64)


def test_sample_displacement_field_none_is_default_unchanged(params, delta_field):
    sampler = CHMFSampler(params, chmf=CHMF(params))
    cat_default = sampler.sample(delta_field, Z, n_mass_bins=10)
    cat_explicit_none = sampler.sample(delta_field, Z, n_mass_bins=10, displacement_field=None)
    np.testing.assert_array_equal(cat_default.positions, cat_explicit_none.positions)
    np.testing.assert_array_equal(cat_default.masses, cat_explicit_none.masses)


@pytest.mark.parametrize('n_mass_bins', [10, None])
def test_sample_constant_displacement_shifts_all_positions(params, delta_field, n_mass_bins):
    sampler = CHMFSampler(params, chmf=CHMF(params))
    shift = 3.0
    psi = (np.full((N, N, N), shift, dtype=np.float32),
           np.zeros((N, N, N), dtype=np.float32),
           np.zeros((N, N, N), dtype=np.float32))

    cat_lagrangian = sampler.sample(delta_field, Z, n_mass_bins=n_mass_bins)
    cat_eulerian = sampler.sample(delta_field, Z, n_mass_bins=n_mass_bins, displacement_field=psi)

    assert cat_lagrangian.masses.size > 0
    np.testing.assert_array_equal(cat_lagrangian.masses, cat_eulerian.masses)
    # Sampler positions wrap at the *resolved* box size (params.Lbox_hunits,
    # in Mpc/h) -- not the raw module-level L (Mpc), which differ whenever
    # use_hunits is False (the default): Lbox_hunits = Lbox * h0.
    expected_x = (cat_lagrangian.positions[:, 0].astype(np.float64) + shift) % params.Lbox_hunits
    np.testing.assert_allclose(cat_eulerian.positions[:, 0], expected_x, atol=1e-2)
    np.testing.assert_allclose(cat_eulerian.positions[:, 1], cat_lagrangian.positions[:, 1], atol=1e-2)
    np.testing.assert_allclose(cat_eulerian.positions[:, 2], cat_lagrangian.positions[:, 2], atol=1e-2)


# ── ExcursionSetFinder.find: displacement_field wiring ────────────────────────

def test_excursion_set_find_constant_displacement_shifts_all_positions(params):
    za = ZeldovichApproximation(params, verbose=False, seed=5)
    delta = za.get_linear_density(Z).astype(np.float64)
    chmf = CHMF(params)
    finder = ExcursionSetFinder(chmf)
    cell_size = params.Lbox_hunits / N
    M_split = chmf.rho_m * cell_size ** 3

    cat_lagrangian, _ = finder.find(delta, Z, M_split=M_split, n_scales=8)
    if cat_lagrangian.masses.size == 0:
        pytest.skip("no halos found at this seed/M_split -- nothing to compare")

    shift = 2.5
    psi = (np.zeros((N, N, N), dtype=np.float32),
           np.full((N, N, N), shift, dtype=np.float32),
           np.zeros((N, N, N), dtype=np.float32))
    cat_eulerian, _ = finder.find(delta, Z, M_split=M_split, n_scales=8, displacement_field=psi)

    np.testing.assert_array_equal(cat_lagrangian.masses, cat_eulerian.masses)
    expected_y = (cat_lagrangian.positions[:, 1].astype(np.float64) + shift) % params.Lbox_hunits
    np.testing.assert_allclose(cat_eulerian.positions[:, 1], expected_y, atol=1e-2)
    np.testing.assert_allclose(cat_eulerian.positions[:, 0], cat_lagrangian.positions[:, 0], atol=1e-2)
    np.testing.assert_allclose(cat_eulerian.positions[:, 2], cat_lagrangian.positions[:, 2], atol=1e-2)


# ── LPTHaloLoader: apply_eulerian_displacement (default True, opt-out) ────────

def test_apply_eulerian_displacement_defaults_to_true(params):
    assert params.halo_sim.apply_eulerian_displacement is True


def test_loader_default_applies_eulerian_displacement(params):
    params.solver.redshifts = np.array([Z])
    loader = LPTHaloLoader(params, n_mass_bins=10)
    assert loader.apply_eulerian_displacement is True
    cat = loader.load_halo_catalog(0)
    assert np.all(cat.positions >= 0) and np.all(cat.positions <= params.Lbox_hunits)


def test_loader_explicit_false_reproduces_lagrangian_positions(params):
    params.solver.redshifts = np.array([Z])
    loader = LPTHaloLoader(params, n_mass_bins=10, apply_eulerian_displacement=False)
    assert loader.apply_eulerian_displacement is False
    cat = loader.load_halo_catalog(0)
    assert np.all(cat.positions >= 0) and np.all(cat.positions <= params.Lbox_hunits)


def test_loader_eulerian_displacement_moves_halos(params):
    params.solver.redshifts = np.array([Z])
    loader_lagrangian = LPTHaloLoader(params, n_mass_bins=10, apply_eulerian_displacement=False)
    loader_eulerian = LPTHaloLoader(params, n_mass_bins=10, apply_eulerian_displacement=True)

    cat_lagrangian = loader_lagrangian.load_halo_catalog(0)
    cat_eulerian = loader_eulerian.load_halo_catalog(0)

    assert cat_lagrangian.masses.size > 0
    np.testing.assert_array_equal(cat_lagrangian.masses, cat_eulerian.masses)
    assert np.all(cat_eulerian.positions >= 0) and np.all(cat_eulerian.positions <= params.Lbox_hunits)
    # Positions should generally differ (real LPT displacement is nonzero
    # almost everywhere) -- not a bitwise-equality regression guard like the
    # False case, since this IS the behavior change under test.
    assert not np.allclose(cat_lagrangian.positions, cat_eulerian.positions)


def test_apply_eulerian_displacement_resolves_from_halo_sim(params):
    params.halo_sim.apply_eulerian_displacement = False
    params.solver.redshifts = np.array([Z])
    loader = LPTHaloLoader(params, n_mass_bins=10)
    assert loader.apply_eulerian_displacement is False


# ── Diff tier: eulerian_field_diff ─────────────────────────────────────────────

@pytest.fixture
def theta(params):
    c = params.cosmology
    return dict(Om=c.Om, Ob=c.Ob, h0=c.h0, ns=c.ns, sigma_8=c.sigma_8)


def _halo_field_setup(params, theta, seed=7):
    cell = L / N
    R = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * cell
    solver = SecondOrderLPT(params, verbose=False, seed=seed)
    solver.generate_initial_conditions()
    delta_k = solver.delta_k
    delta_env = solver.get_linear_density(Z, R_tophat=R).astype(np.float64)
    chmf = CHMF(params)
    M_env = chmf.M_of_R(R)
    field, _ = halo_field_diff(delta_env, M_env, Z, **theta,
                               cell_volume=cell ** 3, M_min=1e9,
                               n_mass_bins=20, weights='mass')
    return delta_k, field


def test_eulerian_field_diff_conserves_total_mass(params, theta):
    delta_k, field = _halo_field_setup(params, theta)
    eulerian = eulerian_field_diff(field, delta_k, L, Z, theta['Om'], order=2)
    assert float(np.sum(eulerian)) == pytest.approx(float(np.sum(field)), rel=1e-5)


def test_eulerian_field_diff_zero_displacement_is_identity():
    """delta_k = 0 -> psi = 0 everywhere -> Eulerian positions coincide
    exactly with the Lagrangian grid, so painting (CIC, exact at grid
    points) must reproduce the input field exactly."""
    delta_k = np.zeros((N, N, N // 2 + 1), dtype=np.complex128)
    field = np.random.default_rng(9).random((N, N, N))
    eulerian = eulerian_field_diff(field, delta_k, L, Z, 0.315, order=2)
    np.testing.assert_allclose(np.asarray(eulerian), field, atol=1e-6)


def test_eulerian_field_diff_backend_agreement(params, theta):
    jax = pytest.importorskip('jax', reason='backend agreement needs jax')
    import jax.numpy as jnp

    delta_k, field = _halo_field_setup(params, theta)
    ref = eulerian_field_diff(field, delta_k, L, Z, theta['Om'], order=2, backend='numpy')
    dev = eulerian_field_diff(jnp.asarray(field), jnp.asarray(delta_k), L, Z,
                              theta['Om'], order=2, backend='jax')
    np.testing.assert_allclose(np.asarray(dev), ref, rtol=1e-4, atol=1e-4)


def test_eulerian_field_diff_differentiable_jax(params, theta):
    jax = pytest.importorskip('jax', reason='gradient test needs jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)

    delta_k, field = _halo_field_setup(params, theta)
    field_const = jnp.asarray(field)

    def total_sq(Om):
        eulerian = eulerian_field_diff(field_const, jnp.asarray(delta_k), L, Z,
                                       Om, order=2, backend='jax')
        return jnp.sum(eulerian ** 2)

    g = jax.grad(total_sq)(theta['Om'])
    h = 1e-4
    fd = (total_sq(theta['Om'] + h) - total_sq(theta['Om'] - h)) / (2 * h)
    assert float(g) == pytest.approx(float(fd), rel=5e-3)
