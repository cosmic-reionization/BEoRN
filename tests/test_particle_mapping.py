"""Unit tests for beorn.particle_mapping.core (numpy backend)."""
import numpy as np
import pytest

from beorn.particle_mapping.core import map_particles_to_mesh, paint_displacement_field
from beorn.particle_mapping import coarsen_field


def _mesh(N=16):
    return np.zeros((N, N, N), dtype=np.float32)


def _positions(n, box_size=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3)) * box_size).astype(np.float32)


# ── Error handling ────────────────────────────────────────────────────────────

def test_invalid_backend_raises():
    mesh = _mesh()
    pos = _positions(10)
    with pytest.raises(ValueError, match="Unknown backend"):
        map_particles_to_mesh(mesh, 1.0, pos, backend='does_not_exist')


# ── NGP ───────────────────────────────────────────────────────────────────────

def test_ngp_conserves_particle_count():
    N, n = 16, 1000
    mesh = _mesh(N)
    pos = _positions(n)
    map_particles_to_mesh(mesh, 1.0, pos, mass_assignment='NGP')
    assert mesh.sum() == pytest.approx(n, rel=1e-5)


def test_ngp_single_particle_lands_in_one_cell():
    mesh = _mesh(8)
    pos = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    map_particles_to_mesh(mesh, 1.0, pos, mass_assignment='NGP')
    assert (mesh > 0).sum() == 1
    assert mesh.sum() == pytest.approx(1.0)


# ── CIC ───────────────────────────────────────────────────────────────────────

def test_cic_conserves_total_weight():
    N, n = 16, 500
    mesh = _mesh(N)
    pos = _positions(n)
    map_particles_to_mesh(mesh, 1.0, pos, mass_assignment='CIC')
    assert mesh.sum() == pytest.approx(n, rel=1e-4)


def test_cic_single_particle_sums_to_one():
    mesh = _mesh(8)
    pos = np.array([[0.3, 0.6, 0.1]], dtype=np.float32)
    map_particles_to_mesh(mesh, 1.0, pos, mass_assignment='CIC')
    assert mesh.sum() == pytest.approx(1.0, rel=1e-5)


# ── TSC ───────────────────────────────────────────────────────────────────────

def test_tsc_conserves_total_weight():
    N, n = 16, 200
    mesh = _mesh(N)
    pos = _positions(n)
    map_particles_to_mesh(mesh, 1.0, pos, mass_assignment='TSC')
    assert mesh.sum() == pytest.approx(n, rel=1e-4)


# ── PCS ───────────────────────────────────────────────────────────────────────

def test_pcs_conserves_total_weight():
    N, n = 16, 100
    mesh = _mesh(N)
    pos = _positions(n)
    map_particles_to_mesh(mesh, 1.0, pos, mass_assignment='PCS')
    assert mesh.sum() == pytest.approx(n, rel=1e-3)


# ── Weighted mapping ──────────────────────────────────────────────────────────

def test_weighted_ngp_sum_equals_total_weight():
    N, n = 16, 100
    mesh = _mesh(N)
    pos = _positions(n)
    weights = np.random.default_rng(1).random(n).astype(np.float32)
    map_particles_to_mesh(mesh, 1.0, pos, mass_assignment='NGP', weights=weights)
    assert mesh.sum() == pytest.approx(weights.sum(), rel=1e-4)


def test_weighted_cic_sum_equals_total_weight():
    N, n = 16, 100
    mesh = _mesh(N)
    pos = _positions(n)
    weights = np.random.default_rng(2).random(n).astype(np.float32)
    map_particles_to_mesh(mesh, 1.0, pos, mass_assignment='CIC', weights=weights)
    assert mesh.sum() == pytest.approx(weights.sum(), rel=1e-4)


# ── Periodic boundary conditions ──────────────────────────────────────────────

def test_ngp_particle_near_box_edge_wraps():
    N = 8
    mesh = _mesh(N)
    # 0.999 * N = 7.992, rounds to 8 -> wraps to cell 0
    pos = np.array([[0.999, 0.5, 0.5]], dtype=np.float32)
    map_particles_to_mesh(mesh, 1.0, pos, mass_assignment='NGP')
    assert mesh.sum() == pytest.approx(1.0)
    assert (mesh > 0).sum() == 1


# ── In-place modification ─────────────────────────────────────────────────────

def test_map_modifies_mesh_in_place():
    mesh = _mesh()
    pos = _positions(50)
    mesh_id = id(mesh)
    map_particles_to_mesh(mesh, 1.0, pos, mass_assignment='NGP')
    assert id(mesh) == mesh_id
    assert mesh.sum() > 0


# ── Accumulation (two calls add up) ──────────────────────────────────────────

def test_two_calls_accumulate():
    N = 8
    mesh = _mesh(N)
    pos = _positions(10, seed=0)
    map_particles_to_mesh(mesh, 1.0, pos, mass_assignment='NGP')
    first_sum = mesh.sum()
    map_particles_to_mesh(mesh, 1.0, pos, mass_assignment='NGP')
    assert mesh.sum() == pytest.approx(2 * first_sum)


# ── Scheme consistency ────────────────────────────────────────────────────────

def test_all_schemes_same_total_mass():
    """All schemes conserve total particle count."""
    N, n = 32, 500
    pos = _positions(n, seed=7)
    totals = {}
    for scheme in ('NGP', 'CIC', 'TSC', 'PCS'):
        m = _mesh(N)
        map_particles_to_mesh(m, 1.0, pos, mass_assignment=scheme)
        totals[scheme] = m.sum()
    for scheme, total in totals.items():
        assert total == pytest.approx(n, rel=1e-3), f"{scheme} total mismatch"


# ── Fused grid+displacement painter (issue #47) ───────────────────────────────

def _displacement(N, box_size=1.0, seed=0, amplitude=0.3):
    """Small random displacement field, shape (N,N,N) per axis."""
    rng = np.random.default_rng(seed)
    cell = box_size / N
    return tuple(
        (rng.standard_normal((N, N, N)) * amplitude * cell).astype(np.float64)
        for _ in range(3)
    )


def _reference_mesh(N, box_size, psi_x, psi_y, psi_z, scheme):
    """Non-fused reference: build the (N^3,3) position array the original
    (get_positions-style) way and paint with map_particles_to_mesh."""
    q1d = (np.arange(N) + 0.5) * (box_size / N)
    x = (q1d[:, None, None] + psi_x) % box_size
    y = (q1d[None, :, None] + psi_y) % box_size
    z = (q1d[None, None, :] + psi_z) % box_size
    positions = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1).astype(np.float32)
    mesh = _mesh(N)
    map_particles_to_mesh(mesh, box_size, positions, mass_assignment=scheme)
    return mesh


@pytest.mark.parametrize('scheme,rel', [
    ('NGP', 1e-5), ('CIC', 1e-4), ('TSC', 1e-4), ('PCS', 1e-3),
])
def test_fused_conserves_total_weight(scheme, rel):
    N = 16
    psi_x, psi_y, psi_z = _displacement(N, seed=1)
    mesh = _mesh(N)
    paint_displacement_field(mesh, 1.0, psi_x, psi_y, psi_z, mass_assignment=scheme, backend='numpy')
    assert mesh.sum() == pytest.approx(N ** 3, rel=rel)


@pytest.mark.parametrize('scheme', ['NGP', 'CIC', 'TSC', 'PCS'])
def test_fused_matches_nonfused_reference(scheme):
    """Fused path (paint_displacement_field) must agree with the original
    get_positions + map_particles_to_mesh path for the same displacement
    field — issue #47's required parity check before the fused path can sit
    alongside (or replace) the existing get_density implementation."""
    N = 16
    psi_x, psi_y, psi_z = _displacement(N, seed=3)
    ref = _reference_mesh(N, 1.0, psi_x, psi_y, psi_z, scheme)
    fused = _mesh(N)
    paint_displacement_field(fused, 1.0, psi_x, psi_y, psi_z, mass_assignment=scheme, backend='numpy')
    np.testing.assert_allclose(fused, ref, rtol=1e-4, atol=1e-5)


def test_fused_weighted_cic_sum_equals_total_weight():
    N = 16
    psi_x, psi_y, psi_z = _displacement(N, seed=4)
    weights = np.random.default_rng(5).random((N, N, N)).astype(np.float32)
    mesh = _mesh(N)
    paint_displacement_field(mesh, 1.0, psi_x, psi_y, psi_z,
                              mass_assignment='CIC', backend='numpy', weights=weights)
    assert mesh.sum() == pytest.approx(weights.sum(), rel=1e-4)


def test_fused_invalid_mass_assignment_raises():
    N = 16
    psi_x, psi_y, psi_z = _displacement(N, seed=6)
    mesh = _mesh(N)
    with pytest.raises(ValueError, match="unknown mass_assignment"):
        paint_displacement_field(mesh, 1.0, psi_x, psi_y, psi_z,
                                  mass_assignment='bogus', backend='numpy')


# ── coarsen_field (issue #48) ──────────────────────────────────────────────────

def test_coarsen_field_conserves_mean():
    rng = np.random.default_rng(0)
    field = rng.random((16, 16, 16))
    coarse = coarsen_field(field, 4)
    assert coarse.shape == (4, 4, 4)
    assert coarse.mean() == pytest.approx(field.mean(), rel=1e-10)


def test_coarsen_field_raises_on_non_divisible_shape():
    field = np.zeros((15, 15, 15))
    with pytest.raises(ValueError, match="not divisible"):
        coarsen_field(field, 4)


def test_coarsen_field_factor_one_is_identity():
    rng = np.random.default_rng(1)
    field = rng.random((8, 8, 8))
    np.testing.assert_array_equal(coarsen_field(field, 1), field)


def test_coarsen_field_matches_nbody_base_coarsen_density():
    """Parity check: BaseNbodyLoader._coarsen_density now delegates to
    coarsen_field (issue #48) -- confirm the two agree on a shared array."""
    from beorn.load_input_data.nbody_base import NBodyLoader

    rng = np.random.default_rng(2)
    field = rng.random((16, 16, 16))
    expected = coarsen_field(field, 4)

    # _coarsen_density is an (unbound) instance method with no use of self;
    # call it directly on the class without constructing a loader.
    actual = NBodyLoader._coarsen_density(None, field, 4)
    np.testing.assert_array_equal(actual, expected)


def test_coarsen_field_jax_matches_numpy():
    jnp = pytest.importorskip("jax.numpy")
    rng = np.random.default_rng(4)
    field = rng.random((16, 16, 16)).astype(np.float32)
    ref = coarsen_field(field, 4)
    out = coarsen_field(jnp.asarray(field), 4)
    assert type(out).__module__.startswith('jax')
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5)


def test_coarsen_field_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    field = rng.random((16, 16, 16)).astype(np.float32)
    ref = coarsen_field(field, 4)
    out = coarsen_field(torch.as_tensor(field), 4)
    assert isinstance(out, torch.Tensor)
    np.testing.assert_allclose(out.numpy(), ref)


# ── upsample_field_fourier (issue #48) ─────────────────────────────────────────

def test_upsample_field_fourier_shape():
    from beorn.particle_mapping import upsample_field_fourier
    rng = np.random.default_rng(6)
    field = rng.standard_normal((16, 16, 16))
    out = upsample_field_fourier(field, 4)
    assert out.shape == (64, 64, 64)


def test_upsample_field_fourier_matches_scipy_reference():
    from scipy.signal import resample
    from beorn.particle_mapping import upsample_field_fourier

    def _scipy_upsample(field, factor):
        N_fine = field.shape[0] * factor
        out = resample(field, N_fine, axis=0)
        out = resample(out, N_fine, axis=1)
        out = resample(out, N_fine, axis=2)
        return out

    rng = np.random.default_rng(7)
    field = rng.standard_normal((15, 15, 15))  # odd N, exercises the non-halved path
    ref = _scipy_upsample(field, 3)
    out = upsample_field_fourier(field, 3)
    np.testing.assert_allclose(out, ref, atol=1e-10)


def test_upsample_field_fourier_jax_matches_numpy():
    jnp = pytest.importorskip("jax.numpy")
    from beorn.particle_mapping import upsample_field_fourier
    rng = np.random.default_rng(8)
    field = rng.standard_normal((16, 16, 16)).astype(np.float32)
    ref = upsample_field_fourier(field, 4)
    out = upsample_field_fourier(jnp.asarray(field), 4)
    assert type(out).__module__.startswith('jax')
    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-4)


def test_upsample_field_fourier_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    from beorn.particle_mapping import upsample_field_fourier
    rng = np.random.default_rng(9)
    field = rng.standard_normal((16, 16, 16)).astype(np.float32)
    ref = upsample_field_fourier(field, 4)
    out = upsample_field_fourier(torch.as_tensor(field), 4)
    assert isinstance(out, torch.Tensor)
    np.testing.assert_allclose(out.numpy(), ref, atol=1e-4)
