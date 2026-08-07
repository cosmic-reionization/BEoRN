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
    map_particles_to_mesh(mesh, 1.0, pos, mass_assignment='NGP', deconvolve=False)
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
    # 0.999*N - 0.5 = 7.492, rounds to cell 7 -- no wrap needed for this
    # particular value under the cell-centered convention (issue #55), but
    # it still must land in exactly one cell with mass conserved.
    pos = np.array([[0.999, 0.5, 0.5]], dtype=np.float32)
    map_particles_to_mesh(mesh, 1.0, pos, mass_assignment='NGP', deconvolve=False)
    assert mesh.sum() == pytest.approx(1.0)
    assert (mesh > 0).sum() == 1


def test_cic_particle_near_left_edge_wraps():
    N = 8
    mesh = _mesh(N)
    # 0.01*N - 0.5 = -0.42 -> floor = -1 -> wraps to cell N-1, split with cell 0.
    pos = np.array([[0.01, 0.5, 0.5]], dtype=np.float32)
    map_particles_to_mesh(mesh, 1.0, pos, mass_assignment='CIC', deconvolve=False)
    assert mesh.sum() == pytest.approx(1.0, rel=1e-5)
    ix = np.nonzero(mesh.sum(axis=(1, 2)))[0]
    assert set(ix.tolist()) == {0, N - 1}


# ── Cell-centered indexing (issue #55) ─────────────────────────────────────────
# Mesh index i is the *center* of cell i, matching the Lagrangian tracer grid
# used everywhere upstream ((arange(N)+0.5)*cell) and py21cmfast's own
# PerturbField.c convention. Before this fix, every scheme anchored its
# stencil to floor(p)/round(p) with no offset -- an undisplaced particle
# (sitting exactly at its own cell's center) split its weight across
# neighboring cells instead of landing entirely in its own cell.

@pytest.mark.parametrize('scheme', ['NGP', 'CIC'])
def test_particle_at_cell_center_lands_entirely_in_that_cell(scheme):
    N, L, i = 16, 1.0, 5
    cell = L / N
    pos = np.array([[(i + 0.5) * cell] * 3], dtype=np.float32)
    mesh = _mesh(N)
    map_particles_to_mesh(mesh, L, pos, mass_assignment=scheme, deconvolve=False)
    assert mesh[i, i, i] == pytest.approx(1.0, rel=1e-5)
    assert mesh.sum() == pytest.approx(1.0, rel=1e-5)


@pytest.mark.parametrize('scheme,expected_center_weight', [
    ('TSC', 0.75 ** 3), ('PCS', (2.0 / 3.0) ** 3),
])
def test_particle_at_cell_center_peaks_at_that_cell(scheme, expected_center_weight):
    """TSC/PCS always spread mass over neighboring cells even at zero
    sub-cell offset (by kernel design), but the stencil must be centered
    exactly on the particle's own cell -- not shifted by one, as it was
    before this fix (where the ambiguous d=0.5 tie broke either way
    depending on cell-index parity)."""
    N, L, i = 16, 1.0, 5
    cell = L / N
    pos = np.array([[(i + 0.5) * cell] * 3], dtype=np.float32)
    mesh = _mesh(N)
    map_particles_to_mesh(mesh, L, pos, mass_assignment=scheme, deconvolve=False)
    assert mesh[i, i, i] == pytest.approx(expected_center_weight, rel=1e-4)
    assert mesh[i, i, i] == mesh.max()
    assert mesh.sum() == pytest.approx(1.0, rel=1e-4)


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


# ── deconvolve integration (issue #48 follow-up) ───────────────────────────────
# Painting functions correct the mass-assignment window in place by default —
# any mesh built here should already match a manual paint(deconvolve=False) +
# deconvolve_mas() call, and its total (weighted) sum should be unaffected
# since the window is 1 at k=0.

from beorn.particle_mapping.window import deconvolve_mas


@pytest.mark.parametrize('scheme', ['NGP', 'CIC', 'TSC', 'PCS'])
def test_map_particles_to_mesh_deconvolve_matches_manual_call(scheme):
    N, n, L = 16, 300, 1.0
    pos = _positions(n, box_size=L, seed=8)

    raw = _mesh(N)
    map_particles_to_mesh(raw, L, pos, mass_assignment=scheme, deconvolve=False)
    expected = deconvolve_mas(raw, L, scheme)

    auto = _mesh(N)
    map_particles_to_mesh(auto, L, pos, mass_assignment=scheme)  # deconvolve=True default
    np.testing.assert_allclose(auto, expected, rtol=1e-5, atol=1e-6)


def test_map_particles_to_mesh_deconvolve_preserves_total():
    N, n, L = 16, 300, 1.0
    pos = _positions(n, box_size=L, seed=9)

    raw = _mesh(N)
    map_particles_to_mesh(raw, L, pos, mass_assignment='CIC', deconvolve=False)

    deconvolved = _mesh(N)
    map_particles_to_mesh(deconvolved, L, pos, mass_assignment='CIC')
    assert deconvolved.sum() == pytest.approx(raw.sum(), rel=1e-4)


def test_paint_displacement_field_deconvolve_matches_manual_call():
    N, L, scheme = 16, 1.0, 'CIC'
    psi_x, psi_y, psi_z = _displacement(N, seed=10)

    raw = _mesh(N)
    paint_displacement_field(raw, L, psi_x, psi_y, psi_z,
                              mass_assignment=scheme, backend='numpy', deconvolve=False)
    expected = deconvolve_mas(raw, L, scheme)

    auto = _mesh(N)
    paint_displacement_field(auto, L, psi_x, psi_y, psi_z,
                              mass_assignment=scheme, backend='numpy')  # deconvolve=True default
    np.testing.assert_allclose(auto, expected, rtol=1e-5, atol=1e-6)


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


# ── dtype precision (issue #52) ────────────────────────────────────────────────
# get_density()/paint_displacement_field() used to hardcode float32 for the
# mesh and particle positions regardless of backend, silently truncating the
# (float64-by-default) LPT displacement field before painting. These tests
# confirm float64 is now selectable end to end for the numpy/numba backends,
# and that the historical float32 default is unchanged.

def _mesh64(N=16):
    return np.zeros((N, N, N), dtype=np.float64)


def _positions64(n, box_size=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3)) * box_size).astype(np.float64)


@pytest.mark.parametrize('scheme', ['NGP', 'CIC', 'TSC', 'PCS'])
def test_map_particles_to_mesh_float64_conserves_total_weight(scheme):
    N, n = 16, 500
    mesh = _mesh64(N)
    pos = _positions64(n)
    map_particles_to_mesh(mesh, 1.0, pos, mass_assignment=scheme, deconvolve=False)
    assert mesh.dtype == np.float64
    assert mesh.sum() == pytest.approx(n, rel=1e-3)


def test_map_particles_to_mesh_mismatched_dtype_raises():
    # backend='numpy' explicit: only that backend (and 'numba') enforce a
    # strict dtype match -- 'torch'/'jax' happily upcast, so leaving backend
    # at its 'auto' default would make this test environment-dependent.
    mesh = _mesh64()
    pos = _positions(10)  # float32
    with pytest.raises(AssertionError, match="dtype"):
        map_particles_to_mesh(mesh, 1.0, pos, mass_assignment='CIC', backend='numpy', deconvolve=False)


def test_map_particles_to_mesh_unsupported_dtype_raises():
    mesh = np.zeros((16, 16, 16), dtype=np.float16)
    pos = (np.random.default_rng(0).random((10, 3))).astype(np.float16)
    with pytest.raises(AssertionError, match="float32 or float64"):
        map_particles_to_mesh(mesh, 1.0, pos, mass_assignment='CIC', backend='numpy', deconvolve=False)


@pytest.mark.parametrize('scheme', ['NGP', 'CIC', 'TSC', 'PCS'])
def test_float64_and_float32_painting_agree_closely(scheme):
    """float64 painting should reproduce float32 painting to within float32
    precision for positions that don't sit near a cell-boundary rounding
    edge — a sanity check that the dtype threading didn't change the maths,
    only the precision."""
    N, n = 32, 2000
    pos32 = _positions(n, seed=11)
    pos64 = pos32.astype(np.float64)

    mesh32 = _mesh(N)
    map_particles_to_mesh(mesh32, 1.0, pos32, mass_assignment=scheme, deconvolve=False)

    mesh64 = _mesh64(N)
    map_particles_to_mesh(mesh64, 1.0, pos64, mass_assignment=scheme, deconvolve=False)

    np.testing.assert_allclose(mesh64, mesh32, rtol=1e-3, atol=1e-4)


def test_fused_paint_displacement_field_float64():
    N = 16
    psi_x, psi_y, psi_z = _displacement(N, seed=12)  # already float64
    mesh = _mesh64(N)
    paint_displacement_field(mesh, 1.0, psi_x, psi_y, psi_z,
                              mass_assignment='CIC', backend='numpy', deconvolve=False)
    assert mesh.dtype == np.float64
    assert mesh.sum() == pytest.approx(N ** 3, rel=1e-4)


def test_fused_paint_displacement_field_float64_matches_float32_closely():
    N = 16
    psi_x64 = np.random.default_rng(13).standard_normal((N, N, N)) * 0.01
    psi_y64 = np.random.default_rng(14).standard_normal((N, N, N)) * 0.01
    psi_z64 = np.random.default_rng(15).standard_normal((N, N, N)) * 0.01

    mesh64 = _mesh64(N)
    paint_displacement_field(mesh64, 1.0, psi_x64, psi_y64, psi_z64,
                              mass_assignment='CIC', backend='numpy', deconvolve=False)

    mesh32 = _mesh(N)
    paint_displacement_field(mesh32, 1.0,
                              psi_x64.astype(np.float32), psi_y64.astype(np.float32),
                              psi_z64.astype(np.float32),
                              mass_assignment='CIC', backend='numpy', deconvolve=False)

    np.testing.assert_allclose(mesh64, mesh32, rtol=1e-3, atol=1e-4)


def test_numba_backend_float64_matches_numpy_backend():
    pytest.importorskip("numba")
    from beorn.particle_mapping import numba_backend

    N, n = 16, 300
    pos = _positions64(n, seed=16)

    mesh_numba = _mesh64(N)
    numba_backend.map_particles_to_mesh(mesh_numba, 1.0, pos, mass_assignment='CIC')

    mesh_numpy = _mesh64(N)
    map_particles_to_mesh(mesh_numpy, 1.0, pos, mass_assignment='CIC',
                           backend='numpy', deconvolve=False)
    np.testing.assert_allclose(mesh_numba, mesh_numpy, rtol=1e-10)


def test_paint_mesh_functional_dtype_float64():
    from beorn.particle_mapping.core import paint_mesh
    N, n = 16, 200
    pos = _positions64(n, seed=17)
    weights = None
    mesh = paint_mesh(pos, weights, N, 1.0, mass_assignment='CIC',
                       backend='numpy', deconvolve=False, dtype='float64')
    assert mesh.dtype == np.float64
    assert mesh.sum() == pytest.approx(n, rel=1e-3)


def test_paint_mesh_functional_default_dtype_is_float32():
    from beorn.particle_mapping.core import paint_mesh
    N, n = 16, 50
    pos = _positions(n, seed=18)
    mesh = paint_mesh(pos, None, N, 1.0, mass_assignment='CIC', backend='numpy', deconvolve=False)
    assert mesh.dtype == np.float32


def test_map_particles_to_mesh_jax_float64_matches_numpy():
    jax = pytest.importorskip("jax")
    jax.config.update('jax_enable_x64', True)
    from beorn.particle_mapping import jax_backend

    N, n = 16, 300
    pos = _positions64(n, seed=19)

    mesh_jax = _mesh64(N)
    jax_backend.map_particles_to_mesh(mesh_jax, 1.0, pos, mass_assignment='CIC')

    mesh_numpy = _mesh64(N)
    map_particles_to_mesh(mesh_numpy, 1.0, pos, mass_assignment='CIC',
                           backend='numpy', deconvolve=False)
    assert mesh_jax.dtype == np.float64
    np.testing.assert_allclose(mesh_jax, mesh_numpy, rtol=1e-6)


def test_map_particles_to_mesh_torch_float64_matches_numpy():
    torch = pytest.importorskip("torch")
    from beorn.particle_mapping import torch_backend

    N, n = 16, 300
    pos = _positions64(n, seed=20)

    mesh_torch = _mesh64(N)
    torch_backend.map_particles_to_mesh(mesh_torch, 1.0, pos, mass_assignment='CIC')

    mesh_numpy = _mesh64(N)
    map_particles_to_mesh(mesh_numpy, 1.0, pos, mass_assignment='CIC',
                           backend='numpy', deconvolve=False)
    assert mesh_torch.dtype == np.float64
    np.testing.assert_allclose(mesh_torch, mesh_numpy, rtol=1e-6)
