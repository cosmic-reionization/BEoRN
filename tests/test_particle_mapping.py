"""Unit tests for beorn.particle_mapping.core (numpy backend)."""
import numpy as np
import pytest

from beorn.particle_mapping.core import map_particles_to_mesh


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
