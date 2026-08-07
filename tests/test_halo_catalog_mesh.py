"""HaloCatalog.to_mesh()'s per-call deconvolve option (issue #48 follow-up)."""
import numpy as np
import pytest

from beorn.structs import Parameters
from beorn.structs.halo_catalog import HaloCatalog
from beorn.particle_mapping.window import deconvolve_mas

N, L = 16, 100.0


def _catalog(param, n=200, seed=0):
    rng = np.random.default_rng(seed)
    positions = (rng.random((n, 3)) * L).astype(np.float32)
    masses = np.full(n, 1e10)
    return HaloCatalog(positions=positions, masses=masses, parameters=param)


@pytest.fixture
def param():
    p = Parameters()
    p.simulation.Ncell = N
    p.simulation.Lbox = L
    return p


def test_to_mesh_deconvolve_defaults_false(param):
    catalog = _catalog(param)

    default = catalog.to_mesh()
    raw = catalog.to_mesh(deconvolve=False)

    np.testing.assert_allclose(default, raw, rtol=1e-5, atol=1e-6)


def test_to_mesh_deconvolve_preserves_total(param):
    catalog = _catalog(param, n=300, seed=1)

    raw = catalog.to_mesh(deconvolve=False)
    deconvolved = catalog.to_mesh(deconvolve=True)
    assert deconvolved.sum() == pytest.approx(raw.sum(), rel=1e-4)


def test_to_mesh_deconvolve_true_matches_function(param):
    catalog = _catalog(param, n=100, seed=2)

    raw = catalog.to_mesh(deconvolve=False)
    forced = catalog.to_mesh(deconvolve=True)
    expected = deconvolve_mas(raw, L, 'NGP')

    np.testing.assert_allclose(forced, expected, rtol=1e-5, atol=1e-6)


# ── halo_sim.mass_assignment (parameters restructuring) ───────────────────────

def test_halo_sim_mass_assignment_defaults_to_ngp(param):
    assert param.halo_sim.mass_assignment == 'NGP'


def test_to_mesh_reads_halo_sim_mass_assignment_default(param):
    """to_mesh() must actually read halo_sim.mass_assignment, not just default
    to a hardcoded 'NGP' literal internally -- overriding the field should
    change the painted mesh."""
    catalog = _catalog(param, n=300, seed=3)
    ngp = catalog.to_mesh(deconvolve=False)

    param.halo_sim.mass_assignment = 'CIC'
    cic = catalog.to_mesh(deconvolve=False)

    assert not np.allclose(ngp, cic)
    assert ngp.sum() == pytest.approx(cic.sum(), rel=1e-4)
