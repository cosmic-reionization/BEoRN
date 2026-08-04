"""HaloCatalog.to_mesh() deconvolve integration (issue #48 follow-up)."""
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


def test_to_mesh_deconvolve_defaults_true_from_parameters(param):
    assert param.simulation.deconvolve_mas is True
    catalog = _catalog(param)

    auto = catalog.to_mesh()
    raw = catalog.to_mesh(deconvolve=False)
    expected = deconvolve_mas(raw, L, 'NGP')

    np.testing.assert_allclose(auto, expected, rtol=1e-5, atol=1e-6)


def test_to_mesh_deconvolve_preserves_total(param):
    catalog = _catalog(param, n=300, seed=1)

    raw = catalog.to_mesh(deconvolve=False)
    deconvolved = catalog.to_mesh(deconvolve=True)
    assert deconvolved.sum() == pytest.approx(raw.sum(), rel=1e-4)


def test_to_mesh_deconvolve_argument_overrides_parameters(param):
    param.simulation.deconvolve_mas = False
    catalog = _catalog(param, n=100, seed=2)

    raw = catalog.to_mesh()  # follows parameters.simulation.deconvolve_mas=False
    forced = catalog.to_mesh(deconvolve=True)
    expected = deconvolve_mas(raw, L, 'NGP')

    np.testing.assert_allclose(forced, expected, rtol=1e-5, atol=1e-6)
