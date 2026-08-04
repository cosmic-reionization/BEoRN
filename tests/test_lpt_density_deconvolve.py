"""LPTBase.get_density's mass_assignment/deconvolve Parameters wiring (issue #48 follow-up)."""
import numpy as np
import pytest

from beorn.structs import Parameters
from beorn.lpt import ZeldovichApproximation

N, L, Z = 16, 100.0, 7.0
SEED = 42


@pytest.fixture
def param():
    p = Parameters()
    p.simulation.Ncell = N
    p.simulation.Lbox = L
    return p


def _solver(param):
    solver = ZeldovichApproximation(param, verbose=False, seed=SEED)
    solver.generate_initial_conditions()
    return solver


def test_get_density_defaults_match_parameters_simulation_fields(param):
    assert param.simulation.mass_assignment == 'CIC'
    assert param.simulation.deconvolve_mas is True

    solver = _solver(param)
    default = solver.get_density(Z)
    explicit = solver.get_density(Z, mass_assignment='CIC', deconvolve=True)
    np.testing.assert_allclose(default, explicit, rtol=1e-4, atol=1e-6)


def test_get_density_deconvolve_changes_output(param):
    solver = _solver(param)
    raw = solver.get_density(Z, deconvolve=False)
    deconvolved = solver.get_density(Z, deconvolve=True)
    assert not np.allclose(raw, deconvolved)
    # Overdensity is (mesh/mean - 1); the window is 1 at k=0, so the mean of
    # the underlying painted mesh is preserved and both remain mean-zero.
    assert raw.mean() == pytest.approx(0.0, abs=1e-4)
    assert deconvolved.mean() == pytest.approx(0.0, abs=1e-4)


def test_get_density_mass_assignment_parameters_override(param):
    param.simulation.mass_assignment = 'TSC'
    solver = _solver(param)

    default = solver.get_density(Z)
    explicit_tsc = solver.get_density(Z, mass_assignment='TSC')
    explicit_cic = solver.get_density(Z, mass_assignment='CIC')

    np.testing.assert_allclose(default, explicit_tsc, rtol=1e-4, atol=1e-6)
    assert not np.allclose(default, explicit_cic)


def test_get_density_deconvolve_parameters_override(param):
    param.simulation.deconvolve_mas = False
    solver = _solver(param)

    default = solver.get_density(Z)  # follows parameters.simulation.deconvolve_mas=False
    explicit_false = solver.get_density(Z, deconvolve=False)
    explicit_true = solver.get_density(Z, deconvolve=True)

    np.testing.assert_allclose(default, explicit_false, rtol=1e-4, atol=1e-6)
    assert not np.allclose(default, explicit_true)


def test_get_density_oversample_deconvolve_applies_before_coarsening(param):
    solver = _solver(param)
    raw = solver.get_density(Z, oversample=4, deconvolve=False)
    deconvolved = solver.get_density(Z, oversample=4, deconvolve=True)
    assert not np.allclose(raw, deconvolved)
    assert deconvolved.shape == (N, N, N)
