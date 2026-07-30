"""LPTBase.get_density(fused=True) parity with the non-fused path (issue #47)."""
import numpy as np
import pytest

from beorn.structs import Parameters
from beorn.lpt import ZeldovichApproximation

N, L, Z = 16, 100.0, 7.0
SEED = 42


@pytest.fixture(scope='module')
def param():
    p = Parameters()
    p.simulation.Ncell = N
    p.simulation.Lbox = L
    return p


@pytest.mark.parametrize('scheme', ['NGP', 'CIC', 'TSC', 'PCS'])
def test_get_density_fused_matches_nonfused(param, scheme):
    solver = ZeldovichApproximation(param, verbose=False, seed=SEED)
    solver.generate_initial_conditions()

    ref = solver.get_density(Z, mass_assignment=scheme, fused=False)
    fused = solver.get_density(Z, mass_assignment=scheme, fused=True)

    np.testing.assert_allclose(fused, ref, rtol=1e-3, atol=1e-3)
