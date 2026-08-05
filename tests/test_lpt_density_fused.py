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


# ── dtype precision (issue #52) ────────────────────────────────────────────────
# get_density()/get_positions() used to hardcode float32 regardless of the
# (float64-by-default) displacement field's own precision. These confirm the
# new dtype='float64' opt-in works and the historical default is unchanged.

def test_get_density_default_dtype_is_float32(param):
    solver = ZeldovichApproximation(param, verbose=False, seed=SEED)
    solver.generate_initial_conditions()
    delta = solver.get_density(Z, mass_assignment='CIC')
    assert delta.dtype == np.float32


@pytest.mark.parametrize('scheme', ['NGP', 'CIC', 'TSC', 'PCS'])
@pytest.mark.parametrize('fused', [True, False])
def test_get_density_dtype_float64(param, scheme, fused):
    solver = ZeldovichApproximation(param, verbose=False, seed=SEED)
    solver.generate_initial_conditions()
    delta = solver.get_density(Z, mass_assignment=scheme, fused=fused, dtype='float64')
    assert delta.dtype == np.float64


def test_get_density_dtype_float64_closely_matches_float32(param):
    """float64 painting shouldn't change the physics, just the precision --
    the two should agree closely for a displacement field with no particles
    sitting exactly on a cell-boundary rounding edge."""
    solver = ZeldovichApproximation(param, verbose=False, seed=SEED)
    solver.generate_initial_conditions()
    delta32 = solver.get_density(Z, mass_assignment='CIC', dtype='float32')
    delta64 = solver.get_density(Z, mass_assignment='CIC', dtype='float64')
    np.testing.assert_allclose(delta64, delta32, rtol=1e-3, atol=1e-3)


def test_get_positions_dtype_float64(param):
    solver = ZeldovichApproximation(param, verbose=False, seed=SEED)
    solver.generate_initial_conditions()
    pos32 = solver.get_positions(Z)
    pos64 = solver.get_positions(Z, dtype='float64')
    assert pos32.dtype == np.float32
    assert pos64.dtype == np.float64
    np.testing.assert_allclose(pos64, pos32, rtol=1e-4, atol=1e-4)
