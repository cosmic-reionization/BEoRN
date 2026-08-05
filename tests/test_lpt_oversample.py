"""Acceptance test for issue #48's fine-paint-then-downsample: oversampling
should reduce the mass-assignment window's P(k) suppression near k_Nyquist,
without any deconvolution.
"""
import numpy as np
import pytest

from beorn.structs import Parameters
from beorn.lpt import ZeldovichApproximation
from beorn.power_spectrum import power_spectrum_1d

N, L, Z = 32, 200.0, 20.0
SEED = 42


@pytest.fixture(scope='module')
def solver():
    p = Parameters()
    p.simulation.Ncell = N
    p.simulation.Lbox = L
    s = ZeldovichApproximation(p, verbose=False, seed=SEED)
    s.generate_initial_conditions()
    return s


def test_get_density_oversample_validates_input(solver):
    with pytest.raises(ValueError, match="oversample must be a positive int"):
        solver.get_density(Z, oversample=0)


def test_get_density_oversample_returns_coarse_grid_shape(solver):
    delta = solver.get_density(Z, mass_assignment='CIC', oversample=4)
    assert delta.shape == (N, N, N)


def test_oversample_reduces_mas_suppression_near_k_nyquist(solver):
    # deconvolve=False: isolate oversampling's own benefit. Since issue #48's
    # deconvolve-at-paint-time follow-up, get_density deconvolves by default,
    # which would otherwise already remove most of the window suppression
    # this test is specifically measuring.
    delta_coarse = solver.get_density(Z, mass_assignment='CIC', oversample=1, deconvolve=False)
    delta_fine = solver.get_density(Z, mass_assignment='CIC', oversample=4, deconvolve=False)

    Pk_coarse, bins_coarse, kny = power_spectrum_1d(delta_coarse, L, kbins=16)
    Pk_fine, bins_fine, _ = power_spectrum_1d(delta_fine, L, kbins=16)

    Pk_lin_coarse = solver.power_spectrum.P(bins_coarse, z=Z)
    Pk_lin_fine = solver.power_spectrum.P(bins_fine, z=Z)

    mask_c = (bins_coarse < kny) & (Pk_lin_coarse > 0)
    mask_f = (bins_fine < kny) & (Pk_lin_fine > 0)

    rel_err_coarse = np.abs(Pk_coarse[mask_c] - Pk_lin_coarse[mask_c]) / Pk_lin_coarse[mask_c]
    rel_err_fine = np.abs(Pk_fine[mask_f] - Pk_lin_fine[mask_f]) / Pk_lin_fine[mask_f]

    # Focus on the highest-k quartile of bins within k_Nyquist, where MAS
    # window suppression bites hardest -- exactly where oversampling should help.
    n_c = max(1, len(rel_err_coarse) // 4)
    n_f = max(1, len(rel_err_fine) // 4)
    assert rel_err_fine[-n_f:].mean() < rel_err_coarse[-n_c:].mean()


def test_get_density_oversample_parameters_default_and_override():
    # Own Parameters instance (not the module-scoped `solver` fixture) since
    # this test mutates parameters.simulation.oversample.
    p = Parameters()
    p.simulation.Ncell = N
    p.simulation.Lbox = L
    assert p.simulation.oversample == 1

    s = ZeldovichApproximation(p, verbose=False, seed=SEED)
    s.generate_initial_conditions()

    default = s.get_density(Z, mass_assignment='CIC')
    explicit_1 = s.get_density(Z, mass_assignment='CIC', oversample=1)
    np.testing.assert_allclose(default, explicit_1, rtol=1e-4, atol=1e-6)

    p.simulation.oversample = 4
    default_4 = s.get_density(Z, mass_assignment='CIC')
    explicit_4 = s.get_density(Z, mass_assignment='CIC', oversample=4)
    np.testing.assert_allclose(default_4, explicit_4, rtol=1e-4, atol=1e-6)
    assert not np.allclose(default_4, explicit_1)


def test_get_density_oversample_dtype_float64(solver):
    """dtype threading (issue #52) also applies to the oversample>1 path."""
    delta = solver.get_density(Z, mass_assignment='CIC', oversample=4, dtype='float64')
    assert delta.dtype == np.float64
    assert delta.shape == (N, N, N)
