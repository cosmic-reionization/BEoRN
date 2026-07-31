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
    delta_coarse = solver.get_density(Z, mass_assignment='CIC', oversample=1)
    delta_fine = solver.get_density(Z, mass_assignment='CIC', oversample=4)

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
