"""Unit tests for the shared-phase multi-resolution IC helpers (issue #56):
synchronized_white_noise / extract_synced_delta_k / _extract_lowk_rfftn.

These back HaloSimParameters.field_oversample -- the CHMF's own conditioning
field needs to be resolvable at a finer grid than Ncell while staying a
phase-consistent view of the same box used elsewhere in the pipeline (not an
independently-drawn, decorrelated realisation).
"""
import numpy as np
import pytest

from beorn.structs import Parameters
from beorn.lpt import ZeldovichApproximation
from beorn.lpt.lpt import (
    synchronized_white_noise, extract_synced_delta_k, _extract_lowk_rfftn,
)
from beorn.lpt.linear_power import get_power_spectrum

L = 200.0
N_FINE = 32
SEED = 12345


def _params(N):
    p = Parameters()
    p.simulation.Ncell = N
    p.simulation.Lbox = L
    p.simulation.use_hunits = True
    return p


def test_identity_matches_generate_initial_conditions_exactly():
    """N_target == N_fine (i.e. field_oversample == 1 everywhere) must
    reproduce today's LPTBase.generate_initial_conditions() bit-for-bit --
    the synchronized path must not change default behaviour."""
    p = _params(N_FINE)
    ps = get_power_spectrum('eisenstein_hu', p)

    noise_k_fine = synchronized_white_noise(N_FINE, SEED, fixed=True)
    dk_synced = extract_synced_delta_k(noise_k_fine, N_FINE, N_FINE, p.Lbox_hunits, ps)

    solver = ZeldovichApproximation(p, verbose=False, seed=SEED, power_spectrum=ps)
    dk_native = solver.generate_initial_conditions()

    np.testing.assert_allclose(dk_synced, dk_native, rtol=1e-12, atol=0)


def test_chained_extraction_matches_direct_extraction():
    """Extracting fine->small directly must equal chaining fine->mid->small --
    verifies the low-k index bookkeeping (esp. the Nyquist-plane wraparound)
    is self-consistent, not just correct in the identity case."""
    noise_k_fine = synchronized_white_noise(N_FINE, SEED, fixed=True)
    N_mid, N_small = 16, 8

    direct = _extract_lowk_rfftn(noise_k_fine, N_FINE, N_small)
    mid = _extract_lowk_rfftn(noise_k_fine, N_FINE, N_mid)
    chained = _extract_lowk_rfftn(mid, N_mid, N_small)

    np.testing.assert_allclose(direct, chained, rtol=1e-12, atol=0)


def test_extract_rejects_larger_target_than_fine():
    noise_k_fine = synchronized_white_noise(16, SEED, fixed=True)
    with pytest.raises(ValueError, match="must not exceed"):
        _extract_lowk_rfftn(noise_k_fine, 16, 32)


def test_extract_rejects_odd_grid_sizes():
    noise_k_fine = synchronized_white_noise(16, SEED, fixed=True)
    with pytest.raises(ValueError, match="must both be even"):
        _extract_lowk_rfftn(noise_k_fine, 16, 5)


def test_different_seeds_give_different_noise():
    a = synchronized_white_noise(N_FINE, SEED, fixed=True)
    b = synchronized_white_noise(N_FINE, SEED + 1, fixed=True)
    assert not np.allclose(a, b)


def test_paired_negative_seed_negates():
    pos = synchronized_white_noise(N_FINE, SEED, fixed=False)
    neg = synchronized_white_noise(N_FINE, -SEED, fixed=False)
    np.testing.assert_allclose(pos, -neg)


def test_extracted_field_has_sane_variance():
    p = _params(N_FINE)
    ps = get_power_spectrum('eisenstein_hu', p)
    N_small = 8
    noise_k_fine = synchronized_white_noise(N_FINE, SEED, fixed=True)
    dk_small = extract_synced_delta_k(noise_k_fine, N_FINE, N_small, p.Lbox_hunits, ps)
    delta_small = np.fft.irfftn(dk_small, s=(N_small, N_small, N_small))
    assert 0.01 < delta_small.var() < 100.0
