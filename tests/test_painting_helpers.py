"""Unit tests for beorn.painting.helpers."""
import numpy as np
import pytest

from beorn.painting.helpers import (
    precompute_fft,
    fft_convolve_periodic,
    profile_to_3Dkernel,
    stacked_lyal_kernel,
    stacked_T_kernel,
)


# ── precompute_fft ────────────────────────────────────────────────────────────

def test_precompute_fft_output_shape():
    arr = np.ones((8, 8, 8))
    fa = precompute_fft(arr)
    # rfftn: last dim is N//2 + 1
    assert fa.shape == (8, 8, 5)


def test_precompute_fft_output_is_complex():
    arr = np.ones((4, 4, 4))
    fa = precompute_fft(arr)
    assert np.iscomplexobj(fa)


def test_precompute_fft_unknown_backend_raises():
    arr = np.ones((4, 4, 4))
    with pytest.raises(ValueError, match="Unknown backend"):
        precompute_fft(arr, backend='invalid')


# ── fft_convolve_periodic ─────────────────────────────────────────────────────

def test_fft_convolve_delta_kernel_returns_input():
    """Convolution with a centered delta kernel is an identity."""
    N = 16
    rng = np.random.default_rng(42)
    signal = rng.random((N, N, N))
    fa = precompute_fft(signal)

    delta = np.zeros((N, N, N))
    delta[N // 2, N // 2, N // 2] = 1.0

    result = fft_convolve_periodic(fa, delta, signal.shape)
    np.testing.assert_allclose(result, signal, atol=1e-7)


def test_fft_convolve_output_shape():
    N = 8
    signal = np.ones((N, N, N))
    kernel = np.zeros((N, N, N))
    kernel[N // 2, N // 2, N // 2] = 1.0
    fa = precompute_fft(signal)
    result = fft_convolve_periodic(fa, kernel, signal.shape)
    assert result.shape == (N, N, N)


def test_fft_convolve_unknown_backend_raises():
    fa = np.ones((4, 4, 3), dtype=complex)
    kernel = np.ones((4, 4, 4))
    with pytest.raises(ValueError, match="Unknown backend"):
        fft_convolve_periodic(fa, kernel, (4, 4, 4), backend='bad')


def test_fft_convolve_constant_signal_times_normalized_kernel():
    """Convolving a constant field with a kernel that sums to 1 gives the same constant."""
    N = 8
    signal = np.full((N, N, N), 3.0)
    fa = precompute_fft(signal)

    kernel = np.zeros((N, N, N))
    kernel[N // 2, N // 2, N // 2] = 1.0

    result = fft_convolve_periodic(fa, kernel, signal.shape)
    np.testing.assert_allclose(result, signal, atol=1e-6)


# ── profile_to_3Dkernel ───────────────────────────────────────────────────────

def test_profile_to_3d_kernel_shape():
    kern = profile_to_3Dkernel(lambda r: np.ones_like(r), nGrid=8, LB=10.0)
    assert kern.shape == (8, 8, 8)


def test_profile_to_3d_kernel_all_finite():
    kern = profile_to_3Dkernel(lambda r: np.exp(-r), nGrid=8, LB=5.0)
    assert np.all(np.isfinite(kern))


def test_profile_to_3d_kernel_symmetric():
    N = 16
    kern = profile_to_3Dkernel(lambda r: np.exp(-r), nGrid=N, LB=4.0)
    # Symmetric about center: corner values should be equal
    assert kern[0, 0, 0] == pytest.approx(kern[-1, -1, -1], rel=1e-10)
    assert kern[0, 0, 0] == pytest.approx(kern[-1, 0, 0], rel=1e-10)


def test_profile_to_3d_kernel_nonfinite_raises():
    def bad_profile(r):
        out = np.ones_like(r)
        out.flat[0] = np.nan
        return out

    with pytest.raises(AssertionError):
        profile_to_3Dkernel(bad_profile, nGrid=4, LB=1.0)


# ── stacked_lyal_kernel ───────────────────────────────────────────────────────

@pytest.fixture
def lyal_profile_data():
    rr = np.linspace(0, 10, 100)
    lyal = np.where(rr < 5.0, np.exp(-rr), 0.0)
    return rr, lyal


def test_stacked_lyal_kernel_shape(lyal_profile_data):
    rr, lyal = lyal_profile_data
    kern = stacked_lyal_kernel(rr, lyal, LBox=100.0, nGrid=8, nGrid_min=8)
    assert kern.shape == (8, 8, 8)


def test_stacked_lyal_kernel_finite(lyal_profile_data):
    rr, lyal = lyal_profile_data
    kern = stacked_lyal_kernel(rr, lyal, LBox=100.0, nGrid=8, nGrid_min=8)
    assert np.all(np.isfinite(kern))


def test_stacked_lyal_kernel_nonnegative(lyal_profile_data):
    rr, lyal = lyal_profile_data
    kern = stacked_lyal_kernel(rr, lyal, LBox=100.0, nGrid=8, nGrid_min=8)
    assert np.all(kern >= -1e-10)


# ── stacked_T_kernel ──────────────────────────────────────────────────────────

@pytest.fixture
def T_profile_data():
    rr = np.linspace(0, 10, 100)
    T_arr = np.where(rr < 5.0, np.exp(-rr), 0.0)
    return rr, T_arr


def test_stacked_t_kernel_shape(T_profile_data):
    rr, T_arr = T_profile_data
    kern = stacked_T_kernel(rr, T_arr, LBox=100.0, nGrid=8, nGrid_min=8)
    assert kern.shape == (8, 8, 8)


def test_stacked_t_kernel_finite(T_profile_data):
    rr, T_arr = T_profile_data
    kern = stacked_T_kernel(rr, T_arr, LBox=100.0, nGrid=8, nGrid_min=8)
    assert np.all(np.isfinite(kern))


def test_stacked_t_kernel_always_nonzero_profile():
    # If T never reaches zero, ind_T_0 = -1, rr_T_max = rr[-1]
    rr = np.linspace(0.1, 10, 100)
    T_arr = np.exp(-rr)  # never exactly zero
    kern = stacked_T_kernel(rr, T_arr, LBox=100.0, nGrid=8, nGrid_min=8)
    assert kern.shape == (8, 8, 8)
    assert np.all(np.isfinite(kern))
