"""Unit tests for beorn.particle_mapping.window (issue #48)."""
import numpy as np
import pytest

from beorn.particle_mapping import k_nyquist, deconvolve_mas
from beorn.particle_mapping.window import _mas_window


# ── k_nyquist ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("L,N", [(1.0, 16), (100.0, 128), (200.0, 512)])
def test_k_nyquist_matches_formula(L, N):
    assert k_nyquist(L, N) == pytest.approx(np.pi * N / L)


def test_k_nyquist_matches_tools21cm_convention():
    t2c = pytest.importorskip("tools21cm")
    L, N = 100.0, 64
    expected = t2c.power_spectrum._resolve_k_limit('nyquist', [L, L, L], (N, N, N))
    assert k_nyquist(L, N) == pytest.approx(expected)


# ── deconvolve_mas ────────────────────────────────────────────────────────────

def _convolve_with_window(field, L, mass_assignment):
    """Apply the MAS window forward (multiply, not divide) -- inverse of
    deconvolve_mas, used to build a synthetic "painted-looking" field."""
    N = field.shape[0]
    field_k = np.fft.rfftn(field)
    field_k = field_k * _mas_window(L, N, mass_assignment)
    return np.fft.irfftn(field_k, s=(N, N, N))


@pytest.mark.parametrize("scheme", ["NGP", "CIC", "TSC", "PCS"])
def test_deconvolve_mas_round_trip(scheme):
    N, L = 16, 10.0
    rng = np.random.default_rng(0)
    field = rng.standard_normal((N, N, N))

    convolved = _convolve_with_window(field, L, scheme)
    recovered = deconvolve_mas(convolved, L, scheme)

    np.testing.assert_allclose(recovered, field, atol=1e-8)


def test_deconvolve_mas_accepts_complex_field_k():
    N, L, scheme = 16, 10.0, 'CIC'
    rng = np.random.default_rng(1)
    field = rng.standard_normal((N, N, N))
    convolved = _convolve_with_window(field, L, scheme)

    field_k = np.fft.rfftn(convolved)
    recovered_from_k = deconvolve_mas(field_k, L, scheme)
    recovered_from_real = deconvolve_mas(convolved, L, scheme)

    np.testing.assert_allclose(recovered_from_k, recovered_from_real)


def test_deconvolve_mas_unknown_scheme_raises():
    N, L = 8, 10.0
    field = np.zeros((N, N, N))
    with pytest.raises(ValueError, match="Unknown mass_assignment"):
        deconvolve_mas(field, L, 'bogus')


# ── jax/torch backend dispatch ────────────────────────────────────────────────

@pytest.mark.parametrize("scheme", ["NGP", "CIC", "TSC", "PCS"])
def test_deconvolve_mas_jax_matches_numpy(scheme):
    jnp = pytest.importorskip("jax.numpy")
    N, L = 16, 10.0
    rng = np.random.default_rng(2)
    field = rng.standard_normal((N, N, N)).astype(np.float32)
    convolved = _convolve_with_window(field, L, scheme).astype(np.float32)

    ref = deconvolve_mas(convolved, L, scheme)
    out = deconvolve_mas(jnp.asarray(convolved), L, scheme)

    assert type(out).__module__.startswith('jax')
    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-4)


@pytest.mark.parametrize("scheme", ["NGP", "CIC", "TSC", "PCS"])
def test_deconvolve_mas_torch_matches_numpy(scheme):
    torch = pytest.importorskip("torch")
    N, L = 16, 10.0
    rng = np.random.default_rng(3)
    field = rng.standard_normal((N, N, N)).astype(np.float32)
    convolved = _convolve_with_window(field, L, scheme).astype(np.float32)

    ref = deconvolve_mas(convolved, L, scheme)
    out = deconvolve_mas(torch.as_tensor(convolved), L, scheme)

    assert isinstance(out, torch.Tensor)
    np.testing.assert_allclose(out.numpy(), ref, atol=1e-4)
