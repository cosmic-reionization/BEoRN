"""Unit tests for beorn.power_spectrum (issue #48)."""
import numpy as np
import pytest
import tools21cm as t2c

from beorn.power_spectrum import power_spectrum_1d
from beorn.particle_mapping import k_nyquist


def _field(N=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, N, N))


def test_forwards_kwargs_to_tools21cm_when_not_deconvolving():
    N, L = 16, 10.0
    field = _field(N)

    Pk_ref, bins_ref = t2c.power_spectrum.power_spectrum_1d(field, box_dims=L, kbins=20)
    Pk, bins, kny = power_spectrum_1d(field, L, kbins=20)

    np.testing.assert_allclose(Pk, Pk_ref)
    np.testing.assert_allclose(bins, bins_ref)
    assert kny == pytest.approx(k_nyquist(L, N))


def test_return_n_modes_still_forwarded():
    N, L = 16, 10.0
    field = _field(N, seed=2)
    result = power_spectrum_1d(field, L, kbins=20, return_n_modes=True)
    assert len(result) == 4  # Pk, bins, n_modes, k_nyquist


def test_deconvolve_requires_mass_assignment():
    N, L = 16, 10.0
    field = _field(N, seed=3)
    with pytest.raises(ValueError, match="deconvolve=True requires mass_assignment"):
        power_spectrum_1d(field, L, deconvolve=True)


def test_deconvolve_changes_result_vs_plain():
    N, L = 16, 10.0
    field = np.abs(_field(N, seed=4)) + 1.0  # positive-ish "density-like" field

    Pk_plain, bins_plain, _ = power_spectrum_1d(field, L, kbins=20)
    Pk_deconv, bins_deconv, _ = power_spectrum_1d(field, L, kbins=20,
                                                   mass_assignment='CIC', deconvolve=True)

    np.testing.assert_allclose(bins_plain, bins_deconv)
    assert not np.allclose(Pk_plain, Pk_deconv)


def test_k_nyquist_scalar_box_dims():
    N, L = 32, 50.0
    field = _field(N, seed=5)
    *_, kny = power_spectrum_1d(field, L, kbins=10)
    assert kny == pytest.approx(k_nyquist(L, N))


# ── jax/torch backend dispatch ────────────────────────────────────────────────

def test_power_spectrum_1d_jax_matches_numpy():
    jnp = pytest.importorskip("jax.numpy")
    N, L = 16, 10.0
    field = np.abs(_field(N, seed=6)).astype(np.float32) + 1.0

    Pk_ref, bins_ref, kny_ref = power_spectrum_1d(field, L, kbins=20,
                                                   mass_assignment='CIC', deconvolve=True)
    Pk, bins, kny = power_spectrum_1d(jnp.asarray(field), L, kbins=20,
                                       mass_assignment='CIC', deconvolve=True)

    np.testing.assert_allclose(np.asarray(Pk), Pk_ref, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(bins), bins_ref, rtol=1e-4)
    assert kny == pytest.approx(kny_ref)


def test_power_spectrum_1d_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    N, L = 16, 10.0
    field = np.abs(_field(N, seed=7)).astype(np.float32) + 1.0

    Pk_ref, bins_ref, kny_ref = power_spectrum_1d(field, L, kbins=20,
                                                   mass_assignment='CIC', deconvolve=True)
    Pk, bins, kny = power_spectrum_1d(torch.as_tensor(field), L, kbins=20,
                                       mass_assignment='CIC', deconvolve=True)

    np.testing.assert_allclose(np.asarray(Pk), Pk_ref, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(bins), bins_ref, rtol=1e-4)
    assert kny == pytest.approx(kny_ref)
