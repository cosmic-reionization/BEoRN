"""GPU-specific tests for issue #48's jax/torch dispatch.

Skipped entirely when no GPU is visible (regular test_particle_mapping*.py
files already cover CPU jax/torch dispatch). These specifically place arrays
on the GPU device and confirm results match the numpy reference and stay
device-resident (no silent host round-trip).
"""
import numpy as np
import pytest

from beorn.particle_mapping import deconvolve_mas, coarsen_field, upsample_field_fourier
from beorn.power_spectrum import power_spectrum_1d


def _jax_gpu_device():
    try:
        import jax
        gpus = [d for d in jax.devices() if d.platform != 'cpu']
        return gpus[0] if gpus else None
    except ImportError:
        return None


def _torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


requires_jax_gpu = pytest.mark.skipif(
    _jax_gpu_device() is None, reason="no jax GPU device available"
)
requires_torch_cuda = pytest.mark.skipif(
    not _torch_cuda_available(), reason="no CUDA device available for torch"
)


# ── deconvolve_mas ────────────────────────────────────────────────────────────

@requires_jax_gpu
@pytest.mark.parametrize("scheme", ["NGP", "CIC", "TSC", "PCS"])
def test_deconvolve_mas_jax_gpu(scheme):
    import jax
    from beorn.particle_mapping.window import _mas_window

    N, L = 16, 10.0
    rng = np.random.default_rng(20)
    field = rng.standard_normal((N, N, N)).astype(np.float32)
    field_k = np.fft.rfftn(field) * _mas_window(L, N, scheme)
    convolved = np.fft.irfftn(field_k, s=(N, N, N)).astype(np.float32)

    ref = deconvolve_mas(convolved, L, scheme)

    gpu = _jax_gpu_device()
    field_gpu = jax.device_put(convolved, gpu)
    out = deconvolve_mas(field_gpu, L, scheme)

    assert out.devices() == {gpu}
    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-4)


@requires_torch_cuda
@pytest.mark.parametrize("scheme", ["NGP", "CIC", "TSC", "PCS"])
def test_deconvolve_mas_torch_cuda(scheme):
    import torch
    from beorn.particle_mapping.window import _mas_window

    N, L = 16, 10.0
    rng = np.random.default_rng(21)
    field = rng.standard_normal((N, N, N)).astype(np.float32)
    field_k = np.fft.rfftn(field) * _mas_window(L, N, scheme)
    convolved = np.fft.irfftn(field_k, s=(N, N, N)).astype(np.float32)

    ref = deconvolve_mas(convolved, L, scheme)

    field_cuda = torch.as_tensor(convolved, device='cuda')
    out = deconvolve_mas(field_cuda, L, scheme)

    assert out.is_cuda
    np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-4)


# ── coarsen_field / upsample_field_fourier ────────────────────────────────────

@requires_jax_gpu
def test_coarsen_and_upsample_jax_gpu():
    import jax
    rng = np.random.default_rng(22)
    field = rng.standard_normal((16, 16, 16)).astype(np.float32)

    ref_up = upsample_field_fourier(field, 4)
    ref_down = coarsen_field(ref_up, 4)

    gpu = _jax_gpu_device()
    field_gpu = jax.device_put(field, gpu)
    up_gpu = upsample_field_fourier(field_gpu, 4)
    down_gpu = coarsen_field(up_gpu, 4)

    assert up_gpu.devices() == {gpu}
    assert down_gpu.devices() == {gpu}
    np.testing.assert_allclose(np.asarray(up_gpu), ref_up, atol=1e-3)
    np.testing.assert_allclose(np.asarray(down_gpu), ref_down, atol=1e-4)


@requires_torch_cuda
def test_coarsen_and_upsample_torch_cuda():
    import torch
    rng = np.random.default_rng(23)
    field = rng.standard_normal((16, 16, 16)).astype(np.float32)

    ref_up = upsample_field_fourier(field, 4)
    ref_down = coarsen_field(ref_up, 4)

    field_cuda = torch.as_tensor(field, device='cuda')
    up_cuda = upsample_field_fourier(field_cuda, 4)
    down_cuda = coarsen_field(up_cuda, 4)

    assert up_cuda.is_cuda
    assert down_cuda.is_cuda
    np.testing.assert_allclose(up_cuda.cpu().numpy(), ref_up, atol=1e-3)
    np.testing.assert_allclose(down_cuda.cpu().numpy(), ref_down, atol=1e-4)


# ── power_spectrum_1d ─────────────────────────────────────────────────────────

@requires_jax_gpu
def test_power_spectrum_1d_jax_gpu():
    import jax
    N, L = 16, 10.0
    rng = np.random.default_rng(24)
    field = (np.abs(rng.standard_normal((N, N, N))) + 1.0).astype(np.float32)

    Pk_ref, bins_ref, kny_ref = power_spectrum_1d(field, L, kbins=20,
                                                   mass_assignment='CIC', deconvolve=True)

    gpu = _jax_gpu_device()
    field_gpu = jax.device_put(field, gpu)
    Pk, bins, kny = power_spectrum_1d(field_gpu, L, kbins=20,
                                       mass_assignment='CIC', deconvolve=True)

    np.testing.assert_allclose(np.asarray(Pk), Pk_ref, rtol=1e-3)
    assert kny == pytest.approx(kny_ref)


@requires_torch_cuda
def test_power_spectrum_1d_torch_cuda():
    import torch
    N, L = 16, 10.0
    rng = np.random.default_rng(25)
    field = (np.abs(rng.standard_normal((N, N, N))) + 1.0).astype(np.float32)

    Pk_ref, bins_ref, kny_ref = power_spectrum_1d(field, L, kbins=20,
                                                   mass_assignment='CIC', deconvolve=True)

    field_cuda = torch.as_tensor(field, device='cuda')
    Pk, bins, kny = power_spectrum_1d(field_cuda, L, kbins=20,
                                       mass_assignment='CIC', deconvolve=True)

    np.testing.assert_allclose(np.asarray(Pk.cpu() if hasattr(Pk, 'cpu') else Pk),
                                Pk_ref, rtol=1e-3)
    assert kny == pytest.approx(kny_ref)
