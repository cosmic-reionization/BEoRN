"""Compute backends for LPT — numpy (default), torch, and jax.

Each backend wraps the FFT, array creation, and random-number operations
needed by the LPT solver so that the same physics code runs on multi-core
CPU (numpy), GPU via PyTorch, or GPU/TPU via JAX without modification.

Parallelism summary
-------------------
NumpyBackend
    Uses ``scipy.fft`` (when available) with ``workers=-1`` — all CPU cores.
    Falls back to single-threaded ``numpy.fft`` if scipy is absent.

TorchBackend
    Uses ``torch.fft``.  Automatically targets ``'cuda'`` if a GPU is
    available, else ``'cpu'``.  Pass an explicit device string to override.
    All arrays are kept on the chosen device; ``to_numpy`` triggers a single
    CPU transfer at output time.

JaxBackend
    Uses ``jax.numpy.fft``.  JAX auto-dispatches to the first available
    accelerator (GPU/TPU).  All intermediate arrays live on-device; only
    ``to_numpy`` moves data back to the host.

Auto-detection priority (``backend='auto'``, opt-in — the solver default is
``'numpy'`` so GPU/float32 execution is never selected silently)
--------------------------------------------
1. ``JaxBackend()``            — if JAX is importable and has a GPU device
2. ``TorchBackend('cuda')``    — if PyTorch is importable and CUDA is present
3. ``TorchBackend('mps')``     — if PyTorch is importable and Apple MPS is
                                 present (float32)
4. ``NumpyBackend()``          — multi-core CPU fallback

Explicit instantiation
----------------------
Pass a backend *instance* directly to any LPT solver to override defaults::

    from beorn.lpt.backends import TorchBackend
    lpt = SecondOrderLPT(param, backend=TorchBackend(device='cuda:1'))
    lpt = SecondOrderLPT(param, backend='auto')
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class LPTBackend(ABC):
    """Abstract compute backend interface."""

    @abstractmethod
    def rfftn(self, x): ...

    @abstractmethod
    def irfftn(self, xk, shape: tuple): ...

    @abstractmethod
    def zeros(self, shape: tuple, dtype=np.float64): ...

    @abstractmethod
    def random_normal(self, shape: tuple, seed: int | None = None): ...

    @abstractmethod
    def to_numpy(self, x) -> np.ndarray:
        """Convert a backend array to a numpy array (host)."""
        ...

    @abstractmethod
    def as_array(self, x, dtype=None):
        """Move / convert *x* to a backend array on the appropriate device."""
        ...

    @abstractmethod
    def where(self, condition, x, y):
        """Element-wise selection: *x* where *condition* else *y*."""
        ...

    def sin(self, x):
        """Element-wise sine (numpy fallback; overridden on device backends)."""
        return np.sin(x)

    def cos(self, x):
        """Element-wise cosine (numpy fallback; overridden on device backends)."""
        return np.cos(x)


# ─────────────────────────────────────────────────────────────────────────────
# NumPy backend  (multi-core via scipy.fft)
# ─────────────────────────────────────────────────────────────────────────────

class NumpyBackend(LPTBackend):
    """CPU backend.  Uses ``scipy.fft`` with ``workers=-1`` when available.

    Args:
        workers: FFT worker threads.  ``-1`` (default) → all CPU cores.
    """

    def __init__(self, workers: int = -1):
        try:
            import scipy.fft as _sfft
            self._fft = _sfft
        except ImportError:
            self._fft = None  # fallback to single-threaded numpy.fft
        self._workers = workers

    @property
    def n_workers(self) -> int | str:
        """Effective worker count (``'all'`` when workers=-1 and scipy present)."""
        if self._fft is None:
            return 1
        if self._workers == -1:
            import os
            return os.cpu_count() or 1
        return self._workers

    def rfftn(self, x):
        if self._fft is not None:
            return self._fft.rfftn(np.asarray(x), workers=self._workers)
        return np.fft.rfftn(np.asarray(x))

    def irfftn(self, xk, shape):
        if self._fft is not None:
            return self._fft.irfftn(np.asarray(xk), s=shape,
                                    workers=self._workers)
        return np.fft.irfftn(np.asarray(xk), s=shape)

    def zeros(self, shape, dtype=np.float64):
        return np.zeros(shape, dtype=dtype)

    def random_normal(self, shape, seed=None):
        return np.random.default_rng(seed).standard_normal(shape)

    def to_numpy(self, x) -> np.ndarray:
        return np.asarray(x)

    def as_array(self, x, dtype=None):
        return np.asarray(x, dtype=dtype)

    def where(self, condition, x, y):
        return np.where(condition, x, y)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch backend  (GPU / CPU)
# ─────────────────────────────────────────────────────────────────────────────

class TorchBackend(LPTBackend):
    """GPU-capable backend via PyTorch.

    Args:
        device: Target device, e.g. ``'cuda'``, ``'cuda:0'``, ``'mps'``,
                ``'cpu'``.  Defaults to ``'cuda'`` when CUDA is available,
                else ``'cpu'``.
        dtype:  Floating precision: ``'float64'`` (default on CUDA/CPU) or
                ``'float32'``.  Opt-in only (issue #42, O5) — matches this
                project's policy of never changing precision/backend
                silently (see the JaxBackend x64 warning below): CUDA/CPU
                keep float64 unless ``dtype='float32'`` is passed explicitly,
                which roughly halves memory and doubles FFT/elementwise
                throughput at the cost of ~1e-6-level precision (empirically
                the same order as the existing MPS float32 path). Ignored on
                MPS, which never supports float64 and is always float32.

    Note:
        MPS (Apple Silicon) does not support float64. On MPS the backend
        automatically uses float32/complex64, which reduces precision.
        CUDA and CPU default to float64/complex128 unless ``dtype='float32'``
        is passed.
    """

    def __init__(self, device: str | None = None, dtype: str = 'float64'):
        try:
            import torch
            self._t = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for backend='torch'. "
                "Install with: pip install torch"
            )
        if device is None:
            device = 'cuda' if self._t.cuda.is_available() else 'cpu'
        self.device = device

        if dtype not in ('float32', 'float64'):
            raise ValueError(f"dtype must be 'float32' or 'float64'; got {dtype!r}.")

        # MPS does not support float64 — fall back to float32 regardless of
        # the requested dtype. Elsewhere, float64 is the unchanged default;
        # float32 is opt-in only, never silent (O5).
        if device == 'mps' or dtype == 'float32':
            self._fdtype = self._t.float32
            self._cdtype = self._t.complex64
        else:
            self._fdtype = self._t.float64
            self._cdtype = self._t.complex128

    def as_array(self, x, dtype=None):
        t = self._t
        if isinstance(x, t.Tensor):
            return x.to(self.device)
        arr = np.asarray(x) if dtype is None else np.asarray(x, dtype=dtype)
        if np.iscomplexobj(arr):
            return t.as_tensor(arr.astype(np.complex64 if self._cdtype == t.complex64
                                          else np.complex128),
                               dtype=self._cdtype).to(self.device)
        return t.as_tensor(arr.astype(np.float32 if self._fdtype == t.float32
                                      else np.float64),
                           dtype=self._fdtype).to(self.device)

    def to_numpy(self, x) -> np.ndarray:
        if isinstance(x, self._t.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def where(self, condition, x, y):
        t = self._t
        if not isinstance(condition, t.Tensor):
            condition = t.as_tensor(np.asarray(condition), dtype=t.bool,
                                    device=self.device)
        return t.where(condition, x, y)

    def sin(self, x):
        return self._t.sin(x)

    def cos(self, x):
        return self._t.cos(x)

    def rfftn(self, x):
        return self._t.fft.rfftn(self.as_array(x))

    def irfftn(self, xk, shape):
        if not isinstance(xk, self._t.Tensor):
            xk = self.as_array(xk)
        return self._t.fft.irfftn(xk, s=shape)

    def zeros(self, shape, dtype=np.float64):
        return self._t.zeros(shape, dtype=self._fdtype, device=self.device)

    def random_normal(self, shape, seed=None):
        if seed is not None:
            self._t.manual_seed(seed)
        return self._t.randn(shape, dtype=self._fdtype, device=self.device)


# ─────────────────────────────────────────────────────────────────────────────
# JAX backend  (GPU / TPU / CPU)
# ─────────────────────────────────────────────────────────────────────────────

class JaxBackend(LPTBackend):
    """GPU/TPU backend via JAX.  Auto-dispatches to the first available
    accelerator — no device argument needed.

    Args:
        dtype: Floating precision for internally-generated arrays
               (:meth:`zeros`, :meth:`random_normal`): ``'float64'``
               (default) or ``'float32'``.  Opt-in only (issue #42, O5) —
               with the default, requesting float64 while jax x64 is
               disabled falls back to jax's own float32 canonicalisation
               (triggering the warning below); ``dtype='float32'`` requests
               float32 explicitly and skips that warning, since it is no
               longer an accidental precision loss.
    """

    def __init__(self, dtype: str = 'float64'):
        try:
            import jax
            import jax.numpy as jnp
            self._jax = jax
            self._jnp = jnp
        except ImportError:
            raise ImportError(
                "JAX is required for backend='jax'. "
                "Install with: pip install jax"
            )
        if dtype not in ('float32', 'float64'):
            raise ValueError(f"dtype must be 'float32' or 'float64'; got {dtype!r}.")
        self._fdtype = jnp.float32 if dtype == 'float32' else jnp.float64

        # V6 (issue #42): never run float32 silently — jax defaults to
        # float32 unless x64 is enabled, unlike the numpy/torch-CUDA paths.
        # Only warn when float32 was NOT explicitly requested: an explicit
        # dtype='float32' is an opt-in (O5), not an accidental precision loss.
        if dtype == 'float64' and not jax.config.jax_enable_x64:
            import warnings
            warnings.warn(
                "JAX x64 mode is disabled — the jax LPT backend will compute "
                "in float32 (results differ from the float64 numpy default "
                "at the ~1e-6 level). Enable float64 with "
                "jax.config.update('jax_enable_x64', True) before creating "
                "arrays, or pass dtype='float32' to commit to float32 "
                "explicitly and silence this warning.",
                stacklevel=3,
            )

    def as_array(self, x, dtype=None):
        return self._jnp.asarray(x, dtype=dtype)

    def to_numpy(self, x) -> np.ndarray:
        return np.asarray(x)

    def where(self, condition, x, y):
        return self._jnp.where(condition, x, y)

    def sin(self, x):
        return self._jnp.sin(x)

    def cos(self, x):
        return self._jnp.cos(x)

    def rfftn(self, x):
        return self._jnp.fft.rfftn(self.as_array(x))

    def irfftn(self, xk, shape):
        return self._jnp.fft.irfftn(self.as_array(xk), s=shape).real

    def zeros(self, shape, dtype=None):
        return self._jnp.zeros(shape, dtype=self._fdtype if dtype is None else dtype)

    def random_normal(self, shape, seed=None):
        key = self._jax.random.PRNGKey(seed if seed is not None else 0)
        return self._jax.random.normal(key, shape=shape, dtype=self._fdtype)


# ─────────────────────────────────────────────────────────────────────────────
# Registry and factory
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[LPTBackend]] = {
    'numpy': NumpyBackend,
    'torch': TorchBackend,
    'jax':   JaxBackend,
}


def get_backend(name: str | LPTBackend = 'numpy',
                verbose: bool = False) -> LPTBackend:
    """Return a backend instance.

    Args:
        name:    One of ``'numpy'``, ``'torch'``, ``'jax'``, ``'auto'``, or an
                 existing :class:`LPTBackend` instance (returned unchanged).
        verbose: If ``True``, print the chosen backend (useful with
                 ``'auto'``).

    Auto-detection priority (``'auto'``):
        1. ``JaxBackend()``         — JAX importable + GPU device present
        2. ``TorchBackend('cuda')`` — PyTorch importable + CUDA present
        3. ``TorchBackend('mps')``  — PyTorch importable + Apple MPS (float32)
        4. ``NumpyBackend()``       — multi-core CPU fallback
    """
    if isinstance(name, LPTBackend):
        if verbose:
            _print_backend_info(name)
        return name

    if name == 'auto':
        backend = _auto_select()
    elif name in _REGISTRY:
        backend = _REGISTRY[name]()
    else:
        raise ValueError(
            f"Unknown backend {name!r}. "
            f"Choose from {list(_REGISTRY) + ['auto']}."
        )

    if verbose:
        _print_backend_info(backend)
    return backend


def _auto_select() -> LPTBackend:
    """Pick the best available backend: JAX (GPU/TPU) > Torch (CUDA/MPS) > NumPy."""
    try:
        import jax
        # d.platform returns 'cpu', 'gpu', or 'tpu' — standard JAX API
        if any(d.platform != 'cpu' for d in jax.devices()):
            return JaxBackend()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return TorchBackend(device='cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return TorchBackend(device='mps')
    except Exception:
        pass
    return NumpyBackend()


def _print_backend_info(backend: LPTBackend) -> None:
    name = type(backend).__name__
    if isinstance(backend, NumpyBackend):
        engine = "scipy.fft" if backend._fft is not None else "numpy.fft"
        workers = backend.n_workers
        print(f"[BEoRN LPT] backend → {name}  ({engine}, {workers} workers)")
    elif isinstance(backend, TorchBackend):
        prec = "float32" if backend._fdtype == backend._t.float32 else "float64"
        print(f"[BEoRN LPT] backend → {name}  (device={backend.device}, {prec})")
    elif isinstance(backend, JaxBackend):
        try:
            devices = backend._jax.devices()
            dev_str = ', '.join(str(d) for d in devices[:3])
            print(f"[BEoRN LPT] backend → {name}  (devices: {dev_str})")
        except Exception:
            print(f"[BEoRN LPT] backend → {name}")
    else:
        print(f"[BEoRN LPT] backend → {name}")
