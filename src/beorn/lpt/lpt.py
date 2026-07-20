"""Lagrangian Perturbation Theory (LPT) displacement field solver.

Hierarchy
---------
LPTBase  (ABC)
├── ZeldovichApproximation  — 1LPT (fastest)
├── SecondOrderLPT          — 2LPT (balanced accuracy)
└── ThirdOrderLPT           — 3LPT (full, both source terms)

All classes share the same interface::

    lpt = SecondOrderLPT(parameters, seed=42)
    psi_x, psi_y, psi_z = lpt.get_displacement(z=10.0)
    delta = lpt.get_density(z=10.0)

IC modes
--------
fixed=True (default)
    Fixed-amplitude ICs (Angulo & Pontzen 2016): each k-mode amplitude is set
    to exactly sqrt(P(k)), only the phases are random.  Eliminates mode-to-mode
    Rayleigh scatter and strongly suppresses cosmic variance in one-point stats.

seed >= 0
    Normal realisation drawn with the given seed.

seed < 0
    Paired realisation: uses abs(seed) as the RNG seed but negates the noise
    field, flipping all phases by π.  Together with the seed >= 0 run, forms a
    paired simulation (Angulo & Pontzen 2016).

External GRF
------------
Pass ``grf`` to :meth:`generate_initial_conditions` to supply your own noise:

* shape ``(N, N, N)`` — real-space white-noise array; FFT is applied and then
  P(k) normalisation is imposed (use this to inject a 21cmFAST noise cube).
* shape ``(N, N, N//2+1)`` complex — already-weighted δ(k); used as-is,
  skipping both the RNG step and P(k) normalisation.

Parallelism
-----------
All FFTs are routed through the active backend, so:

* ``backend='numpy'``  — multi-core CPU via ``scipy.fft`` (``workers=-1``); default
* ``backend='torch'``  — GPU via PyTorch CUDA (auto-detected)
* ``backend='jax'``    — GPU/TPU via JAX (auto-detected)
* ``backend='auto'``   — opt-in: picks best available, JAX GPU > Torch CUDA/MPS > NumPy
  (note MPS runs in float32 — results differ from the numpy default at the
  1e-6 level, so GPU execution is never selected silently)

Intermediate arrays stay on the backend device throughout
:meth:`get_displacement`; only the final Ψ components are transferred back
to CPU numpy at output time.

Units
-----
- Positions / displacements in Mpc/h (same as parameters.simulation.Lbox).
- k-vectors in h/Mpc.
- Power spectrum in (Mpc/h)³.
"""
from __future__ import annotations

from abc import ABC
import numpy as np

from ..cosmo import D, hubble
from .backends import get_backend, LPTBackend, NumpyBackend
from .linear_power import PowerSpectrum, get_power_spectrum


# ============================================================
# Helpers
# ============================================================

def _kvectors(N: int, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wavenumber arrays for an rfftn-shaped grid (N, N, N//2+1).

    Returns:
        kx, ky, kz — broadcastable shapes (N,1,1), (1,N,1), (1,1,N//2+1)
                     (h/Mpc); every caller only combines them into ``k2``,
                     so keeping them unbroadcast avoids materialising three
                     full (N, N, N//2+1) grids for no reason (issue #42, O3).
        k2         — full (N, N, N//2+1): kx**2 + ky**2 + kz**2.
    """
    dk = 2.0 * np.pi / L
    kx = (np.fft.fftfreq(N, d=1.0 / N) * dk)[:, None, None]
    ky = (np.fft.fftfreq(N, d=1.0 / N) * dk)[None, :, None]
    kz = (np.fft.rfftfreq(N, d=1.0 / N) * dk)[None, None, :]
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    return kx, ky, kz, k2


# ============================================================
# Abstract base
# ============================================================

class LPTBase(ABC):
    """Abstract base class for all LPT solvers.

    Args:
        parameters:  BEoRN Parameters object.
        ps_method:   Power spectrum method passed to :func:`get_power_spectrum`.
                     Default ``'eisenstein_hu'``.
        backend:     Compute backend: ``'numpy'`` (default), ``'torch'``,
                     ``'jax'``, ``'auto'``, or an :class:`LPTBackend` instance.
                     ``'auto'`` picks JAX GPU > Torch CUDA > NumPy.
        seed:        RNG seed.  Positive → normal realisation; negative →
                     paired realisation using ``abs(seed)`` with negated noise.
                     Default ``42``.
        fixed:       If ``True`` (default), use fixed-amplitude ICs: each
                     k-mode amplitude is set to exactly sqrt(P(k)) and only
                     the phases are randomised.  Reduces cosmic variance.
        verbose:     If ``True`` (default), print the chosen backend and its
                     parallelism info on construction.
        f1_method:   How to compute the growth rate f₁ = dlnD₁/dlna:
                     ``'fd'`` (default) — central finite difference through
                     ``beorn.cosmo.D`` (legacy behaviour); ``'autodiff'`` —
                     exact autodiff through the fixed-node growth integral
                     (``beorn.cosmo.growth_rate``, jax or torch required).
        **ps_kwargs: Extra keyword arguments forwarded to the power spectrum
                     constructor (e.g. ``wiggle=True`` for the E&H with-wiggle
                     fit, or ``ps_file='Pk_camb.dat'`` for the Boltzmann
                     solver).
        power_spectrum: Pre-built :class:`~beorn.lpt.linear_power.PowerSpectrum`
                     instance to use directly instead of constructing one from
                     ``ps_method``/``ps_kwargs`` (issue #42, O10) — pass this
                     to share a single instance with other consumers (e.g.
                     :class:`~beorn.lpt.chmf.CHMF`) instead of each building
                     its own. ``ps_method``/``ps_kwargs`` are ignored when set.

    Performance note
    ----------------
    The z-independent k-space source terms (δ(k) and the 2LPT/3LPT source
    fields) are computed once per IC realisation and cached on the backend
    device, together with the k-vector grids.  After the first call, every
    :meth:`get_displacement` / :meth:`get_velocity` at any redshift costs
    exactly **3 inverse FFTs** — the growth factors enter as scalar
    coefficients of the cached sources.
    """

    def __init__(
        self,
        parameters,
        ps_method: str = 'eisenstein_hu',
        backend: str | LPTBackend = 'numpy',
        seed: int = 42,
        fixed: bool = True,
        verbose: bool = True,
        f1_method: str = 'fd',
        power_spectrum: PowerSpectrum | None = None,
        **ps_kwargs,
    ):
        self.parameters = parameters
        self.seed = seed
        self.fixed = fixed
        self.verbose = verbose
        self.f1_method = f1_method
        self._backend: LPTBackend = get_backend(backend, verbose=verbose)
        # issue #42, O10: accept a pre-built PowerSpectrum so callers that
        # also need one elsewhere (e.g. LPTHaloLoader's CHMF) can share a
        # single instance instead of each paying its own A_s normalisation.
        # ps_method/ps_kwargs are ignored when power_spectrum is given.
        self.power_spectrum: PowerSpectrum = (
            power_spectrum if power_spectrum is not None
            else get_power_spectrum(ps_method, parameters, **ps_kwargs)
        )
        self._delta_k: np.ndarray | None = None  # cached IC realisation
        self._k_cache = None            # backend k-vectors, built once
        self._dk_dev = None             # δ(k) on the backend device
        self._sources_k_cache = None    # z-independent k-space source terms

    # ------------------------------------------------------------------
    # Grid geometry
    # ------------------------------------------------------------------

    @property
    def N(self) -> int:
        return self.parameters.simulation.Ncell

    @property
    def L(self) -> float:
        return self.parameters.simulation.Lbox

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    def generate_initial_conditions(
        self,
        seed: int | None = None,
        grf: np.ndarray | None = None,
        n_k_nodes: int | None = 1000,
    ) -> np.ndarray:
        """Build and cache δ(k) consistent with P(k).

        The field is fixed at z = 0; growth-factor rescaling to the target
        redshift is applied inside :meth:`get_displacement`.

        Args:
            seed: Override the instance seed for this call only.  Negative
                  values trigger the paired-simulation mode (see class docs).
            grf:  External Gaussian random field to use instead of drawing
                  new noise.  Two accepted shapes:

                  * ``(N, N, N)`` — real-space white-noise cube.  Will be
                    FFT'd and then P(k)-normalised.  Use this to inject a
                    noise cube from e.g. 21cmFAST for structure-by-structure
                    comparison.
                  * ``(N, N, N//2+1)`` complex — pre-computed δ(k).  Used
                    as-is; P(k) normalisation and the ``fixed`` flag are both
                    skipped.
            n_k_nodes: Resolution of the 1-D log-k table P(k) is evaluated on
                  before log-log interpolation onto the full 3-D k-grid
                  (issue #42, O8) — a closed-form transfer function is cheap
                  per call, but ~N^3/2 elementwise evaluations still cost far
                  more than a ~1000-node table + interpolation. ``None``
                  disables the table and evaluates P(k) directly on the full
                  grid (validation reference / opt-out).

        Returns:
            delta_k — complex array of shape (N, N, N//2+1).
        """
        N, L = self.N, self.L
        V = L ** 3

        # New realisation → drop device caches derived from the old δ(k)
        self._dk_dev = None
        self._sources_k_cache = None

        # Amplitude grid (always built in numpy — cheap scalar ops)
        kx, ky, kz, k2 = _kvectors(N, L)
        k_safe = np.where(k2 == 0, 1.0, np.sqrt(k2))
        if n_k_nodes is None:
            Pk = self.power_spectrum.P(k_safe, z=0.0)
        else:
            # P(k) is smooth and close to a power law between nodes, so
            # log-log linear interpolation from a 1-D table is accurate to
            # far better than 1e-4 relative — same pattern already used for
            # sigma^2(M) and the kappa(T) coupling tables elsewhere.
            k_lo, k_hi = float(k_safe[k2 > 0].min()), float(k_safe.max())
            k_1d = np.logspace(np.log10(k_lo), np.log10(k_hi), n_k_nodes)
            Pk_1d = np.asarray(self.power_spectrum.P(k_1d, z=0.0), dtype=float)
            log_Pk = np.interp(np.log(k_safe), np.log(k_1d), np.log(Pk_1d))
            Pk = np.exp(log_Pk)
        Pk[k2 == 0] = 0.0
        amplitude = np.sqrt(Pk * N ** 3 / V)

        # ── External GRF path ─────────────────────────────────────────────
        if grf is not None:
            grf = np.asarray(grf)
            if grf.shape == (N, N, N):
                noise_k = np.fft.rfftn(grf)
                self._delta_k = noise_k * amplitude
            elif grf.shape == (N, N, N // 2 + 1):
                self._delta_k = grf.astype(complex)
            else:
                raise ValueError(
                    f"grf shape {grf.shape} must be (N,N,N)={(N,N,N)} or "
                    f"(N,N,N//2+1)={(N, N, N//2+1)}"
                )
            return self._delta_k

        # ── Internal noise path ───────────────────────────────────────────
        _seed = seed if seed is not None else self.seed

        # Paired mode: seed < 0 → abs(seed) with negated noise
        paired = (_seed < 0)
        actual_seed = abs(_seed) if paired else _seed

        noise = self._backend.random_normal((N, N, N), seed=actual_seed)
        noise_k = self._backend.to_numpy(self._backend.rfftn(noise))

        if paired:
            noise_k = -noise_k

        if self.fixed:
            # Fixed-amplitude ICs: set |noise_k| = sqrt(N³) for every mode,
            # keeping only the random phase.  rfftn of N(0,1) noise has
            # E[|noise_k|²] = N³, so this preserves the correct power while
            # eliminating mode-to-mode Rayleigh scatter.
            abs_nk = np.abs(noise_k)
            target = np.sqrt(float(N ** 3))
            noise_k = np.where(abs_nk > 0, noise_k / abs_nk * target, 0.0)

        self._delta_k = noise_k * amplitude
        return self._delta_k

    @property
    def delta_k(self) -> np.ndarray:
        """Return cached δ(k), generating with the instance seed if needed."""
        if self._delta_k is None:
            self.generate_initial_conditions()
        return self._delta_k

    # ------------------------------------------------------------------
    # Growth factor helpers
    # ------------------------------------------------------------------

    def _D1(self, z: float) -> float:
        """First-order linear growth factor D(z), normalised to D(0) = 1."""
        a = 1.0 / (1.0 + z)
        return D(a, self.parameters) / D(1.0, self.parameters)

    def _D2(self, z: float) -> float:
        """D₂(z) ≈ −3/7 D₁²(z) (accurate to < 1% at z > 1)."""
        return -3.0 / 7.0 * self._D1(z) ** 2

    def _f1(self, z: float) -> float:
        """Linear growth rate f₁ = d ln D₁ / d ln a.

        ``f1_method='fd'`` (default): central finite difference through
        ``beorn.cosmo.D`` — legacy behaviour, ~1e-3–1e-4 accuracy floor.
        ``f1_method='autodiff'``: exact autodiff of the fixed-node growth
        integral via :func:`beorn.cosmo.growth_rate` (jax preferred, torch
        fallback).
        """
        a = 1.0 / (1.0 + z)
        if self.f1_method == 'autodiff':
            from ..cosmo.differentiable import growth_rate
            Om = self.parameters.cosmology.Om
            for be in ('jax', 'torch'):
                try:
                    return float(growth_rate(a, Om, backend=be))
                except ImportError:
                    continue
            raise ImportError(
                "f1_method='autodiff' requires jax or torch; neither is installed.")
        da = a * 1e-4
        D0 = D(1.0, self.parameters)
        dD_da = (D(a + da, self.parameters) - D(a - da, self.parameters)) / (2.0 * da * D0)
        return a / self._D1(z) * dD_da

    # ------------------------------------------------------------------
    # Shared k-space helpers (used by all subclasses)
    # ------------------------------------------------------------------

    def _setup_k(self):
        """Compute k-vectors + inv_k2 as numpy arrays.

        Returns:
            kx, ky, kz — broadcastable shapes (N,1,1), (1,N,1), (1,1,N//2+1)
                         in units h/Mpc (saves 3 full grids of memory).
            inv_k2     — full (N, N, N//2+1): 1/k² with the k=0 mode set to 0
                         (avoids divergence *and* zeros the DC displacement).
        """
        N, L = self.N, self.L
        dk = 2.0 * np.pi / L
        kx = (np.fft.fftfreq(N, d=1.0 / N) * dk)[:, None, None]
        ky = (np.fft.fftfreq(N, d=1.0 / N) * dk)[None, :, None]
        kz = (np.fft.rfftfreq(N, d=1.0 / N) * dk)[None, None, :]
        k2 = kx ** 2 + ky ** 2 + kz ** 2
        inv_k2 = np.where(k2 == 0, 0.0, 1.0 / k2)
        return kx, ky, kz, inv_k2

    def _setup_k_backend(self):
        """k-vectors and inv_k2 as backend arrays, built once and cached."""
        if self._k_cache is None:
            b = self._backend
            kx, ky, kz, inv_k2 = self._setup_k()
            self._k_cache = (b.as_array(kx), b.as_array(ky), b.as_array(kz),
                             b.as_array(inv_k2))
        return self._k_cache

    def _dk_backend(self):
        """δ(k) on the backend device, uploaded once per realisation."""
        if self._dk_dev is None:
            self._dk_dev = self._backend.as_array(self.delta_k)
        return self._dk_dev

    def _phi1_derivs_b(self, dk, kx, ky, kz, ik2):
        """Second derivatives of φ₁ as backend arrays (real-valued).

        φ₁,ᵢⱼ(k) = −kᵢkⱼ/k² · δ(k).  The DC component is zero via *ik2*.

        Args:
            dk:           δ(k) as a backend array.
            kx, ky, kz:  k-component arrays (backend).
            ik2:          1/k² array with k=0 set to 0 (backend).

        Returns:
            dict with keys ``'xx', 'yy', 'zz', 'xy', 'xz', 'yz'``, each a
            real backend array of shape (N, N, N).
        """
        b = self._backend
        N = self.N

        def _deriv(ki, kj):
            return b.irfftn(-ki * kj * ik2 * dk, (N, N, N))

        return {
            'xx': _deriv(kx, kx), 'yy': _deriv(ky, ky), 'zz': _deriv(kz, kz),
            'xy': _deriv(kx, ky), 'xz': _deriv(kx, kz), 'yz': _deriv(ky, kz),
        }

    # ------------------------------------------------------------------
    # Cached z-independent sources + growth-coefficient combination
    # ------------------------------------------------------------------

    def _sources_k(self) -> list:
        """z-independent k-space source terms, computed once and cached.

        Order n term is defined so that the order-n displacement is
        Ψ⁽ⁿ⁾ᵢ(z) = Dₙ(z) · ℱ⁻¹[ i kᵢ/k² · sourceₙ(k) ].
        """
        if self._sources_k_cache is None:
            self._sources_k_cache = self._compute_sources_k()
        return self._sources_k_cache

    def _compute_sources_k(self) -> list:
        """1LPT default: the single source is δ(k) itself."""
        return [self._dk_backend()]

    def _growth_coeffs(self, z: float) -> list[float]:
        """Displacement growth factors, one per source term."""
        return [self._D1(z)]

    def _velocity_orders(self) -> list[float]:
        """Effective LPT order n per source term (EdS: fₙ ≈ n·f₁)."""
        return [1.0]

    def _velocity_coeffs(self, z: float) -> list[float]:
        """Velocity coefficients a·H/h·fₙ·Dₙ, one per source term (km/s)."""
        a = 1.0 / (1.0 + z)
        Hz = hubble(z, self.parameters) / self.parameters.cosmology.h0
        f1 = self._f1(z)
        return [a * Hz * n * f1 * Dn
                for n, Dn in zip(self._velocity_orders(), self._growth_coeffs(z))]

    def _combine(self, coeffs: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Σₙ cₙ · ℱ⁻¹[i kᵢ/k² · sourceₙ(k)] via one combined k-field → 3 iFFTs."""
        b = self._backend
        N = self.N
        kx, ky, kz, ik2 = self._setup_k_backend()
        sources = self._sources_k()
        ck = coeffs[0] * sources[0]
        for c, s in zip(coeffs[1:], sources[1:]):
            ck = ck + c * s
        ck = ik2 * ck
        return (
            b.to_numpy(b.irfftn(1j * kx * ck, (N, N, N))),
            b.to_numpy(b.irfftn(1j * ky * ck, (N, N, N))),
            b.to_numpy(b.irfftn(1j * kz * ck, (N, N, N))),
        )

    # ------------------------------------------------------------------
    # Public interface (shared by all orders)
    # ------------------------------------------------------------------

    def get_displacement(self, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the LPT displacement field (Ψx, Ψy, Ψz) at redshift z.

        Each component has shape (N, N, N) in units of Mpc/h.
        Lagrangian positions q on a regular grid are displaced as:
            x = q + Ψ(q)

        Costs 3 inverse FFTs after the first call (sources are cached).
        """
        return self._combine(self._growth_coeffs(z))

    def get_velocity(self, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Peculiar velocity (vx, vy, vz) at redshift z in km/s.

        v = a(z) H(z)/h Σₙ fₙ(z) Dₙ(z) Ψ⁽ⁿ⁾, with the EdS approximation
        fₙ ≈ n·f₁ (accurate to < 1 % at z > 2).  Reuses the cached
        displacement sources, so it also costs just 3 inverse FFTs.

        Returns:
            (vx, vy, vz) — each shape (N, N, N), units km/s.
        """
        return self._combine(self._velocity_coeffs(z))

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_positions(self, z: float) -> np.ndarray:
        """Displaced particle positions at redshift z.

        Returns:
            Array of shape (N³, 3) with positions in Mpc/h, periodic in [0, L).
        """
        N, L = self.N, self.L
        psi_x, psi_y, psi_z = self.get_displacement(z)

        cell = L / N
        q1d = (np.arange(N) + 0.5) * cell

        # Broadcast the 1-D Lagrangian grid directly against the displacement
        # fields instead of materialising 3 full (N,N,N) qx/qy/qz arrays via
        # np.meshgrid first (issue #42, O7) — same result, ~3 fewer N^3
        # float64 allocations (e.g. ~3.2 GB saved at N=512).
        x = (q1d[:, None, None] + psi_x) % L
        y = (q1d[None, :, None] + psi_y) % L
        z_pos = (q1d[None, None, :] + psi_z) % L

        positions = np.stack([x.ravel(), y.ravel(), z_pos.ravel()], axis=-1)
        return positions.astype(np.float32)

    def get_density(self, z: float, mass_assignment: str = 'CIC') -> np.ndarray:
        """Matter overdensity δ(x) at redshift z via particle painting.

        Args:
            z: Redshift.
            mass_assignment: ``'NGP'``, ``'CIC'`` (default), ``'TSC'``, or
                ``'PCS'`` — forwarded to
                :func:`~beorn.particle_mapping.map_particles_to_mesh`.

        Returns:
            delta — shape (N, N, N), mean-zero overdensity.
        """
        from ..particle_mapping import map_particles_to_mesh

        N, L = self.N, self.L
        mesh = np.zeros((N, N, N), dtype=np.float32)
        positions = self.get_positions(z)
        map_particles_to_mesh(mesh, L, positions, mass_assignment=mass_assignment)
        mean = mesh.mean()
        if mean > 0:
            mesh = mesh / mean - 1.0
        return mesh

    def get_linear_density(self, z: float, R_tophat: float | None = None) -> np.ndarray:
        """Linear overdensity δ(x) at redshift z from the k-space IC field.

        Unlike :meth:`get_density`, this method does **not** use CIC particle
        painting — it computes δ(x) = IRFFT[D₁(z) × δ(k)] directly, giving a
        clean Gaussian field free of shot noise.  When ``R_tophat`` is given, a
        real-space top-hat window W(kR) is applied before the transform, and the
        variance of the returned field equals σ²(R_tophat, z).

        This is the preferred density field for CHMF conditioning, because EPS
        self-consistency requires the conditioning field to be Gaussian with
        variance σ²(M_env).

        Args:
            z:        Redshift.
            R_tophat: If provided, apply a top-hat window of radius R_tophat
                      (Mpc/h) to the k-space field before inverting.

        Returns:
            delta — real-valued array of shape (N, N, N).
        """
        N, L = self.N, self.L
        D1 = self._D1(z)

        # Non-numpy backends: FFT on the device using the cached δ(k) and
        # broadcastable k-grids; one host transfer at output time (G3).
        # The numpy default keeps the legacy np.fft path byte-identical.
        if not isinstance(self._backend, NumpyBackend):
            b = self._backend
            dkz = D1 * self._dk_backend()
            if R_tophat is not None:
                kx, ky, kz, _ = self._setup_k_backend()
                k2 = kx ** 2 + ky ** 2 + kz ** 2
                kR = b.where(k2 > 0, k2, 1.0) ** 0.5 * R_tophat
                W = b.where(
                    kR < 1e-3,
                    1.0 - kR ** 2 / 10.0 + kR ** 4 / 280.0,
                    3.0 * (b.sin(kR) - kR * b.cos(kR)) / kR ** 3,
                )
                W = b.where(k2 > 0, W, 1.0)  # preserve the DC (mean = 0) mode
                dkz = dkz * W
            return b.to_numpy(b.irfftn(dkz, (N, N, N))).astype(np.float32)

        kx, ky, kz, k2 = _kvectors(N, L)

        delta_kz = D1 * self.delta_k  # (N, N, N//2+1) complex

        if R_tophat is not None:
            k = np.sqrt(np.where(k2 == 0, 1.0, k2))
            kR = k * R_tophat
            W = np.where(
                kR < 1e-3,
                1.0 - kR ** 2 / 10.0 + kR ** 4 / 280.0,
                3.0 * (np.sin(kR) - kR * np.cos(kR)) / kR ** 3,
            )
            W[k2 == 0] = 1.0  # preserve the DC (mean = 0) mode
            delta_kz = delta_kz * W

        return np.fft.irfftn(delta_kz, s=(N, N, N)).astype(np.float32)


# ============================================================
# 1LPT — Zel'dovich Approximation
# ============================================================

class ZeldovichApproximation(LPTBase):
    """First-order Lagrangian Perturbation Theory (Zel'dovich Approximation).

    Displacement: Ψ⁽¹⁾ = ℱ⁻¹[ i k/k² · δ(k) ] scaled by D₁(z).

    All FFTs run on the active backend device (GPU if available); the base
    class defaults implement 1LPT, so nothing to override here.
    """


# ============================================================
# 2LPT
# ============================================================

class SecondOrderLPT(LPTBase):
    """Second-order Lagrangian Perturbation Theory.

    Total displacement: Ψ = D₁ Ψ⁽¹⁾ + D₂ Ψ⁽²⁾

    The 2LPT source is:
        Δ⁽²⁾ = Σᵢ<ⱼ [ φ₁,ᵢᵢ φ₁,ⱼⱼ − (φ₁,ᵢⱼ)² ]
    where φ₁,ᵢⱼ ≡ ∂²φ₁/∂xᵢ∂xⱼ are the second-derivative fields of the
    1LPT potential φ₁ (with ∇²φ₁ = δ).

    The 2LPT source field Δ⁽²⁾(k) is z-independent; it is computed once,
    cached on the backend device, and recombined with (D₁, D₂) per redshift.
    """

    def _compute_sources_k(self) -> list:
        b = self._backend
        kx, ky, kz, ik2 = self._setup_k_backend()
        dk = self._dk_backend()
        d = self._phi1_derivs_b(dk, kx, ky, kz, ik2)
        source2 = (
            d['xx'] * d['yy'] - d['xy'] ** 2
            + d['xx'] * d['zz'] - d['xz'] ** 2
            + d['yy'] * d['zz'] - d['yz'] ** 2
        )
        return [dk, b.rfftn(source2)]

    def _growth_coeffs(self, z: float) -> list[float]:
        return [self._D1(z), self._D2(z)]

    def _velocity_orders(self) -> list[float]:
        return [1.0, 2.0]   # f₂ ≈ 2f₁ in EdS


# ============================================================
# 3LPT
# ============================================================

class ThirdOrderLPT(LPTBase):
    """Third-order Lagrangian Perturbation Theory.

    Total displacement: Ψ = D₁Ψ⁽¹⁾ + D₂Ψ⁽²⁾ + D₃ₐΨ⁽³ᵃ⁾ + D₃ᵦΨ⁽³ᵇ⁾

    Two independent 3LPT source terms (Catelan 1995; Leclercq et al. 2013):

    Type (a) — determinant of the 1LPT deformation tensor:
        Δ⁽³ᵃ⁾ = det[ φ⁽¹⁾,ᵢⱼ ]

    Type (b) — symmetric cross-coupling of 1LPT and 2LPT second derivatives:
        Δ⁽³ᵇ⁾ = Σᵢ<ⱼ [ φ⁽¹⁾,ᵢᵢ φ⁽²⁾,ⱼⱼ + φ⁽²⁾,ᵢᵢ φ⁽¹⁾,ⱼⱼ − 2 φ⁽¹⁾,ᵢⱼ φ⁽²⁾,ᵢⱼ ]

    EdS growth-factor approximations (accurate to < 1% at z > 2):
        D₃ₐ(z) ≈  1/3   × D₁³(z)
        D₃ᵦ(z) ≈ −5/21  × D₁³(z)

    All three source fields (Δ⁽²⁾, Δ⁽³ᵃ⁾, Δ⁽³ᵇ⁾) are z-independent; they are
    computed once, cached on the backend device, and recombined with the
    growth factors per redshift (3 inverse FFTs per call).
    """

    def _phi2_derivs_b(self, source2_k, kx, ky, kz, ik2):
        """Second derivatives of φ₂ as backend arrays (real-valued).

        φ₂,ᵢⱼ(k) = +kᵢkⱼ/k² · Δ⁽²⁾(k).  Note the positive sign (opposite
        to φ₁ derivs) because φ₂ = −Δ⁽²⁾/k² and the extra minus from the
        second spatial derivative cancels.

        Args:
            source2_k: Δ⁽²⁾(k) as a backend array (already FFT'd by the
                caller — avoids recomputing the 2LPT source).
        """
        b = self._backend
        N = self.N

        def _deriv(ki, kj):
            return b.irfftn(ki * kj * ik2 * source2_k, (N, N, N))

        return {
            'xx': _deriv(kx, kx), 'yy': _deriv(ky, ky), 'zz': _deriv(kz, kz),
            'xy': _deriv(kx, ky), 'xz': _deriv(kx, kz), 'yz': _deriv(ky, kz),
        }

    def _D3a(self, z: float) -> float:
        return (1.0 / 3.0) * self._D1(z) ** 3

    def _D3b(self, z: float) -> float:
        return (-5.0 / 21.0) * self._D1(z) ** 3

    def _compute_sources_k(self) -> list:
        # Intermediates are dropped with `del` as soon as their last use is
        # done (issue #42, O4) — d1's 6 arrays are the biggest holdout (needed
        # through source3b), source2/source3a/source3b are transient. Pure
        # bookkeeping: values and FFT calls are unchanged, so results are
        # identical to before.
        b = self._backend
        kx, ky, kz, ik2 = self._setup_k_backend()
        dk = self._dk_backend()

        # φ₁ second derivs (shared by 2LPT, 3LPT-a, 3LPT-b)
        d1 = self._phi1_derivs_b(dk, kx, ky, kz, ik2)

        # 2LPT
        source2 = (
            d1['xx'] * d1['yy'] - d1['xy'] ** 2
            + d1['xx'] * d1['zz'] - d1['xz'] ** 2
            + d1['yy'] * d1['zz'] - d1['yz'] ** 2
        )
        source2_k = b.rfftn(source2)
        del source2

        # 3LPT type (a): det[ φ⁽¹⁾,ᵢⱼ ]
        source3a = (
            d1['xx'] * (d1['yy'] * d1['zz'] - d1['yz'] ** 2)
            - d1['xy'] * (d1['xy'] * d1['zz'] - d1['yz'] * d1['xz'])
            + d1['xz'] * (d1['xy'] * d1['yz'] - d1['yy'] * d1['xz'])
        )
        source3a_k = b.rfftn(source3a)
        del source3a

        # 3LPT type (b): symmetric φ⁽¹⁾ × φ⁽²⁾ cross term
        d2 = self._phi2_derivs_b(source2_k, kx, ky, kz, ik2)
        source3b = (
            d1['xx'] * d2['yy'] + d2['xx'] * d1['yy'] - 2 * d1['xy'] * d2['xy']
            + d1['xx'] * d2['zz'] + d2['xx'] * d1['zz'] - 2 * d1['xz'] * d2['xz']
            + d1['yy'] * d2['zz'] + d2['yy'] * d1['zz'] - 2 * d1['yz'] * d2['yz']
        )
        del d1, d2
        source3b_k = b.rfftn(source3b)
        del source3b

        return [dk, source2_k, source3a_k, source3b_k]

    def _growth_coeffs(self, z: float) -> list[float]:
        return [self._D1(z), self._D2(z), self._D3a(z), self._D3b(z)]

    def _velocity_orders(self) -> list[float]:
        return [1.0, 2.0, 3.0, 3.0]   # fₙ ≈ n·f₁ in EdS
