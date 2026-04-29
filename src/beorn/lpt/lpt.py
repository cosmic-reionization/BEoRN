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

* ``backend='numpy'``  — multi-core CPU via ``scipy.fft`` (``workers=-1``)
* ``backend='torch'``  — GPU via PyTorch CUDA (auto-detected)
* ``backend='jax'``    — GPU/TPU via JAX (auto-detected)
* ``backend='auto'``   — picks best available: JAX GPU > Torch CUDA > NumPy

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

from abc import ABC, abstractmethod
import numpy as np

from ..cosmo import D, hubble
from .backends import get_backend, LPTBackend
from .linear_power import PowerSpectrum, get_power_spectrum


# ============================================================
# Helpers
# ============================================================

def _kvectors(N: int, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wavenumber arrays for an rfftn-shaped grid (N, N, N//2+1).

    Returns:
        kx, ky, kz, k2 — each of shape (N, N, N//2+1).
        kx, ky are in full-FFT ordering; kz covers [0, k_nyq].
        Units: h/Mpc.
    """
    dk = 2.0 * np.pi / L
    kvals_full = np.fft.fftfreq(N, d=1.0 / N) * dk   # shape (N,)
    kvals_half = np.fft.rfftfreq(N, d=1.0 / N) * dk  # shape (N//2+1,)
    kx = kvals_full[:, None, None] * np.ones((1, N, N // 2 + 1))
    ky = kvals_full[None, :, None] * np.ones((N, 1, N // 2 + 1))
    kz = kvals_half[None, None, :] * np.ones((N, N, 1))
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
        **ps_kwargs: Extra keyword arguments forwarded to the power spectrum
                     constructor (e.g. ``wiggle=True`` for the E&H with-wiggle
                     fit, or ``ps_file='Pk_camb.dat'`` for the Boltzmann
                     solver).
    """

    def __init__(
        self,
        parameters,
        ps_method: str = 'eisenstein_hu',
        backend: str | LPTBackend = 'auto',
        seed: int = 42,
        fixed: bool = True,
        verbose: bool = True,
        **ps_kwargs,
    ):
        self.parameters = parameters
        self.seed = seed
        self.fixed = fixed
        self.verbose = verbose
        self._backend: LPTBackend = get_backend(backend, verbose=verbose)
        self.power_spectrum: PowerSpectrum = get_power_spectrum(
            ps_method, parameters, **ps_kwargs
        )
        self._delta_k: np.ndarray | None = None  # cached IC realisation

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

        Returns:
            delta_k — complex array of shape (N, N, N//2+1).
        """
        N, L = self.N, self.L
        V = L ** 3

        # Amplitude grid (always built in numpy — cheap scalar ops)
        kx, ky, kz, k2 = _kvectors(N, L)
        k_safe = np.where(k2 == 0, 1.0, np.sqrt(k2))
        Pk = self.power_spectrum.P(k_safe, z=0.0)
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
        """Linear growth rate f₁ = d ln D₁ / d ln a (numerical finite difference)."""
        a = 1.0 / (1.0 + z)
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
            kx, ky, kz — shape (N, N, N//2+1), units h/Mpc.
            inv_k2     — 1/k² with the k=0 mode set to 0 (avoids divergence
                         *and* automatically zeros the DC displacement).
        """
        kx, ky, kz, k2 = _kvectors(self.N, self.L)
        inv_k2 = np.where(k2 == 0, 0.0, 1.0 / k2)
        return kx, ky, kz, inv_k2

    def _setup_k_backend(self):
        """Return k-vectors and inv_k2 as backend arrays on the active device."""
        b = self._backend
        kx, ky, kz, inv_k2 = self._setup_k()
        return (b.as_array(kx), b.as_array(ky), b.as_array(kz),
                b.as_array(inv_k2))

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
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_displacement(self, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the LPT displacement field (Ψx, Ψy, Ψz) at redshift z.

        Each component has shape (N, N, N) in units of Mpc/h.
        Lagrangian positions q on a regular grid are displaced as:
            x = q + Ψ(q)
        """

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
        qx, qy, qz = np.meshgrid(q1d, q1d, q1d, indexing='ij')

        x = (qx + psi_x) % L
        y = (qy + psi_y) % L
        z_pos = (qz + psi_z) % L

        positions = np.stack([x.ravel(), y.ravel(), z_pos.ravel()], axis=-1)
        return positions.astype(np.float32)

    def get_density(self, z: float) -> np.ndarray:
        """Matter overdensity δ(x) at redshift z via CIC particle painting.

        Returns:
            delta — shape (N, N, N), mean-zero overdensity.
        """
        from ..particle_mapping import map_particles_to_mesh

        N, L = self.N, self.L
        mesh = np.zeros((N, N, N), dtype=np.float32)
        positions = self.get_positions(z)
        map_particles_to_mesh(mesh, L, positions, mass_assignment='CIC')
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

    All FFTs run on the active backend device (GPU if available).
    """

    def get_displacement(self, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        N, L = self.N, self.L
        b = self._backend
        D1 = self._D1(z)

        kx, ky, kz, ik2 = self._setup_k_backend()
        dk = b.as_array(self.delta_k)  # move to device once

        # Ψᵢ(k) = i kᵢ / k² · δ(k)  (DC is zero via ik2)
        psi_x = D1 * b.to_numpy(b.irfftn(1j * kx * ik2 * dk, (N, N, N)))
        psi_y = D1 * b.to_numpy(b.irfftn(1j * ky * ik2 * dk, (N, N, N)))
        psi_z = D1 * b.to_numpy(b.irfftn(1j * kz * ik2 * dk, (N, N, N)))

        return psi_x, psi_y, psi_z

    def get_velocity(self, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Peculiar velocity (vx, vy, vz) at redshift z in km/s.

        v = a(z) H(z)/h f₁(z) Ψ⁽¹⁾(q, z)

        where H(z)/h is in km/s/(Mpc/h) and Ψ⁽¹⁾ = D₁(z) × base displacement.

        Returns:
            (vx, vy, vz) — each shape (N, N, N), units km/s.
        """
        a = 1.0 / (1.0 + z)
        Hz = hubble(z, self.parameters) / self.parameters.cosmology.h0
        factor = a * Hz * self._f1(z)
        psi_x, psi_y, psi_z = self.get_displacement(z)
        return factor * psi_x, factor * psi_y, factor * psi_z


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

    All intermediate arrays stay on the backend device; only the final Ψ
    components are transferred to CPU numpy.
    """

    def get_displacement(self, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        N, L = self.N, self.L
        b = self._backend
        D1, D2 = self._D1(z), self._D2(z)

        kx, ky, kz, ik2 = self._setup_k_backend()
        dk = b.as_array(self.delta_k)

        # ── 1LPT ────────────────────────────────────────────────────────
        psi1_x_k = 1j * kx * ik2 * dk
        psi1_y_k = 1j * ky * ik2 * dk
        psi1_z_k = 1j * kz * ik2 * dk

        # ── 2LPT source (all backend ops, stays on device) ──────────────
        d = self._phi1_derivs_b(dk, kx, ky, kz, ik2)
        source2 = (
            d['xx'] * d['yy'] - d['xy'] ** 2
            + d['xx'] * d['zz'] - d['xz'] ** 2
            + d['yy'] * d['zz'] - d['yz'] ** 2
        )
        source2_k = b.rfftn(source2)
        psi2_x_k = 1j * kx * ik2 * source2_k
        psi2_y_k = 1j * ky * ik2 * source2_k
        psi2_z_k = 1j * kz * ik2 * source2_k

        # ── Combine + single to_numpy per component ──────────────────────
        irfft = lambda xk: b.irfftn(xk, (N, N, N))
        psi_x = b.to_numpy(D1 * irfft(psi1_x_k) + D2 * irfft(psi2_x_k))
        psi_y = b.to_numpy(D1 * irfft(psi1_y_k) + D2 * irfft(psi2_y_k))
        psi_z = b.to_numpy(D1 * irfft(psi1_z_k) + D2 * irfft(psi2_z_k))

        return psi_x, psi_y, psi_z

    def get_velocity(self, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Peculiar velocity (vx, vy, vz) at redshift z in km/s.

        v = a H/h [f₁ D₁ Ψ⁽¹⁾ + f₂ D₂ Ψ⁽²⁾]

        Uses EdS approximation f₂ ≈ 2 f₁ (accurate to < 1 % at z > 2).

        Returns:
            (vx, vy, vz) — each shape (N, N, N), units km/s.
        """
        N, L = self.N, self.L
        b = self._backend
        a = 1.0 / (1.0 + z)
        Hz = hubble(z, self.parameters) / self.parameters.cosmology.h0
        f1 = self._f1(z)
        D1, D2 = self._D1(z), self._D2(z)
        vD1 = a * Hz * f1 * D1
        vD2 = a * Hz * 2.0 * f1 * D2  # f₂ ≈ 2f₁ in EdS

        kx, ky, kz, ik2 = self._setup_k_backend()
        dk = b.as_array(self.delta_k)

        psi1_x_k = 1j * kx * ik2 * dk
        psi1_y_k = 1j * ky * ik2 * dk
        psi1_z_k = 1j * kz * ik2 * dk

        d = self._phi1_derivs_b(dk, kx, ky, kz, ik2)
        source2 = (
            d['xx'] * d['yy'] - d['xy'] ** 2
            + d['xx'] * d['zz'] - d['xz'] ** 2
            + d['yy'] * d['zz'] - d['yz'] ** 2
        )
        source2_k = b.rfftn(source2)
        psi2_x_k = 1j * kx * ik2 * source2_k
        psi2_y_k = 1j * ky * ik2 * source2_k
        psi2_z_k = 1j * kz * ik2 * source2_k

        irfft = lambda xk: b.irfftn(xk, (N, N, N))
        v_x = b.to_numpy(vD1 * irfft(psi1_x_k) + vD2 * irfft(psi2_x_k))
        v_y = b.to_numpy(vD1 * irfft(psi1_y_k) + vD2 * irfft(psi2_y_k))
        v_z = b.to_numpy(vD1 * irfft(psi1_z_k) + vD2 * irfft(psi2_z_k))
        return v_x, v_y, v_z


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

    All intermediate arrays stay on the backend device.
    """

    def _phi2_derivs_b(self, d1, kx, ky, kz, ik2):
        """Second derivatives of φ₂ as backend arrays (real-valued).

        φ₂,ᵢⱼ(k) = +kᵢkⱼ/k² · Δ⁽²⁾(k).  Note the positive sign (opposite
        to φ₁ derivs) because φ₂ = −Δ⁽²⁾/k² and the extra minus from the
        second spatial derivative cancels.

        Args:
            d1: dict of φ₁ second-derivative backend arrays (from
                :meth:`_phi1_derivs_b`).
        """
        b = self._backend
        N = self.N
        source2 = (
            d1['xx'] * d1['yy'] - d1['xy'] ** 2
            + d1['xx'] * d1['zz'] - d1['xz'] ** 2
            + d1['yy'] * d1['zz'] - d1['yz'] ** 2
        )
        source2_k = b.rfftn(source2)

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

    def get_displacement(self, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        N, L = self.N, self.L
        b = self._backend
        D1, D2 = self._D1(z), self._D2(z)
        D3a, D3b = self._D3a(z), self._D3b(z)

        kx, ky, kz, ik2 = self._setup_k_backend()
        dk = b.as_array(self.delta_k)

        # ── 1LPT ────────────────────────────────────────────────────────
        psi1_x_k = 1j * kx * ik2 * dk
        psi1_y_k = 1j * ky * ik2 * dk
        psi1_z_k = 1j * kz * ik2 * dk

        # ── φ₁ second derivs (shared by 2LPT, 3LPT-a, 3LPT-b) ──────────
        d1 = self._phi1_derivs_b(dk, kx, ky, kz, ik2)

        # ── 2LPT ────────────────────────────────────────────────────────
        source2 = (
            d1['xx'] * d1['yy'] - d1['xy'] ** 2
            + d1['xx'] * d1['zz'] - d1['xz'] ** 2
            + d1['yy'] * d1['zz'] - d1['yz'] ** 2
        )
        source2_k = b.rfftn(source2)
        psi2_x_k = 1j * kx * ik2 * source2_k
        psi2_y_k = 1j * ky * ik2 * source2_k
        psi2_z_k = 1j * kz * ik2 * source2_k

        # ── 3LPT type (a): det[ φ⁽¹⁾,ᵢⱼ ] ─────────────────────────────
        source3a = (
            d1['xx'] * (d1['yy'] * d1['zz'] - d1['yz'] ** 2)
            - d1['xy'] * (d1['xy'] * d1['zz'] - d1['yz'] * d1['xz'])
            + d1['xz'] * (d1['xy'] * d1['yz'] - d1['yy'] * d1['xz'])
        )
        source3a_k = b.rfftn(source3a)
        psi3a_x_k = 1j * kx * ik2 * source3a_k
        psi3a_y_k = 1j * ky * ik2 * source3a_k
        psi3a_z_k = 1j * kz * ik2 * source3a_k

        # ── 3LPT type (b): symmetric φ⁽¹⁾ × φ⁽²⁾ cross term ────────────
        d2 = self._phi2_derivs_b(d1, kx, ky, kz, ik2)
        source3b = (
            d1['xx'] * d2['yy'] + d2['xx'] * d1['yy'] - 2 * d1['xy'] * d2['xy']
            + d1['xx'] * d2['zz'] + d2['xx'] * d1['zz'] - 2 * d1['xz'] * d2['xz']
            + d1['yy'] * d2['zz'] + d2['yy'] * d1['zz'] - 2 * d1['yz'] * d2['yz']
        )
        source3b_k = b.rfftn(source3b)
        psi3b_x_k = 1j * kx * ik2 * source3b_k
        psi3b_y_k = 1j * ky * ik2 * source3b_k
        psi3b_z_k = 1j * kz * ik2 * source3b_k

        # ── Combine + single to_numpy per component ──────────────────────
        irfft = lambda xk: b.irfftn(xk, (N, N, N))
        psi_x = b.to_numpy(
            D1 * irfft(psi1_x_k) + D2 * irfft(psi2_x_k)
            + D3a * irfft(psi3a_x_k) + D3b * irfft(psi3b_x_k)
        )
        psi_y = b.to_numpy(
            D1 * irfft(psi1_y_k) + D2 * irfft(psi2_y_k)
            + D3a * irfft(psi3a_y_k) + D3b * irfft(psi3b_y_k)
        )
        psi_z = b.to_numpy(
            D1 * irfft(psi1_z_k) + D2 * irfft(psi2_z_k)
            + D3a * irfft(psi3a_z_k) + D3b * irfft(psi3b_z_k)
        )

        return psi_x, psi_y, psi_z

    def get_velocity(self, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Peculiar velocity (vx, vy, vz) at redshift z in km/s.

        v = aH/h [f₁D₁Ψ⁽¹⁾ + f₂D₂Ψ⁽²⁾ + f₃ₐD₃ₐΨ⁽³ᵃ⁾ + f₃ᵦD₃ᵦΨ⁽³ᵇ⁾]

        EdS growth rates: fₙ ≈ n·f₁ for Dₙ ∝ D₁ⁿ.

        Returns:
            (vx, vy, vz) — each shape (N, N, N), units km/s.
        """
        N, L = self.N, self.L
        b = self._backend
        a = 1.0 / (1.0 + z)
        Hz = hubble(z, self.parameters) / self.parameters.cosmology.h0
        f1 = self._f1(z)
        D1, D2 = self._D1(z), self._D2(z)
        D3a, D3b = self._D3a(z), self._D3b(z)
        vD1  = a * Hz * f1       * D1
        vD2  = a * Hz * 2.0 * f1 * D2
        vD3a = a * Hz * 3.0 * f1 * D3a
        vD3b = a * Hz * 3.0 * f1 * D3b

        kx, ky, kz, ik2 = self._setup_k_backend()
        dk = b.as_array(self.delta_k)

        psi1_x_k = 1j * kx * ik2 * dk
        psi1_y_k = 1j * ky * ik2 * dk
        psi1_z_k = 1j * kz * ik2 * dk

        d1 = self._phi1_derivs_b(dk, kx, ky, kz, ik2)
        source2 = (
            d1['xx'] * d1['yy'] - d1['xy'] ** 2
            + d1['xx'] * d1['zz'] - d1['xz'] ** 2
            + d1['yy'] * d1['zz'] - d1['yz'] ** 2
        )
        source2_k = b.rfftn(source2)
        psi2_x_k = 1j * kx * ik2 * source2_k
        psi2_y_k = 1j * ky * ik2 * source2_k
        psi2_z_k = 1j * kz * ik2 * source2_k

        source3a = (
            d1['xx'] * (d1['yy'] * d1['zz'] - d1['yz'] ** 2)
            - d1['xy'] * (d1['xy'] * d1['zz'] - d1['yz'] * d1['xz'])
            + d1['xz'] * (d1['xy'] * d1['yz'] - d1['yy'] * d1['xz'])
        )
        source3a_k = b.rfftn(source3a)
        psi3a_x_k = 1j * kx * ik2 * source3a_k
        psi3a_y_k = 1j * ky * ik2 * source3a_k
        psi3a_z_k = 1j * kz * ik2 * source3a_k

        d2 = self._phi2_derivs_b(d1, kx, ky, kz, ik2)
        source3b = (
            d1['xx'] * d2['yy'] + d2['xx'] * d1['yy'] - 2 * d1['xy'] * d2['xy']
            + d1['xx'] * d2['zz'] + d2['xx'] * d1['zz'] - 2 * d1['xz'] * d2['xz']
            + d1['yy'] * d2['zz'] + d2['yy'] * d1['zz'] - 2 * d1['yz'] * d2['yz']
        )
        source3b_k = b.rfftn(source3b)
        psi3b_x_k = 1j * kx * ik2 * source3b_k
        psi3b_y_k = 1j * ky * ik2 * source3b_k
        psi3b_z_k = 1j * kz * ik2 * source3b_k

        irfft = lambda xk: b.irfftn(xk, (N, N, N))
        v_x = b.to_numpy(
            vD1 * irfft(psi1_x_k) + vD2 * irfft(psi2_x_k)
            + vD3a * irfft(psi3a_x_k) + vD3b * irfft(psi3b_x_k)
        )
        v_y = b.to_numpy(
            vD1 * irfft(psi1_y_k) + vD2 * irfft(psi2_y_k)
            + vD3a * irfft(psi3a_y_k) + vD3b * irfft(psi3b_y_k)
        )
        v_z = b.to_numpy(
            vD1 * irfft(psi1_z_k) + vD2 * irfft(psi2_z_k)
            + vD3a * irfft(psi3a_z_k) + vD3b * irfft(psi3b_z_k)
        )
        return v_x, v_y, v_z
