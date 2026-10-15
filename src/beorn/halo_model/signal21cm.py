"""21-cm power spectrum from the halo model, using BEoRN's radiation profiles.

:class:`HaloModel21cm` is a :class:`~beorn.halo_model.HaloModel` with four
tracers — matter plus three built from the radiation profiles.  The
painting pipeline and this module start from the same
:class:`~beorn.structs.RadiationProfiles` (Lyman-alpha flux, X-ray heating,
ionised-bubble radius per halo mass, accretion rate and redshift).
Instead of stamping them onto a grid around simulated haloes, the
profiles are Fourier transformed and combined with n(M), b(M) and
P_lin(k) (Schneider, Giri & Mirocha 2021; Schneider, Giri et al. 2023).

Fields (per halo, comoving coordinates):

* ``m``     — matter overdensity, NFW: W_m = M/rho_m u(k|M)
* ``ion``   — ionisation fraction, sphere of radius R_bubble: W = 4/3 pi R^3 W_TH(kR)
* ``alpha`` — Lyman-alpha coupling x_alpha = 1.81e11 rho_alpha/(1+z) S_alpha/(4 pi)
* ``heat``  — X-ray heating temperature increment rho_heat [K]

exactly as in :meth:`beorn.painting.PaintingCoordinator.paint_single_mass_bin`.
Mean fields are the k -> 0 limits of the mass integrals.  The
differential brightness temperature

    dTb = F sqrt(1+z) (1 - T_cmb/T_k) (1 - x_HII) x_tot/(1 + x_tot) (1 + delta_b)

is linearised around the mean fields, dTb - <dTb> = sum_X A_X dX, with
T_k = T_ad (1 + delta)^(2/3) + T_heat and x_tot = x_alpha + x_coll(1 + delta), so

    P_21(k) = sum_XY A_X A_Y P_XY(k).

Bubble overlap: the ``ion`` tracer is a *sum* of spheres, i.e. the filling
factor Q = \int n V_b dlnM and its fluctuations, which keep growing once
bubbles overlap.  For randomly overlapping bubbles x_HII = 1 - exp(-Q), so
dx_HII = (1 - x_HII) dQ: the ionisation fluctuations are suppressed by
(1 - <x_HII>) and P_21 vanishes as reionisation completes.  This is the default
(``bubble_correction=True``, ``b_1h = b_2h = 1``); HMreio's phenomenological
exponents (e.g. ``b_1h=1.2, b_2h=0``) are a generalisation of it.  The mean
<x_HII> = min(Q, 1) is photon-conserving, as in the painting pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..constants import M_sun, cm_per_Mpc, m_H, rhoc0
from ..cosmo import T_adiab, T_cmb, dTb_factor
from ..couplings import S_alpha, x_coll
from ..io import Handler
from ..structs import RadiationProfiles
from .base import HaloModel
from .tracers import ProfileTracer, TopHatTracer

logger = logging.getLogger(__name__)

FIELDS = ('m', 'ion', 'alpha', 'heat')
"""Tracers of the 21-cm halo model."""


class HaloModel21cm(HaloModel):
    """Halo-model 21-cm power spectrum and all auto/cross spectra.

    Registers the tracers ``'ion'`` (:class:`~.tracers.TopHatTracer`, radius
    R_bubble), ``'alpha'`` and ``'heat'`` (:class:`~.tracers.ProfileTracer`)
    from :class:`~beorn.structs.RadiationProfiles`, next to the NFW matter
    tracer ``'m'`` of :class:`~.base.HaloModel`.  Further tracers can be added
    with :meth:`add_tracer` and appear in :meth:`power_spectra`.

    Args:
        parameters: BEoRN Parameters.
        profiles:   Precomputed :class:`~beorn.structs.RadiationProfiles`.
                    If ``None`` (default), they are computed with
                    :class:`~beorn.precomputation.RadiationProfileSolver` on
                    ``parameters.solver.redshifts``.
        cache_profiles: Profile caching when ``profiles`` is ``None``.
                    ``False`` (default) solves them in memory on every call —
                    the profile solver takes seconds, and repeated runs (scans,
                    inference) would otherwise fill the disk with one HDF5 file
                    per parameter set. ``True`` caches under ``./cache``; a
                    path or :class:`~beorn.io.Handler` selects the location.
                    Cached profiles are reused for identical profile parameters
                    (``Parameters.profiles_hash``), as in the painting pipeline.
        force_recompute: With caching on, ignore existing cached profiles.
        alpha:      Exponential accretion rate used to select the profiles
                    (nearest bin of ``parameters.solver.halo_mass_accretion_alpha``).
        bubble_correction: Suppress the ionisation fluctuations by the
                    bubble-overlap factors ``(1 - x_HII)^b_1h`` (1h) and
                    ``(1 - x_HII)^b_2h`` (2h) (HMreio ``bubble_corr``). Default
                    True — without it P_21 keeps growing until the end of
                    reionisation instead of dropping to zero.
        b_1h, b_2h: Exponents of the bubble-overlap correction. The default
                    ``1, 1`` is the random (Poisson) overlap result
                    ``dx_HII = (1 - x_HII) dQ``; HMreio examples use e.g. ``1.2, 0``.
        **kwargs:   Forwarded to :class:`~.base.HaloModel` (``ps_method``,
                    ``hmf_model``, mass grid, Boltzmann-solver ``ps_var`` ...).

    Example::

        hm  = HaloModel21cm(param, ps_method='camb')        # profiles solved in memory
        out = hm.compute(k=np.logspace(-2, 1, 50), z=[8.0, 10.0])
        out['P_21']      # (Nz, Nk) [mK^2 (Mpc/h)^3]

        hm.update_parameters(param_new)   # new astro parameters: only profiles are recomputed
        HaloModel21cm(param, cache_profiles='my_cache')     # opt-in disk cache
    """

    def __init__(
        self,
        parameters,
        profiles: RadiationProfiles | None = None,
        cache_profiles: bool | str | Path | Handler = False,
        force_recompute: bool = False,
        alpha: float = 0.79,
        bubble_correction: bool = True,
        b_1h: float = 1.0,
        b_2h: float = 1.0,
        **kwargs,
    ):
        super().__init__(parameters, **kwargs)
        self.cache_profiles = cache_profiles
        self.force_recompute = force_recompute
        self.requested_alpha = alpha
        self.bubble_correction = bubble_correction
        self.b_1h, self.b_2h = b_1h, b_2h
        self._set_profiles(profiles)

        # the profile tracers read self.profiles lazily, so update_parameters() needs no rebuild
        self.add_tracer(TopHatTracer('ion', self._R_bubble, self._mass_edges))
        self.add_tracer(ProfileTracer('alpha', self._r_lyal, self._x_alpha_profile, self._mass_edges))
        self.add_tracer(ProfileTracer('heat', self._r_heat, self._T_heat_profile, self._mass_edges))

    # ------------------------------------------------------------------
    # Radiation profiles
    # ------------------------------------------------------------------

    def _profile_handler(self) -> Handler | None:
        cache = self.cache_profiles
        if cache is False or cache is None:
            return None
        if isinstance(cache, Handler):
            return cache
        return Handler(Path('cache') if cache is True else Path(cache))

    def _set_profiles(self, profiles: RadiationProfiles | None) -> None:
        if profiles is None:
            from ..precomputation import RadiationProfileSolver
            solver = RadiationProfileSolver(self.parameters, self.parameters.solver.redshifts)
            handler = self._profile_handler()
            if handler is None:
                profiles = solver.solve()
            else:
                profiles = solver.get_or_compute_profiles(handler, force_recompute=self.force_recompute)
        self.profiles = profiles
        alpha_edges = np.asarray(self.parameters.solver.halo_mass_accretion_alpha, dtype=float)
        alpha_centres = 0.5 * (alpha_edges[1:] + alpha_edges[:-1])
        self.alpha_index = int(np.argmin(np.abs(alpha_centres - self.requested_alpha)))
        self.alpha = float(alpha_centres[self.alpha_index])
        for tracer in getattr(self, 'tracers', {}).values():
            tracer.clear_cache()

    def update_parameters(self, parameters, profiles: RadiationProfiles | None = None) -> None:
        """Switch to new (astrophysical / solver) parameters and refresh the profiles.

        The linear P(k), mass function and bias are kept, so repeated runs
        (parameter scans, inference) only pay for the profile solver. The
        cosmology must therefore be unchanged — build a new instance otherwise.

        Args:
            parameters: New BEoRN Parameters (same cosmology).
            profiles:   Optional precomputed profiles for ``parameters``;
                        computed (and cached only if ``cache_profiles``) when ``None``.
        """
        if asdict(parameters.cosmology) != asdict(self.parameters.cosmology):
            raise ValueError(
                "update_parameters() keeps P(k) and n(M): the cosmology must not change. "
                "Create a new HaloModel21cm for a different cosmology."
            )
        self.parameters = parameters
        self.hmf.parameters = parameters
        self._set_profiles(profiles)

    def _z_index(self, z: float, warn: bool = True) -> int:
        z_hist = np.asarray(self.profiles.z_history[:])
        i = int(np.argmin(np.abs(z_hist - z)))
        if warn and abs(z_hist[i] - z) > 1e-3:
            spacing = np.max(np.abs(np.diff(z_hist))) if z_hist.size > 1 else 0.0
            outside = z < z_hist.min() or z > z_hist.max()
            log = logger.warning if (outside or abs(z_hist[i] - z) > spacing) else logger.debug
            log(f"Using profiles at z={z_hist[i]:.3f} for requested z={z:.3f}.")
        return i

    # profile accessors used by the tracers (comoving radii, z = profile redshift)

    def _mass_edges(self, z):
        return np.asarray(self.profiles.halo_mass_bins[:, self.alpha_index, self._z_index(z, warn=False)])

    def _R_bubble(self, z):
        return np.asarray(self.profiles.R_bubble[:, self.alpha_index, self._z_index(z, warn=False)])  # cMpc/h

    def _r_lyal(self, z):
        return np.asarray(self.profiles.r_lyal[:]) * (1.0 + z)                     # pMpc/h -> cMpc/h

    def _x_alpha_profile(self, z):
        """x_alpha profile before the S_alpha / (4 pi) factor (as in the painting)."""
        rho = np.asarray(self.profiles.rho_alpha[:, :, self.alpha_index, self._z_index(z, warn=False)])
        return 1.81e11 * rho / (1.0 + z)

    def _r_heat(self, z):
        return np.asarray(self.profiles.r_grid_cell[:])                            # cMpc/h

    def _T_heat_profile(self, z):
        return np.asarray(self.profiles.rho_heat[:, :, self.alpha_index, self._z_index(z, warn=False)])

    def profile_redshift(self, z: float) -> float:
        """Nearest redshift of the radiation profiles."""
        return float(self.profiles.z_history[self._z_index(z)])

    # ------------------------------------------------------------------
    # Mean fields and linear response coefficients
    # ------------------------------------------------------------------

    def _baryon_number_density(self, z: float) -> float:
        """Mean hydrogen number density [physical cm^-3] (same as ``CoevalCube.coef``)."""
        c = self.parameters.cosmology
        return rhoc0 * c.h0 ** 2 * c.Ob * (1 + z) ** 3 * M_sun / cm_per_Mpc ** 3 / m_H

    def mean_fields(self, z: float, tracer_means: dict | None = None) -> dict:
        """Global quantities and linear-response coefficients at a profile redshift.

        Args:
            z:            Redshift (a profile redshift, see :meth:`profile_redshift`).
            tracer_means: ``{'ion': .., 'alpha': .., 'heat': ..}`` tracer means
                          (computed if not given; ``alpha`` before S_alpha).
        """
        if tracer_means is None:
            tracer_means = {f: self.tracers[f].mean(z) for f in ('ion', 'alpha', 'heat')}
        x_HII = min(tracer_means['ion'], 1.0)
        x_HII = max(x_HII, self.parameters.source.min_xHII_value)
        T_ad = float(T_adiab(z, self.parameters))
        T_k = T_ad + tracer_means['heat']
        s_alpha = float(S_alpha(z, T_k, 1.0 - x_HII))
        x_alpha = s_alpha / (4.0 * np.pi) * tracer_means['alpha']
        x_c = float(x_coll(z=z, Tk=T_k, xHI=1.0 - x_HII, rho_b=self._baryon_number_density(z)))
        x_tot = x_alpha + x_c

        F = dTb_factor(self.parameters) * np.sqrt(1.0 + z)
        Tc = T_cmb(z)
        X = x_tot / (1.0 + x_tot)
        dTb = F * (1.0 - Tc / T_k) * (1.0 - x_HII) * X

        # A_X = d dTb / d X at the mean fields
        A_ion = -F * (1.0 - Tc / T_k) * X
        A_xtot = F * (1.0 - Tc / T_k) * (1.0 - x_HII) / (1.0 + x_tot) ** 2
        A_T = F * (1.0 - x_HII) * X * Tc / T_k ** 2
        A_m = dTb + A_T * (2.0 / 3.0) * T_ad + A_xtot * x_c

        return {
            'z': z, 'dTb': dTb, 'x_HII': x_HII, 'T_k': T_k, 'T_adiab': T_ad,
            'x_alpha': x_alpha, 'x_coll': x_c, 'S_alpha': s_alpha,
            'A': {'m': A_m, 'ion': A_ion, 'alpha': A_xtot, 'heat': A_T},
        }

    # ------------------------------------------------------------------
    # Power spectra
    # ------------------------------------------------------------------

    def compute_single(self, k, z: float) -> dict:
        """All spectra at one redshift; see :meth:`compute`."""
        k = np.asarray(k, dtype=float)
        z_p = self.profile_redshift(z)
        out = self.power_spectra(k, z_p, tracers=FIELDS)
        means = self.mean_fields(z_p, {f: out[f'mean_{f}'] for f in ('ion', 'alpha', 'heat')})

        # field rescalings that depend on the mean fields (the halo windows are linear in them):
        # alpha -> x_alpha with the mean S_alpha / (4 pi); ion -> bubble-overlap correction
        scale_1h = {f: 1.0 for f in FIELDS}
        scale_2h = {f: 1.0 for f in FIELDS}
        scale_1h['alpha'] = scale_2h['alpha'] = means['S_alpha'] / (4.0 * np.pi)
        if self.bubble_correction:
            scale_1h['ion'] = (1.0 - means['x_HII']) ** self.b_1h
            scale_2h['ion'] = (1.0 - means['x_HII']) ** self.b_2h
        out['mean_alpha'] = means['x_alpha']
        for i, X in enumerate(FIELDS):
            for Y in FIELDS[i:]:
                key = f'P_{X}_{Y}'
                out[key + '_1h'] = out[key + '_1h'] * scale_1h[X] * scale_1h[Y]
                out[key + '_2h'] = out[key + '_2h'] * scale_2h[X] * scale_2h[Y]
                out[key] = out[key + '_1h'] + out[key + '_2h']

        A = means['A']
        for term in ('_1h', '_2h'):
            P21 = 0.0
            for i, X in enumerate(FIELDS):
                for Y in FIELDS[i:]:
                    fac = 1.0 if X == Y else 2.0
                    P21 = P21 + fac * A[X] * A[Y] * out[f'P_{X}_{Y}{term}']
            out[f'P_21{term}'] = P21
        out['P_21'] = out['P_21_1h'] + out['P_21_2h']
        out['Delta2_21'] = k ** 3 * out['P_21'] / (2.0 * np.pi ** 2)
        out.update({key: v for key, v in means.items() if key != 'z'})
        return out

    def compute(self, k, z) -> dict:
        """Halo-model 21-cm power spectra.

        Args:
            k: (Nk,) wavenumbers [h/Mpc].
            z: Redshift or array of redshifts; the nearest profile redshift
               is used.

        Returns:
            dict with ``k`` (Nk,), ``z`` (Nz,), spectra of shape (Nz, Nk):

            * ``P_21``, ``P_21_1h``, ``P_21_2h`` [mK^2 (Mpc/h)^3] and
              ``Delta2_21`` [mK^2];
            * ``P_<X>_<Y>``, ``P_<X>_<Y>_1h``, ``P_<X>_<Y>_2h`` for X, Y in
              ``('m', 'ion', 'alpha', 'heat')`` — fluctuations of delta_m,
              x_HII, x_alpha and T_heat [K];
            * ``P_lin``;

            and, shape (Nz,): mean fields ``dTb`` [mK], ``x_HII``, ``T_k`` [K],
            ``T_adiab``, ``x_alpha``, ``x_coll``, ``S_alpha``, tracer means
            ``mean_<X>``, plus ``A`` (linear-response coefficients per field).
        """
        z_arr = np.atleast_1d(np.asarray(z, dtype=float))
        results = [self.compute_single(k, zi) for zi in z_arr]
        out = {'k': results[0]['k']}
        for key in results[0]:
            if key == 'k':
                continue
            if key == 'A':
                out['A'] = {f: np.array([r['A'][f] for r in results]) for f in FIELDS}
            else:
                out[key] = np.array([r[key] for r in results])
        return out
