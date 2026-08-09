"""Deterministic excursion-set halo finder for massive halos.

Implements the exact, numpy-only tier of the hybrid halo sampler described
in Davies, Mesinger & Murray (2025, 21cmFASTv4, arXiv:2504.17254): massive
halos (mass >= ``HaloSimParameters.M_split``) are found deterministically
by walking a decreasing-smoothing-scale hierarchy on the linear density
field and looking for barrier first-crossings (Mesinger & Furlanetto 2007's
DexM algorithm) -- connected patches of crossing cells are merged into
single halos, avoiding one-halo-per-gridpoint double counting. Everything
below ``M_split`` remains :class:`~beorn.lpt.chmf.CHMFSampler`'s stochastic
responsibility, with each cell's conditioning mass reduced by whatever this
tier already claimed there (``CHMFSampler.sample``'s
``deterministic_mass_fraction`` argument).

This is the "exact" counterpart of ``HaloSimParameters.excursion_set_method``
-- see :mod:`beorn.lpt.excursion_set_diff` for the differentiable surrogate
(mirrors :mod:`beorn.painting.spread`/:mod:`beorn.painting.differentiable`'s
``spreading_method='exact'``/``'diffusion'`` split for ionization-excess
spreading).

Known limitation: connected-component labelling does not wrap patches
across periodic box boundaries (the same limitation
:func:`beorn.painting.spread.spreading_excess_fast` already has) -- accepted
for now; the bias is small whenever halo sizes are small relative to the
box.
"""
from __future__ import annotations

import logging

import numpy as np
from skimage.measure import label

from ..structs import HaloCatalog
from .chmf import CHMF, tophat_k2_grid, tophat_smooth_static

logger = logging.getLogger(__name__)


class ExcursionSetFinder:
    """Exact excursion-set halo finder built on a :class:`~beorn.lpt.chmf.CHMF`.

    Args:
        chmf:        A :class:`~beorn.lpt.chmf.CHMF` instance (supplies
                     ``rho_m``, ``M_of_R``/``R_of_M``, ``barrier``, and
                     ``parameters`` for the returned catalog).
        chmf_recipe: Barrier definition used for the first-crossing test —
                     ``'BarkanaLoeb2004'`` (default) is the constant
                     spherical-collapse barrier (:attr:`CHMF.delta_c`, the
                     classic Press-Schechter excursion set); ``'MovingBarrier'``
                     is the ellipsoidal-collapse moving barrier (Sheth &
                     Tormen 2002). See :meth:`~beorn.lpt.chmf.CHMF.barrier`.
    """

    def __init__(self, chmf: CHMF, chmf_recipe: str = 'BarkanaLoeb2004'):
        self.chmf = chmf
        self.chmf_recipe = chmf_recipe

    def find(
        self,
        delta_field: np.ndarray,
        z: float,
        M_split: float,
        M_max: float | None = None,
        n_scales: int = 32,
        min_patch_mass: float | None = None,
    ) -> tuple[HaloCatalog, np.ndarray]:
        """Walk the excursion-set hierarchy and return the resulting halos.

        Args:
            delta_field: Linear overdensity, shape (N, N, N) — the raw
                (un-smoothed) field at whatever resolution the walk should
                operate on (the ``field_oversample``-refined fine field when
                available, or the coarse ``Ncell`` field otherwise).
            z:           Redshift.
            M_split:     Lower mass bound of the walk, in M_sun — halos
                below this remain the stochastic CHMFSampler's
                responsibility. Must resolve to a smoothing radius at or
                above ``delta_field``'s own cell size.
            M_max:       Upper mass bound of the walk, in M_sun. ``None``
                (default) uses 10% of the box mass.
            n_scales:    Number of log-spaced mass nodes between ``M_max``
                and ``M_split``.
            min_patch_mass: Reject patches below this mass (their cells
                remain eligible to cross at a smaller scale). ``None``
                (default) uses ``M_split`` itself.

        Returns:
            (catalog, claimed) — a :class:`~beorn.structs.HaloCatalog` of
            the accepted halos, and a boolean array (shape of
            ``delta_field``) marking cells claimed by some accepted patch —
            the latter is what the caller reduces to
            ``CHMFSampler.sample``'s ``deterministic_mass_fraction`` (every
            claimed cell contributes exactly ``rho_m`` to its patch, by
            construction, so "claimed" is already the per-cell collapsed
            fraction before any coarse block-averaging).

        Raises:
            ValueError: If ``M_max <= M_split``, or if ``M_split`` implies a
                smoothing radius below ``delta_field``'s own cell size (use
                a finer, ``field_oversample``-refined field, or raise
                ``M_split``).
        """
        params = self.chmf.parameters
        Lbox = params.Lbox_hunits
        N = delta_field.shape[0]
        cell_size = Lbox / N
        cell_volume = cell_size ** 3

        if M_max is None:
            M_max = 0.1 * self.chmf.rho_m * Lbox ** 3
        if min_patch_mass is None:
            min_patch_mass = M_split
        if M_max <= M_split:
            raise ValueError(
                f"excursion_set_M_max ({M_max:.3e} Msun) must exceed "
                f"M_split ({M_split:.3e} Msun)."
            )

        # A single pixel of *this* field represents a cube-shaped Lagrangian
        # patch of mass rho_m*cell_size**3 -- the same cube-volume M_env
        # convention _environment() uses (not a sphere-of-radius-R_of_M(M)
        # comparison against cell_size, which would always fail: the
        # top-hat sphere-equivalent radius of a cube of side cell_size is
        # always smaller than cell_size itself, by construction, regardless
        # of resolution -- see issue #54's own R_eq < cell_size note).
        M_env_of_field = self.chmf.rho_m * cell_size ** 3
        if M_split < M_env_of_field:
            raise ValueError(
                f"M_split ({M_split:.3e} Msun) is below what this field's "
                f"own cell size ({cell_size:.3e} Mpc/h, M_env-equivalent "
                f"{M_env_of_field:.3e} Msun) can resolve. Use a "
                f"field_oversample > 1 conditioning field, or raise M_split."
            )

        M_values = np.logspace(np.log10(M_max), np.log10(M_split), n_scales)
        R_values = self.chmf.R_of_M(M_values)

        delta_k = np.fft.rfftn(delta_field)
        k2 = tophat_k2_grid(N, Lbox)

        active = np.ones(delta_field.shape, dtype=bool)
        positions_list: list[np.ndarray] = []
        masses_list: list[float] = []

        for M_R, R in zip(M_values, R_values):
            if not np.any(active):
                break
            barrier_val = self.chmf.barrier(float(M_R), z, recipe=self.chmf_recipe)
            delta_smoothed = tophat_smooth_static(delta_field, float(R), Lbox,
                                                  delta_k=delta_k, k2=k2)
            mask = active & (delta_smoothed >= barrier_val)
            if not np.any(mask):
                continue

            labeled = label(mask)
            for patch_id in range(1, int(labeled.max()) + 1):
                patch_mask = labeled == patch_id
                n_pix = int(np.count_nonzero(patch_mask))
                patch_mass = self.chmf.rho_m * n_pix * cell_volume
                if patch_mass < min_patch_mass:
                    continue  # leave active -- may resolve at a smaller R

                patch_coords = np.argwhere(patch_mask)
                patch_vals = delta_smoothed[patch_mask]
                peak_idx = patch_coords[np.argmax(patch_vals)]
                positions_list.append((peak_idx + 0.5) * cell_size)
                masses_list.append(patch_mass)
                active[patch_mask] = False

        if positions_list:
            positions = np.asarray(positions_list, dtype=np.float32)
            masses = np.asarray(masses_list, dtype=np.float64)
        else:
            positions = np.zeros((0, 3), dtype=np.float32)
            masses = np.zeros(0, dtype=np.float64)

        logger.debug(
            "Excursion-set finder claimed %d halos at z=%.3f "
            "(M range [%.2e, %.2e] Msun, %d/%d cells claimed)",
            len(masses), z, M_split, M_max,
            int(np.count_nonzero(~active)), active.size,
        )

        catalog = HaloCatalog(positions=positions, masses=masses,
                              parameters=params, redshift=float(z))
        return catalog, ~active
