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

Mass convention (fixed after an under-prediction bug): each accepted patch
is assigned the *nominal filter mass* ``M(R)`` at its crossing scale
(Mesinger & Furlanetto 2007's own convention), not the pixel count of its
thresholded blob. A real density peak's smoothed value only exceeds the
barrier in a small core near its center -- nowhere close to the full
geometric sphere of radius R the filter scale represents -- so counting
literal crossing pixels systematically undercounted the true mass, worse at
larger R, and caused patches near ``M_split`` to fall below
``min_patch_mass`` and be dropped entirely. The pixel count is still used,
but only as a size filter to reject spurious few-pixel noise crossings
before trusting a patch's nominal ``M(R)``.

Correspondingly, once a patch is accepted its full R-sphere (not just its
thresholded pixels) is excluded from smaller scales -- the "full exclusion"
criterion Bond & Myers (1996b) use to stop a halo's outer envelope from
being independently re-detected as extra, smaller halos once its denser
core keeps crossing at finer R. A patch whose peak has already been claimed
by another patch accepted earlier at the *same* scale (their spheres can
overlap even when the raw threshold-crossing blobs do not touch) is
discarded outright, matching the same criterion.

Known limitation: neither the connected-component labelling nor the sphere
exclusion wraps across periodic box boundaries (the same limitation
:func:`beorn.painting.spread.spreading_excess_fast` already has) -- accepted
for now; the bias is small whenever halo sizes are small relative to the
box.
"""
from __future__ import annotations

import logging

import numpy as np
from skimage.measure import label

from ..structs import HaloCatalog
from ..particle_mapping import displace_positions
from .chmf import CHMF, tophat_k2_grid, tophat_smooth_static

logger = logging.getLogger(__name__)


def _exclude_sphere(active: np.ndarray, peak_idx: np.ndarray, R: float,
                    cell_size: float) -> None:
    """Mark all cells within physical radius ``R`` of ``peak_idx`` as
    inactive, in place -- the "full exclusion" criterion (Bond & Myers 1996b)
    that stops an accepted halo's outer envelope from being independently
    re-detected as extra, smaller halos at finer scales.

    Restricted to a local bounding box around ``peak_idx`` (cheap even for
    a large R, since the box is at most a small fraction of the full field).
    Non-periodic -- clipped at the box edges, the same accepted limitation
    as the connected-component labelling's own lack of periodic wrapping.
    """
    shape = active.shape
    r_cells = int(np.ceil(R / cell_size))
    lo = [max(int(p) - r_cells, 0) for p in peak_idx]
    hi = [min(int(p) + r_cells + 1, s) for p, s in zip(peak_idx, shape)]
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    grids = np.meshgrid(*[np.arange(l, h) for l, h in zip(lo, hi)], indexing='ij')
    dist2 = sum((g - p) ** 2 for g, p in zip(grids, peak_idx)) * cell_size ** 2
    active[sl][dist2 <= R ** 2] = False


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
        displacement_field: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
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
            min_patch_mass: Reject patches whose thresholded blob's own
                pixel-count mass falls below this (their cells remain
                eligible to cross at a smaller scale) -- a size filter
                against spurious few-pixel noise crossings, not the mass
                actually assigned to an accepted patch (see module
                docstring). ``None`` (default) uses ``M_split`` itself.
            displacement_field: ``(psi_x, psi_y, psi_z)`` LPT displacement
                components -- see
                :meth:`~beorn.lpt.chmf.CHMFSampler.sample`'s identical
                argument. When given, each accepted patch's Lagrangian
                position (``(peak_idx + 0.5) * cell_size``) is corrected to
                Eulerian the same way. ``None`` (default) leaves positions
                at their Lagrangian peak location, today's behavior
                unchanged.

        Returns:
            (catalog, claimed) — a :class:`~beorn.structs.HaloCatalog` of
            the accepted halos (each mass exactly one of the walk's
            log-spaced ``M(R)`` nodes, per the module docstring's mass
            convention), and a boolean array (shape of ``delta_field``)
            marking cells excluded by some accepted patch's full R-sphere —
            the latter is what the caller reduces to
            ``CHMFSampler.sample``'s ``deterministic_mass_fraction``.

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
                patch_coords = np.argwhere(patch_mask)
                patch_vals = delta_smoothed[patch_mask]
                peak_idx = patch_coords[np.argmax(patch_vals)]

                if not active[tuple(peak_idx)]:
                    # Already claimed by another patch accepted earlier at
                    # this same scale -- their exclusion spheres can overlap
                    # even when the raw threshold-crossing blobs themselves
                    # do not touch (full exclusion, Bond & Myers 1996b).
                    continue

                n_pix = int(np.count_nonzero(patch_mask))
                filter_mass = self.chmf.rho_m * n_pix * cell_volume
                if filter_mass < min_patch_mass:
                    continue  # leave active -- may resolve at a smaller R

                # Mass is the nominal filter mass M(R) at this crossing
                # scale (see module docstring) -- not filter_mass, which
                # systematically undercounts the true mass and is used only
                # as the size filter above.
                positions_list.append((peak_idx + 0.5) * cell_size)
                masses_list.append(float(M_R))

                active[patch_mask] = False
                _exclude_sphere(active, peak_idx, float(R), cell_size)

        if positions_list:
            positions = np.asarray(positions_list, dtype=np.float32)
            masses = np.asarray(masses_list, dtype=np.float64)
        else:
            positions = np.zeros((0, 3), dtype=np.float32)
            masses = np.zeros(0, dtype=np.float64)

        if displacement_field is not None and positions.shape[0] > 0:
            positions = displace_positions(positions, displacement_field, Lbox)

        logger.debug(
            "Excursion-set finder claimed %d halos at z=%.3f "
            "(M range [%.2e, %.2e] Msun, %d/%d cells claimed)",
            len(masses), z, M_split, M_max,
            int(np.count_nonzero(~active)), active.size,
        )

        catalog = HaloCatalog(positions=positions, masses=masses,
                              parameters=params, redshift=float(z))
        return catalog, ~active
