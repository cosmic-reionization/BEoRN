"""Generic base loader for N-body simulations with merger tree support.

Provides :class:`MergerTreeLoader`, an abstract mid-level class that sits
between :class:`~beorn.load_input_data.base.BaseLoader` and concrete
simulation-specific loaders (e.g.
:class:`~beorn.load_input_data.cosmo_sim_thesan.ThesanLoader`).

Responsibilities handled here so subclasses do not have to repeat them:

- **Main-progenitor branch walking** — given a cached flat tree, walks
  each halo's main progenitor branch over the requested lookback window.
- **Vectorised alpha fitting** — calls
  :func:`~beorn.load_input_data.alpha_fitting.vectorized_alpha_fit` on the
  assembled mass histories to derive per-halo accretion rates.
- **Fallback alpha** — halos not present in the tree, or whose history is
  too short to fit reliably, receive a fallback value controlled by
  ``parameters.source.alpha_fallback``.
- **Alpha range clamping** — per-halo alphas are clipped to the paintable
  range ``[solver.halo_mass_accretion_alpha[0],
  solver.halo_mass_accretion_alpha[-2]]`` so every halo maps to a valid
  profile bin.
- **HaloCatalog assembly** — merges tree-derived alphas with positions and
  masses from the group/FoF catalog returned by the subclass.
"""
from __future__ import annotations

import hashlib
import logging
from abc import abstractmethod
import numpy as np

from .base import BaseLoader
from .alpha_fitting import vectorized_alpha_fit
from ..structs import HaloCatalog, Parameters

logger = logging.getLogger(__name__)


class MergerTreeLoader(BaseLoader):
    """Abstract base loader for simulations with merger trees.

    Subclass this and implement :meth:`load_tree_cache`,
    :meth:`get_halo_information_from_catalog`, and :meth:`load_density_field`.
    Optionally override :meth:`load_rsd_fields` if velocity data is available.
    The higher-level
    :meth:`load_halo_catalog` is implemented here and calls those methods
    automatically.

    The tree cache is a flat HDF5 file containing four or five parallel arrays
    that describe every entry across all snapshots:

    - ``tree_halo_ids``      — subhalo index within its snapshot
    - ``tree_snap_num``      — snapshot index
    - ``tree_mass``          — halo mass (simulation units; conversion is
      done by the subclass in
      :meth:`get_halo_information_from_catalog`)
    - ``tree_main_progenitor`` — index into these arrays pointing to the
      main progenitor entry (one snapshot earlier), or -1 if none
    - ``tree_is_central``    — (optional) bool flag; True for FoF centrals.
      When present, satellite halos are excluded from alpha fitting.

    This minimal schema is simulation-agnostic.  It can be produced from
    LHaloTree (IllustrisTNG / THESAN), Consistent-Trees, SubFind, or any
    other format by writing an extraction notebook that reads the native
    format and writes this simplified cache.

    Args:
        parameters (Parameters): BEoRN parameter container.
    """

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

    # ── Abstract interface ─────────────────────────────────────────────────

    @abstractmethod
    def load_tree_cache(self):
        """Load the simplified merger tree cache from disk.

        Returns:
            tuple of four or five 1D arrays, all with the same length:

            - ``tree_halo_ids``        (int): subhalo index within snapshot
            - ``tree_snap_num``        (int): snapshot index
            - ``tree_mass``            (float): halo mass (raw simulation units)
            - ``tree_main_progenitor`` (int): array index of main progenitor,
              or -1 when there is no progenitor
            - ``tree_is_central``      (bool, optional): True for FoF central
              subhalos.  When present, satellite subhalos are excluded from
              alpha fitting (satellites undergo tidal stripping, producing
              decreasing mass histories and unreliable alpha fits).
        """

    @abstractmethod
    def get_halo_information_from_catalog(
        self, redshift_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load positions, masses, and the subhalo→group mapping for one snapshot.

        Args:
            redshift_index (int): Index into :attr:`redshifts`.

        Returns:
            tuple of three arrays:

            - ``positions``              (N, 3) float — comoving positions in Mpc/h
            - ``masses``                 (N,)   float — halo masses in M☉
            - ``subhalo_to_group_map``   (S,)   int   — for each subhalo entry
              in the tree, the group/FoF index it belongs to (length S ≥ N)
        """

    def load_rsd_fields(self, redshift_index: int):
        """RSD velocity fields — not required for basic post-processing.

        Override in a subclass if particle velocity data is available and
        you want redshift-space-distortion corrections to the 21cm signal.

        Raises:
            NotImplementedError: Always, unless overridden.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide RSD fields. "
            "Override load_rsd_fields() in a subclass if velocity data is available."
        )

    @property
    def catalogs(self) -> list:
        """Sentinel list used by PaintingCoordinator to identify halo-bearing snapshots.

        Every snapshot in a merger-tree loader has a halo catalog, so all
        entries are non-None.  The list length equals ``self.redshifts.size``.
        """
        return [True] * self.redshifts.size

    @property
    def snapshot_numbers(self) -> np.ndarray:
        """Raw simulation snapshot number backing each entry of :attr:`redshifts`.

        ``redshifts`` may be a filtered subset of the simulation's snapshots (by
        redshift range, or by which data products are on disk), so a position in
        that array is **not** interchangeable with the snapshot number stored in
        the tree cache's ``tree_snap_num``.  Anything that cross-references the two
        must translate through this property.

        The default assumes the identity mapping, which is correct for loaders that
        expose every snapshot.  Subclasses that filter must override it.
        """
        return np.arange(self.redshifts.size)

    @property
    def all_snapshot_redshifts(self) -> np.ndarray:
        """Redshift of *every* simulation snapshot, indexed by raw snapshot number.

        Unlike :attr:`redshifts` this is not filtered, so it can be indexed directly
        with ``tree_snap_num`` values.  Progenitor branches step through consecutive
        simulation snapshots regardless of which ones the loader exposes, so fitting
        an accretion rate needs the unfiltered redshifts.

        The default returns :attr:`redshifts`, correct only when
        :attr:`snapshot_numbers` is the identity.  Subclasses that filter must override.
        """
        return np.asarray(self.redshifts)

    @property
    def input_tag(self) -> str:
        """Short identifier for this dataset, used to namespace output files.

        Derived from the simulation code name and the Ncell/Lbox grid settings.
        Subclasses may override to include a path-based hash if multiple
        simulations with the same code name are used simultaneously.
        """
        sim = self.parameters.simulation
        code = getattr(self, "simulation_code", "mergertree")
        h = hashlib.md5(code.encode()).hexdigest()[:8]
        return f"{code.lower().replace('-', '_')}_N{sim.Ncell}_L{int(sim.Lbox)}_{h}"

    # ── Concrete HaloCatalog assembly ─────────────────────────────────────

    def load_halo_catalog(self, redshift_index: int) -> HaloCatalog:
        """Load halo catalog with per-halo accretion rates from the merger tree.

        Halos that are not represented in the tree, or whose mass history is
        too short to fit, receive the fallback alpha value (see
        :meth:`fallback_alpha`).  All alphas are clipped to the paintable
        range defined by ``solver.halo_mass_accretion_alpha``.

        When ``source.alpha_constant`` (or ``source.alpha_constant_z``) is set the merger
        tree is bypassed entirely: every halo receives that value and no tree data is read.

        Args:
            redshift_index (int): Index into :attr:`redshifts`.

        Returns:
            HaloCatalog: Catalog with positions, masses, and per-halo alphas.
        """
        alpha_constant = getattr(self.parameters.source, "alpha_constant", None)
        alpha_constant_z = getattr(self.parameters.source, "alpha_constant_z", None)
        if alpha_constant is None and alpha_constant_z is not None:
            z_table, alpha_table = alpha_constant_z[0], alpha_constant_z[1]
            alpha_constant = float(
                np.interp(self.redshifts[redshift_index], z_table, alpha_table)
            )

        if alpha_constant is not None:
            # Constant-alpha mode: skip the tree read and the progenitor walk.
            positions, masses, _ = self.get_halo_information_from_catalog(redshift_index)
            full_alphas = np.full(masses.size, float(alpha_constant))
        else:
            # 1. Fit alpha from merger tree for halos that appear in it
            tree_halo_ids, halo_alphas = self.get_halo_accretion_rate_from_tree(redshift_index)
            # Halos whose branch is shorter than the lookback window come back as NaN
            # (step 4 turns them into alpha_fb).  They must not poison the mean/median
            # that defines alpha_fb itself, so derive it from the well-fitted halos only.
            alpha_fb = self.fallback_alpha(halo_alphas[np.isfinite(halo_alphas)])

            # 2. Load positions/masses from the group catalog
            positions, masses, subhalo_to_group_map = self.get_halo_information_from_catalog(redshift_index)
            n_groups = masses.size

            # 3. Map tree-derived alphas onto the full group array
            full_alphas = np.full(n_groups, alpha_fb)
            if tree_halo_ids.size > 0:
                sorting = np.argsort(tree_halo_ids)
                sorted_ids = tree_halo_ids[sorting]
                sorted_alphas = halo_alphas[sorting]
                group_ids = subhalo_to_group_map[sorted_ids]
                valid = (group_ids >= 0) & (group_ids < n_groups)
                full_alphas[group_ids[valid]] = sorted_alphas[valid]

            # 4. Replace non-finite values (inf/nan from short/missing histories)
            full_alphas[~np.isfinite(full_alphas)] = alpha_fb

        # 5. Clamp to the paintable alpha range
        alpha_range = self.parameters.solver.halo_mass_accretion_alpha
        if alpha_constant is not None:
            # Nothing to clamp — but a value outside the grid would silently paint
            # every halo with the wrong profile bin, so fail loudly instead.
            if not (alpha_range[0] <= alpha_constant < alpha_range[-1]):
                raise ValueError(
                    f"source.alpha_constant={alpha_constant} lies outside the paintable "
                    f"alpha grid solver.halo_mass_accretion_alpha="
                    f"[{alpha_range[0]}, {alpha_range[-1]}]."
                )
        else:
            below = full_alphas < alpha_range[0]
            above = full_alphas > alpha_range[-2]
            full_alphas[below] = alpha_range[0]
            full_alphas[above] = alpha_range[-2]
            logger.debug(
                f"z-index {redshift_index}: clamped {below.sum()} alphas below "
                f"{alpha_range[0]:.2f} and {above.sum()} above {alpha_range[-2]:.2f}"
            )

        catalog = HaloCatalog(
            positions=positions,
            masses=masses,
            alphas=full_alphas,
            parameters=self.parameters,
            redshift_index=redshift_index,
            redshift=float(self.redshifts[redshift_index]),
        )
        logger.debug(
            f"Catalog z={catalog.redshift:.3f}: "
            f"α min={full_alphas.min():.2f}  max={full_alphas.max():.2f}  "
            f"mean={full_alphas.mean():.2f}  std={full_alphas.std():.2f}"
        )
        return catalog

    # ── Tree walking and alpha fitting ────────────────────────────────────

    def get_halo_accretion_rate_from_tree(
        self, redshift_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Walk main progenitor branches and fit alpha for each halo.

        Halos not represented in the tree are omitted from the returned arrays;
        the caller fills the remainder with the fallback value.

        ``parameters.source.mass_accretion_lookback`` counts **simulation
        snapshots along the branch**, not entries of :attr:`redshifts`.  A branch
        steps through consecutive snapshots whether or not the loader exposes
        them, so the window — and hence the fitted alpha — does not change when
        the loader's snapshot list is filtered.  Halos whose branch ends before
        the window is full are returned as ``NaN`` and picked up by
        :meth:`fallback_alpha` in :meth:`load_halo_catalog`.

        Args:
            redshift_index (int): Index into :attr:`redshifts`.

        Returns:
            tuple:
            - ``halo_ids``  (K,) int   — subhalo indices for fitted halos
            - ``alphas``    (K,) float — fitted alpha, ``NaN`` where the branch
              was too short to fit
        """
        lookback = self.parameters.source.mass_accretion_lookback
        # tree_snap_num holds raw simulation snapshot numbers; redshift_index is a
        # position in the (possibly filtered) redshifts array.  Translate before
        # comparing the two -- they coincide only when no filtering happened.
        snap_now = int(self.snapshot_numbers[redshift_index])
        snapshot_redshifts = np.asarray(self.all_snapshot_redshifts)

        cache = self.load_tree_cache()
        if len(cache) == 5:
            tree_halo_ids, tree_snap_num, tree_mass, tree_main_progenitor, tree_is_central = cache
        else:
            tree_halo_ids, tree_snap_num, tree_mass, tree_main_progenitor = cache
            tree_is_central = None

        current_mask = (tree_snap_num == snap_now) & (tree_mass > 0)
        if tree_is_central is not None:
            current_mask &= tree_is_central
        current_halo_ids = tree_halo_ids[current_mask]
        n_halos = int(current_mask.sum())

        if snap_now + 1 < lookback:
            # Not enough snapshots exist yet for any branch to fill the window.
            logger.warning(
                f"Snapshot {snap_now} has only {snap_now + 1} snapshots behind it "
                f"(lookback {lookback}); returning near-zero fallback alphas."
            )
            return current_halo_ids, np.full(n_halos, 0.04)

        # Walk the main progenitor branch, recording the mass *and* the snapshot it
        # came from.  Column 0 is the halo itself, later columns step back in time.
        mass_history = np.zeros((n_halos, lookback), dtype=np.float64)
        snap_history = np.full((n_halos, lookback), -1, dtype=np.int64)

        pointer = np.flatnonzero(current_mask)
        for step in range(lookback):
            alive = pointer >= 0
            if not alive.any():
                break
            # Clip the -1 terminators before indexing: a raw -1 would wrap around to
            # the last entry of the cache and splice an unrelated halo onto the branch.
            safe = np.where(alive, pointer, 0)
            mass = np.where(alive, tree_mass[safe], 0.0)
            # A non-positive mass ends the usable history: the fit works in log space.
            good = alive & (mass > 0)
            mass_history[:, step] = np.where(good, mass, 0.0)
            snap_history[:, step] = np.where(good, tree_snap_num[safe], -1)
            pointer = np.where(good, tree_main_progenitor[safe], -1)

        # Branches share only a handful of distinct snapshot patterns (main-progenitor
        # links occasionally skip a snapshot), so group identical patterns and fit each
        # group in one vectorised call against its own redshift vector.
        halo_alphas = np.full(n_halos, np.nan, dtype=np.float64)
        lengths = (snap_history >= 0).sum(axis=1)
        fittable = lengths == lookback
        if fittable.any():
            rows_fittable = np.flatnonzero(fittable)
            patterns, inverse = np.unique(
                snap_history[rows_fittable], axis=0, return_inverse=True
            )
            inverse = inverse.ravel()
            # Bucket rows by pattern with one sort rather than rescanning `inverse`
            # per group, which would be O(n_groups * n_halos).
            order = np.argsort(inverse, kind="stable")
            starts = np.concatenate(
                [[0], np.cumsum(np.bincount(inverse, minlength=patterns.shape[0]))]
            )
            for group, pattern in enumerate(patterns):
                rows = rows_fittable[order[starts[group]:starts[group + 1]]]
                if rows.size == 0:
                    continue
                # Snapshot numbers decrease along a branch and snapshot_redshifts is
                # descending in the index, so this is ascending in z (now -> past),
                # which is what vectorized_alpha_fit requires.
                halo_alphas[rows] = vectorized_alpha_fit(
                    snapshot_redshifts[pattern], mass_history[rows]
                )

        n_short = int((~fittable).sum())
        logger.debug(
            f"Alpha fit at snapshot {snap_now}: {n_halos} halos, {n_short} with a branch "
            f"shorter than {lookback} snapshots (-> fallback), "
            f"{int(np.isinf(halo_alphas).sum())} Inf values"
        )
        return current_halo_ids, halo_alphas

    # ── Fallback alpha ────────────────────────────────────────────────────

    def fallback_alpha(self, fitted_alphas: np.ndarray) -> float:
        """Return the fallback alpha for halos not represented in the tree.

        Controlled by ``parameters.source.alpha_fallback``:

        - ``float``    — use that fixed value
        - ``"mean"``   — mean of ``fitted_alphas``
        - ``"median"`` — median of ``fitted_alphas``

        Args:
            fitted_alphas (np.ndarray): Alphas already fitted from the tree
                (used when fallback is ``"mean"`` or ``"median"``).

        Returns:
            float: Fallback value.
        """
        fb = self.parameters.source.alpha_fallback
        if isinstance(fb, (int, float)):
            return float(fb)
        if fb == "mean":
            return float(np.mean(fitted_alphas)) if fitted_alphas.size else 0.6
        if fb == "median":
            return float(np.median(fitted_alphas)) if fitted_alphas.size else 0.6
        raise ValueError(
            f"parameters.source.alpha_fallback must be a float, 'mean', or 'median'; "
            f"got {fb!r}."
        )
