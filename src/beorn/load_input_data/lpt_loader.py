"""LPT-based halo catalog loader using the Conditional Halo Mass Function."""
from __future__ import annotations

import numpy as np

from .base import BaseLoader
from ..structs import Parameters, HaloCatalog
from ..lpt import ZeldovichApproximation, LPTBase
from ..lpt.chmf import CHMF, CHMFSampler


class LPTHaloLoader(BaseLoader):
    """Generate halo catalogs from an LPT density field via the CHMF.

    The loader pairs a Lagrangian Perturbation Theory solver (default:
    Zel'dovich Approximation) with a :class:`~beorn.lpt.chmf.CHMFSampler` to
    sample halos from the conditional halo mass function at each snapshot.

    Density fields are produced deterministically by the LPT solver (seeded at
    construction).  Halo positions within each cell are drawn from a per-snapshot
    seed derived from ``base_seed + redshift_index``, so repeated calls to
    :meth:`load_halo_catalog` return the same catalog.

    Args:
        parameters:   BEoRN :class:`~beorn.structs.Parameters` object.
        lpt_solver:   Pre-built :class:`~beorn.lpt.LPTBase` instance.  If
                      ``None`` (default) a :class:`~beorn.lpt.ZeldovichApproximation`
                      solver is built automatically.
        ps_method:    Power spectrum method forwarded to the solver and
                      :class:`~beorn.lpt.chmf.CHMF` (default
                      ``'eisenstein_hu'``).
        seed:         RNG seed for the LPT initial conditions
                      (default ``42``).
        R_env:        Environmental smoothing scale in Mpc/h passed to
                      :meth:`~beorn.lpt.chmf.CHMFSampler.sample`.  ``None``
                      (default) uses the cell size as the conditioning scale.
        n_mass_bins:  Number of log-spaced mass bins for the CHMF sampling
                      (default ``40``).
        delta_c:      Linear collapse threshold (default ``1.686``).
        **ps_kwargs:  Extra keyword arguments forwarded to the power spectrum
                      constructor (e.g. ``wiggle=True``).
    """

    def __init__(
        self,
        parameters: Parameters,
        lpt_solver: LPTBase | None = None,
        ps_method: str = 'eisenstein_hu',
        seed: int = 42,
        R_env: float | None = None,
        n_mass_bins: int = 40,
        delta_c: float = 1.686,
        **ps_kwargs,
    ):
        super().__init__(parameters)
        self.R_env = R_env
        self.n_mass_bins = n_mass_bins
        self._base_seed = seed

        if lpt_solver is None:
            self.lpt_solver = ZeldovichApproximation(
                parameters, ps_method=ps_method, seed=seed,
                verbose=False, **ps_kwargs,
            )
        else:
            self.lpt_solver = lpt_solver

        chmf = CHMF(parameters, ps_method=ps_method, delta_c=delta_c, **ps_kwargs)
        self.sampler = CHMFSampler(parameters, chmf=chmf)

        # Top-hat scale for linear density field conditioning.
        # Using the sphere-equivalent radius for the cell volume so that
        # Var[delta_env] = sigma^2(M_env, z) exactly (EPS requirement).
        if R_env is not None:
            self._R_tophat = R_env
        else:
            cell = parameters.simulation.Lbox / parameters.simulation.Ncell
            self._R_tophat = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * cell

    # ------------------------------------------------------------------
    # BaseLoader interface
    # ------------------------------------------------------------------

    @property
    def redshifts(self) -> np.ndarray:
        """Snapshot redshifts from ``parameters.cosmo_sim.snapshot_redshifts``
        if set, otherwise ``parameters.solver.redshifts``."""
        snap = self.parameters.cosmo_sim.snapshot_redshifts
        return snap if snap is not None else self.parameters.solver.redshifts

    def load_density_field(self, redshift_index: int) -> np.ndarray:
        """Return the LPT matter overdensity delta(x) at the given snapshot.

        Args:
            redshift_index: Index into :attr:`redshifts`.

        Returns:
            Mean-zero overdensity array of shape ``(Ncell, Ncell, Ncell)``.
        """
        z = float(self.redshifts[redshift_index])
        return self.lpt_solver.get_density(z)

    def load_halo_catalog(self, redshift_index: int) -> HaloCatalog:
        """Generate a halo catalog at the given snapshot via the CHMF.

        The density field is computed from the LPT solver and passed to
        :meth:`~beorn.lpt.chmf.CHMFSampler.sample`.  Halo sampling uses a
        deterministic seed ``base_seed ^ redshift_index`` so repeated calls
        return identical catalogs.

        Args:
            redshift_index: Index into :attr:`redshifts`.

        Returns:
            :class:`~beorn.structs.HaloCatalog` with positions in Mpc/h and
            masses in M_sun.
        """
        z = float(self.redshifts[redshift_index])
        # Use the linear density field (IRFFT of D1*delta_k) rather than CIC
        # particle painting to avoid shot noise, which would produce extreme
        # overdensities in individual cells and contaminate the CHMF near M_env.
        delta = self.lpt_solver.get_linear_density(z, R_tophat=self._R_tophat)
        # XOR with index for a unique but reproducible per-snapshot seed
        sample_seed = self._base_seed ^ redshift_index
        return self.sampler.sample(
            delta_field=delta,
            z=z,
            R_env=self.R_env,
            n_mass_bins=self.n_mass_bins,
            seed=sample_seed,
        )

    def load_rsd_fields(self, redshift_index: int):
        """RSD fields are not available via this loader.

        Use :attr:`lpt_solver`.get_velocity() directly to obtain the
        peculiar velocity field.
        """
        raise NotImplementedError(
            "LPTHaloLoader does not provide RSD meshes. "
            "Call self.lpt_solver.get_velocity(z) to get (vx, vy, vz) in km/s."
        )
