"""LPT-based halo catalog loader using the Conditional Halo Mass Function."""
from __future__ import annotations

import warnings

import numpy as np

from .base import BaseLoader
from ..structs import Parameters, HaloCatalog
from ..lpt import SecondOrderLPT, LPTBase
from ..lpt.chmf import CHMF, CHMFSampler
from ..lpt.linear_power import get_power_spectrum


class LPTHaloLoader(BaseLoader):
    """Generate halo catalogs from an LPT density field via the CHMF.

    The loader pairs a Lagrangian Perturbation Theory solver (default: 2LPT,
    matching ``parameters.cosmo_sim.density_source``'s own default) with a
    :class:`~beorn.lpt.chmf.CHMFSampler` to sample halos from the conditional
    halo mass function at each snapshot.

    Density fields are produced deterministically by the LPT solver (seeded at
    construction).  Halo positions within each cell are drawn from a per-snapshot
    seed derived from ``base_seed + redshift_index``, so repeated calls to
    :meth:`load_halo_catalog` return the same catalog.

    Args:
        parameters:   BEoRN :class:`~beorn.structs.Parameters` object.
        lpt_solver:   Pre-built :class:`~beorn.lpt.LPTBase` instance.  If
                      ``None`` (default) a :class:`~beorn.lpt.SecondOrderLPT`
                      (2LPT) solver is built automatically.
        ps_method:    Power spectrum method used to build the (single, shared
                      — issue #42, O10) :class:`~beorn.lpt.linear_power.PowerSpectrum`
                      instance passed to both the default solver and the CHMF
                      (default ``'eisenstein_hu'``). Ignored if ``lpt_solver``
                      is given — the CHMF then reuses ``lpt_solver.power_spectrum``
                      instead, so both stay on the same cosmology.
        seed:         RNG seed for the LPT initial conditions used to build
                      this loader's own density field. ``None`` (default)
                      reads ``parameters.halo_sim.IC_seed`` (itself ``None``
                      by default, inheriting ``parameters.cosmo_sim.IC_seed``
                      — see :attr:`~beorn.structs.HaloSimParameters.IC_seed`).
                      A ``UserWarning`` is issued if the resulting solver's
                      actual seed (whether resolved this way or via a
                      directly-supplied ``lpt_solver``) differs from
                      ``parameters.cosmo_sim.IC_seed``, since that means the
                      sampled halos will not be spatially correlated with a
                      density field built elsewhere from that seed.
        halo_seed:    RNG seed for halo-catalog generation (Poisson draws +
                      intra-cell position sampling) — independent of ``seed``
                      (the LPT IC seed) so the two never conflict. ``None``
                      (default) reads ``parameters.halo_sim.halo_sampler_seed``.
        R_env:        Environmental smoothing scale in Mpc/h passed to
                      :meth:`~beorn.lpt.chmf.CHMFSampler.sample`.  ``None``
                      (default) reads ``parameters.halo_sim.R_env`` (itself
                      ``None`` — cell size is used as the conditioning scale).
        n_mass_bins:  Number of log-spaced mass bins for the CHMF sampling.
                      ``None`` (default) reads ``parameters.halo_sim.n_mass_bins``.
        delta_c:      Linear collapse threshold. ``None`` (default) reads
                      ``parameters.halo_sim.delta_c``.
        hmf_model:    ``'PS'`` — pure EPS conditional sampling. ``'ST'`` —
                      Barkana & Loeb (2004) hybrid: rescale so the mean mass
                      function matches Sheth-Tormen (as in 21cmFAST-family
                      codes). ``None`` (default) reads
                      ``parameters.halo_sim.hmf_model`` (itself ``'ST'`` by
                      default).
        **ps_kwargs:  Extra keyword arguments forwarded to the power spectrum
                      constructor (e.g. ``wiggle=True``).
    """

    def __init__(
        self,
        parameters: Parameters,
        lpt_solver: LPTBase | None = None,
        ps_method: str = 'eisenstein_hu',
        seed: int | None = None,
        halo_seed: int | None = None,
        R_env: float | None = None,
        n_mass_bins: int | None = None,
        delta_c: float | None = None,
        hmf_model: str | None = None,
        **ps_kwargs,
    ):
        super().__init__(parameters)
        self.R_env = R_env if R_env is not None else parameters.halo_sim.R_env
        self.n_mass_bins = n_mass_bins if n_mass_bins is not None else parameters.halo_sim.n_mass_bins
        self._base_seed = halo_seed if halo_seed is not None else parameters.halo_sim.halo_sampler_seed

        # issue #42, O10: build ONE PowerSpectrum instance and share it with
        # the CHMF below, instead of each independently constructing its own
        # (and each paying its own A_s normalisation). If a pre-built
        # lpt_solver is supplied, its own power_spectrum is reused instead —
        # otherwise the solver and the CHMF could silently run on different
        # cosmologies.
        if lpt_solver is None:
            resolved_seed = seed if seed is not None else parameters.halo_sim.IC_seed
            shared_ps = get_power_spectrum(ps_method, parameters, **ps_kwargs)
            self.lpt_solver = SecondOrderLPT(
                parameters, power_spectrum=shared_ps, seed=resolved_seed, verbose=False,
            )
        else:
            self.lpt_solver = lpt_solver
            shared_ps = lpt_solver.power_spectrum

        # issue #56: this loader builds its own, independently-seeded density
        # field for the CHMF to condition on. If that seed doesn't match
        # cosmo_sim.IC_seed, the sampled halos won't be spatially correlated
        # with a density field built elsewhere from cosmo_sim.IC_seed.
        if self.lpt_solver.seed != parameters.cosmo_sim.IC_seed:
            warnings.warn(
                f"LPTHaloLoader's density-field seed ({self.lpt_solver.seed}) "
                f"differs from parameters.cosmo_sim.IC_seed "
                f"({parameters.cosmo_sim.IC_seed}); the sampled halo catalog "
                "will not be spatially correlated with a density field built "
                "from cosmo_sim.IC_seed elsewhere. Set halo_sim.IC_seed to "
                "match (or leave it None to inherit it) for spatial "
                "correlation between halos and density.",
                stacklevel=2,
            )

        chmf = CHMF(parameters, power_spectrum=shared_ps, delta_c=delta_c)
        self.sampler = CHMFSampler(parameters, chmf=chmf, hmf_model=hmf_model)

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
        # Raw field -- CHMFSampler.sample smooths it to the conditioning
        # scale internally (issue #54).
        delta = self.lpt_solver.get_linear_density(z)
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
