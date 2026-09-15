"""
Global parameters for this simulation. They encompass the astrophysical parameters of the source, the cosmological parameters, the simulation parameters, the solver parameters, the excursion set parameters, and the halo mass function parameters.
Slots are used to prevent the creation of new attributes. This is useful to avoid typos and to have a clear overview of the parameters.
"""

from pathlib import Path
import hashlib
from dataclasses import dataclass, field, is_dataclass, fields
from typing import Literal
import numpy as np
import inspect
import yaml
import h5py
import logging

from .helpers import bin_centers

logger = logging.getLogger(__name__)


@dataclass(slots = True)
class SourceParameters:
    """
    Parameters for the sources of radiation. Sensible defaults are provided.
    """

    source_type: Literal['SED', 'Ghara', 'Ross', 'constant'] = 'SED'
    """source type. SED, Ghara, Ross, constant"""

    energy_min_sed_xray: int = 500
    """minimum energy of normalization of xrays in eV"""

    energy_max_sed_xray: int = 2000
    """maximum energy of normalization of xrays in eV"""

    energy_cutoff_min_xray: int = 500
    """lower energy cutoff for the xray band"""

    energy_cutoff_max_xray: int = 2000
    """upper energy cutoff for the xray band"""

    alS_xray: float = 1.00001
    """TODO"""
    """PL sed Xray part N ~ nu**-alS [nbr of photons/s/Hz]"""

    xray_normalisation: float = 3.4e40
    """Xray normalization [(erg/s) * (yr/Msun)] (astro-ph/0607234 eq22)"""

    n_lyman_alpha_photons: int = 9690
    """number of lyal photons per baryons in stars"""

    lyman_alpha_power_law: float = 0.0
    """power law index for lyal. 0.0 for constant, 1.0 for linear, 2.0 for quadratic"""

    halo_mass_min: float = 1e8
    """Minimum star-forming halo mass [Msun/h]. Objects below this mass are not painted. (finding 13: Msun/h, not Msun -- BEoRN keeps little-h internally.)"""

    halo_mass_max: float = 1e16
    """Maximum star-forming halo mass [Msun/h]. Objects above this mass are not painted."""

    f_st: float = 0.05
    """the prefactor of the star formation efficiency f_star which is a function of halo mass"""

    # --- f_st grid precomputation (used by RadiationProfileFstSolver) ---
    f_st_grid_min: float = 0.01
    """Minimum f_st value for precomputing (mass, alpha, f_st, z) radiation profiles."""

    f_st_grid_max: float = 0.2
    """Maximum f_st value for precomputing (mass, alpha, f_st, z) radiation profiles."""

    f_st_grid_n: int = 30
    """Number of f_st grid points for precomputing (mass, alpha, f_st, z) radiation profiles."""

    # --- stochastic f_st painting controls (used by PaintingCoordinator.paint_single_fstar) ---
    f_st_paint_distribution: Literal['lognormal', 'normal', 'uniform'] = 'lognormal'
    """Distribution used to sample per-halo f_st during painting."""

    f_st_paint_sigma: float = 0.5
    """Width parameter for the f_st sampling distribution (log-space sigma for lognormal).
    Used as the constant scatter when f_st_paint_sigma0 is None (default, backward-compatible)."""

    f_st_paint_sigma0: float | None = None
    """Mass-dependent scatter model: sigma_dex(Mh) = f_st_paint_sigma0 + f_st_paint_sigma1 *
    log10(Mh / f_st_paint_sigma_mpiv), calibrated per fit_thesan_fst.md section 6. ``None``
    (default) falls back to the constant ``f_st_paint_sigma`` for backward compatibility."""

    f_st_paint_sigma1: float = 0.0
    """Slope of the mass-dependent scatter model, in dex per dex of log10(Mh / mpiv).
    ``0.0`` (default) recovers a mass-independent scatter equal to f_st_paint_sigma0."""

    f_st_paint_sigma_mpiv: float = 1e11
    """Pivot halo mass [Msun/h] for the mass-dependent scatter model."""

    f_st_paint_min: float = 0.01
    """Lower clipping bound for sampled f_st during painting."""

    f_st_paint_max: float = 0.2
    """Upper clipping bound for sampled f_st during painting."""

    f_st_paint_seed: int | None = None
    """Optional RNG seed for reproducible per-snapshot f_st sampling."""

    Mp: float = 2.8e11 * 0.68
    """pivot mass of the double power law describing the star formation rate"""

    g1: float = 0.49
    """power law index of the star formation rate"""

    g2: float = -0.61
    """power law index of the star formation rate"""

    Mt: float = 1e8
    """turnover mass of the low mass suppression term of the star formation rate"""

    g3: float = 4
    """power law index of the low mass suppression term of the star formation rate"""

    g4: float = -1
    """power law index of the low mass suppression term of the star formation rate"""

    Nion: int = 5000
    """number of ionizing photons per baryon in stars"""

    f0_esc: float = 0.2
    """photon escape fraction f_esc = f0_esc * (Mp/M)^pl_esc"""

    Mp_esc: float = 1e10
    """pivot mass for the escape fraction"""

    pl_esc: float = 0.0
    """power law index for the escape fraction"""

    min_xHII_value: int = 0
    """lower limit for the ionization fraction. All pixels with xHII < min_xHII_value will be set to this value."""

    mass_accretion_lookback: int = 10
    """Number of snapshots to look back when fitting the per-halo accretion rate alpha from merger trees.
    The thesis by Moll (2025) shows that the mean alpha stabilises at n=10 lookback snapshots,
    corresponding to a causal timescale of ~300 Myr (Δz≈4 from z=8). Values below 5 give
    unstable fits; going beyond 10 only marginally reduces scatter."""

    alpha_fallback: "float | str" = "mean"
    """Fallback alpha value for halos not found in the merger tree, or whose mass history
    is too short to fit reliably.  Options:
    - float  : fixed value (e.g. 0.6, the typical mean from THESAN-DARK 2 at z~8)
    - 'mean' : mean of the fitted alphas at that snapshot (default — adapts with redshift)
    - 'median': median of the fitted alphas at that snapshot
    """

    alpha_constant: "float | None" = None
    """Constant mass-accretion alpha assigned to *every* halo, bypassing the merger tree.

    ``None`` (default) fits alpha per halo from the merger-tree mass history (see
    ``mass_accretion_lookback`` and ``alpha_fallback``).  When set to a float, no tree
    data is read or walked at all and every halo receives this value — useful for a
    deterministic run at a measured population mean (e.g. 0.4577 for THESAN-1), and much
    cheaper since the multi-GB tree cache is never touched.  The value should lie inside
    ``solver.halo_mass_accretion_alpha``, which still clamps it."""

    alpha_constant_z: "np.ndarray | None" = None
    """Redshift-dependent constant mass-accretion alpha, as an alternative to
    ``alpha_constant``.

    Shape ``(2, N)``: row 0 is ascending redshifts, row 1 is the matching alpha value.
    When set (and ``alpha_constant`` is ``None``), every halo at a given snapshot still
    receives one shared value — same tree-bypass cost savings as ``alpha_constant`` — but
    that value is ``np.interp``'d from this table at the snapshot's redshift, rather than
    fixed for the whole run.  Values outside the table's z-range are flat-extrapolated
    (``np.interp``'s default).  Not read from parameter yaml files; set directly in Python,
    like ``solver.halo_mass_accretion_alpha``."""

    t_source_age: float = None
    """Maximum source age in Myr.  When set, the X-ray emission integral (``rho_xray``) is
    limited to a lookback window of this duration rather than integrating all the way back
    to ``solver.z_source_start``, which prevents unphysically old emission histories for
    halos that formed recently.  Only the X-ray integral honours it: the Lyman-alpha
    integral (``rho_alpha_profile``) always looks back to ``solver.z_source_start``, and the
    bubble and heating ODEs integrate over ``solver.redshifts``.  ``None`` (default)
    integrates back to ``solver.z_source_start``.
    """



@dataclass(slots = True)
class SolverParameters:
    """
    Solver parameters for the simulation.
    """
    redshifts: np.ndarray = field(default_factory=lambda: np.arange(25, 6, -0.5))
    """High-resolution redshift grid used by the 1D RT profile solver.
    Should span the full redshift range of interest at fine enough resolution for accurate profile integration.
    Stored inside the RadiationProfiles cache — does not need to be written to igm_data/igm_params.yaml."""

    fXh: Literal['constant', 'variable'] = 'constant'
    """if fXh is constant here, it will take the value 0.11. Otherwise, we will compute the free e- fraction in neutral medium and take the fit fXh = xe**0.225"""

    halo_mass_accretion_alpha: np.ndarray = field(default_factory=lambda: np.linspace(0.1, 0.9, 10))
    """Coefficient for exponential mass accretion. Since beorn distinguishes between accretion rates a range should be specified"""

    halo_mass_bin_min: float = 1e5
    """Minimum halo mass bin edge [Msun/h] (finding 13: Msun/h, not Msun)."""

    halo_mass_bin_max: float = 1e14
    """Maximum halo mass bin edge [Msun/h]."""

    halo_mass_nbin: int = 100
    """Number of mass bins."""

    HI_frac: float = 1 - 0.08
    """HI number fraction. Only used when running H_He_Final."""

    clumping: int = 1
    """Rescale the background density. Set to 1 to get the normal 2h profile term."""

    z_decoupling: int = 135
    """Redshift at which the gas decouples from CMB and starts cooling adiabatically."""

    z_source_start: float = 35.0
    """Redshift at which sources start emitting: the maximum lookback redshift of the X-ray
    (``rho_xray``) and Lyman-alpha (``rho_alpha_profile``) emission integrals.  For X-rays
    only, ``source.t_source_age`` can cap the window further (whichever limit is reached
    first applies).  ``R_bubble`` and ``rho_heat`` do not read it: they integrate over
    ``solver.redshifts``, so extend that grid up to this redshift for them to start here."""

    ode_rtol: float = 1e-2
    """Relative tolerance for ODE integrations (R_bubble, rho_heat).
    Looser values speed up the solver; tighten if profiles show numerical artefacts."""

    ode_atol: float = 1e-2
    """Absolute tolerance for ODE integrations (R_bubble, rho_heat)."""

    ode_method: str = 'RK45'
    """Integration method passed to scipy.integrate.solve_ivp for R_bubble and rho_heat.
    'RK45' (default) is fine for non-stiff or mildly stiff systems.
    'LSODA' auto-switches between Adams and BDF and is a good all-round choice.
    'Radau' or 'BDF' are best when the system is strongly stiff (large recombination
    rates or very fine redshift grids at high z)."""

    # derived properties that are directly related to the parameters
    @property
    def halo_mass_bins(self) -> np.ndarray:
        return np.logspace(np.log10(self.halo_mass_bin_min), np.log10(self.halo_mass_bin_max), self.halo_mass_nbin, base=10)

    @property
    def halo_mass_bin_centers(self) -> np.ndarray:
        return bin_centers(self.halo_mass_bins)

    @property
    def halo_mass_accretion_alpha_bin_centers(self) -> np.ndarray:
        return bin_centers(self.halo_mass_accretion_alpha)

    def __post_init__(self):
        if isinstance(self.redshifts, list):
            self.redshifts = np.array(self.redshifts)
        if isinstance(self.halo_mass_accretion_alpha, list):
            self.halo_mass_accretion_alpha = np.array(self.halo_mass_accretion_alpha)



@dataclass(slots = True)
class SimulationParameters:
    """
    Parameters that are used to run the simulation. These are used in the generation of the halo profiles and when converting the halo profiles to a grid.
    """

    Ncell: int = 128
    """Number of pixels of the final grid. This is the number of pixels in each dimension. The total number of pixels will be Ncell^3."""

    Lbox: float = 100
    """Box length, in [Mpc/h]. This is the length of the box in each dimension. The total volume will be Lbox^3."""

    store_grids: list = ('delta_b', 'Grid_Temp', 'Grid_xHII', 'Grid_xal')
    """Base grids to write to the HDF5 output file. These four fields are the independent outputs of the painting stage.
    Derived quantities such as 'Grid_dTb' are *not* stored by default because they can be recomputed on the fly
    as cached properties from the base fields (``Grid_dTb = f(delta_b, Grid_Temp, Grid_xHII, Grid_xal, z)``).
    Add 'Grid_dTb' here only if you need pre-computed access to it for very large grids where recomputation is expensive."""

    cores: int = 1
    """Number of cores used in parallelization. The computation for each redshift can be parallelized with a shared memory approach. This is the number of cores used for this. Keeping the number at 1 disables parallelization."""

    fft_backend: str = 'auto'
    """FFT backend for 3D convolutions during painting.  ``'auto'`` (default)
    selects the fastest available backend: jax (GPU/TPU) > torch (GPU) > numpy
    (CPU via scipy/pocketfft).  Override with ``'numpy'``, ``'jax'``, or
    ``'torch'``.  GPU backends process mass bins sequentially on the device
    (the GPU provides internal parallelism); numpy uses :attr:`cores` worker
    processes."""

    spreading_pixel_threshold: int = -1
    """UNUSED / dead parameter (finding 13). Intended: when spreading the excess ionization
    fraction, treat connected regions smaller than this as one region to speed up. No code in
    src/beorn reads it (spreading_excess_fast ignores it); many yamls set it to -1 out of
    habit. Kept only so those yamls still load. Wire it into spread.py or drop it + the yaml keys."""

    spreading_subgrid_approximation: bool = True
    """UNUSED / dead parameter (finding 13). Intended: toggle the distance_transform_edt subgrid
    approximation during excess-ionization spreading. Not read anywhere in src/beorn."""

    minimum_grid_size_heat: int = 4
    """Minimum grid size used when computing the heat kernel from its associated profile."""

    minimum_grid_size_lyal: int = 16
    """Minimum grid size used when computing the lyal kernel from its associated profile."""

    compute_s_alpha_fluctuations: bool = True
    """Whether or not to include the fluctuations in the suppression factor S_alpha when computing the x_al fraction."""

    compute_x_coll_fluctuations: bool = True
    """Whether or not to include the fluctuations in the collisional coupling coefficient x_coll when computing the x_tot fraction."""

    degrade_resolution: int = 1
    """Downsample density grids read from N-body files by this integer factor before painting.
    A value of 1 (default) applies no degradation. A value of N block-averages each N³ voxel
    into one, e.g. degrade_resolution=4 turns a 256³ grid into 64³.
    Set Ncell to the native grid size divided by degrade_resolution."""

    @property
    def kbins(self) -> np.ndarray:
        """
        Returns the k bins for the power spectrum. The bins are logarithmically spaced between k_min and k_max.
        The number of bins is determined by the size of the simulation box and the number of cells.
        """
        k_min = 1 / self.Lbox
        k_max = self.Ncell / self.Lbox
        # TODO - explain the factor of 6
        bin_count = int(6 * np.log10(k_max / k_min))

        return np.logspace(np.log10(k_min), np.log10(k_max), bin_count, base=10)

    def __post_init__(self):
        # ensure the items of the store_grids are strings. When loading from hdf5 they might be bytes
        self.store_grids = [s.decode() if isinstance(s, bytes) else s for s in self.store_grids]



@dataclass(slots = True)
class CosmologyParameters:
    """
    Cosmological parameters for the simulation.

    Attributes:
        Om: Matter density parameter.
        Ob: Baryon density parameter.
        Ol: Dark energy density parameter.
        rho_c: Critical density of the universe.
        h0: Dimensionless Hubble parameter.
        sigma_8: Amplitude of the matter power spectrum on 8 Mpc/h scales.
        ns: Scalar spectral index.
    """

    # TODO - remove and set an astropy cosmology object instead of individual parameters
    Om: float = 0.315
    Ob: float = 0.045
    Ol: float = 1 - 0.315
    rho_c: float = 2.775e11
    h0: float = 0.673
    sigma_8: float = 0.83
    ns: float = 0.96


@dataclass(slots = True)
class CosmoSimParameters:
    """Parameters specific to N-body/cosmo-sim inputs (py21cmfast, Thesan, PKDGrav, etc.)."""

    py21cmfast_high_res_factor: int = 3
    """Resolution enhancement factor for py21cmfast internal grid (DIM = Ncell * py21cmfast_high_res_factor).
    A larger factor resolves lower halo masses at the cost of more memory and compute time.
    The minimum resolvable halo mass scales roughly as (Lbox / DIM)^3."""

    random_seed: int = 12345
    """Random seed for the random number generator. This is used to generate the random numbers for the halo catalogs and the density fields when using 21cmfast."""

    halo_catalogs_thesan_mass_assignment: Literal['NGP', 'CIC'] = 'CIC'
    """Method used to assign the halo mass to the grid. Can be either NGP (Nearest Grid Point) or CIC (Cloud In Cell)."""

    snapshot_redshifts: np.ndarray = None
    """Redshifts of the cosmo-sim snapshots that will be painted (e.g. py21cmfast outputs).
    Can be a coarse subset of ``solver.redshifts`` — even 1–2 values.
    If ``None``, the full ``solver.redshifts`` profile grid is used for painting too (backward-compatible default).
    Inferred from filenames on disk; not written to igm_data/igm_params.yaml."""

    file_root: Path = None

    particle_mapping_backend: str = 'auto'
    """Backend used by :func:`beorn.particle_mapping.map_particles_to_mesh` when
    painting particle snapshots onto a grid.  ``'auto'`` (default) selects the
    fastest available backend: jax (GPU) > torch (GPU) > numba (CPU JIT) > numpy.
    Override with ``'numpy'``, ``'numba'``, ``'pylians'``, ``'torch'``, or ``'jax'``."""

    def __post_init__(self):
        if isinstance(self.snapshot_redshifts, list):
            self.snapshot_redshifts = np.array(self.snapshot_redshifts)
        if isinstance(self.file_root, str):
            self.file_root = Path(self.file_root)


@dataclass(slots = True)
class Parameters:
    """
    Group all the parameters for the simulation.
    """
    source: SourceParameters = field(default_factory = SourceParameters)
    """source parameters"""
    solver: SolverParameters = field(default_factory = SolverParameters)
    """solver parameters"""
    cosmology: CosmologyParameters = field(default_factory = CosmologyParameters)
    """cosmological parameters"""
    simulation: SimulationParameters = field(default_factory = SimulationParameters)
    """simulation parameters"""
    cosmo_sim: CosmoSimParameters = field(default_factory = CosmoSimParameters)
    """cosmo-sim input parameters (py21cmfast, Thesan, PKDGrav, etc.)"""


    # ── KNOWN ISSUE: fragile hashing (fix_plan_2026-09-03 finding 12) ──────────────
    # unique_hash / profiles_hash / beorn_hash all MD5 the *string form* of a dict that
    # contains numpy scalars and arrays (via `f"{d}"` / `str(d)`). A numpy object's string
    # is a display convention, not a serialization contract, so this hash is NOT stable:
    #   1. np.set_printoptions(precision=...) changes the string -> different hash for the
    #      same parameters (a cache MISS), and rounding at low precision makes DIFFERENT
    #      parameter sets stringify identically -> a hash COLLISION (silent wrong-cache reuse);
    #   2. the `threshold` print option truncates long arrays with "..." -> collisions on
    #      arrays that differ only in the hidden middle;
    #   3. numpy's scalar/array repr changed between numpy 1.x and 2.x, so a numpy upgrade
    #      re-hashes identical parameters and invalidates every cache.
    # RECOMMENDED FIX (do as its own commit -- it changes EVERY hash and so re-baselines all
    # caches and painted outputs): serialize deterministically before hashing --
    #   json.dumps(_to_jsonable(d), sort_keys=True)  where _to_jsonable converts arrays with
    #   .tolist() and formats floats with a fixed form (e.g. f"{x:.17g}") -- none of which
    #   depends on print options or the numpy version. Add a test asserting the hash is
    #   invariant under np.set_printoptions changes and across numpy string forms.
    # Left as-is for now: within one numpy version at default print options the hash is
    # self-consistent (all existing runs are fine); the fix is a deliberate re-baseline.
    def unique_hash(self) -> str:
        """
        Generates a unique hash for the current set of parameters. This can be used as a unique key when caching the computations.
        """
        dict_params = to_dict(self)
        # using the string representation of the dictionary is not optimal because it is not guaranteed to be the same for the same dictionary (if the order of the keys is different for instance)
        # but the key is that the hashes are guaranteed to be different for unique parameter sets
        # See the "KNOWN ISSUE: fragile hashing (finding 12)" note above.
        dict_string = f"{dict_params}"

        return hashlib.md5(dict_string.encode()).hexdigest()

    def profiles_hash(self) -> str:
        """Short MD5 hash of parameters that affect the 1D radiation profiles.

        Covers source parameters, cosmology, solver redshifts, the halo
        mass / accretion-rate bins, and the ODE tolerances and method (they change
        R_bubble and rho_heat, so a cube solved at other tolerances must not be
        reused; review_2026-09-14 finding 4).  Intentionally excludes random seed, grid
        dimensions (Ncell, Lbox, py21cmfast_high_res_factor), and other
        simulation parameters that do not influence the 1D profile shapes.
        This allows profiles to be reused when re-running BEoRN with a
        different py21cmfast seed or a different grid resolution.
        """
        d = {
            'source': to_dict(self.source),
            'cosmology': to_dict(self.cosmology),
            'redshifts': list(self.solver.redshifts),
            'fXh': self.solver.fXh,
            'halo_mass_bin_min': self.solver.halo_mass_bin_min,
            'halo_mass_bin_max': self.solver.halo_mass_bin_max,
            'halo_mass_nbin': self.solver.halo_mass_nbin,
            'halo_mass_accretion_alpha': list(self.solver.halo_mass_accretion_alpha),
            'HI_frac': self.solver.HI_frac,
            'clumping': self.solver.clumping,
            'z_decoupling': self.solver.z_decoupling,
            'z_source_start': self.solver.z_source_start,
            't_source_age': self.source.t_source_age,
            'ode_rtol': self.solver.ode_rtol,
            'ode_atol': self.solver.ode_atol,
            'ode_method': self.solver.ode_method,
        }
        return hashlib.md5(str(d).encode()).hexdigest()[:8]

    # Keys in the source section that control stochastic painting but do not
    # affect the shape of the precomputed 1D radiation profiles.
    _PAINT_ONLY_SOURCE_KEYS = frozenset({
        "f_st_paint_distribution",
        "f_st_paint_sigma",
        "f_st_paint_sigma0",
        "f_st_paint_sigma1",
        "f_st_paint_sigma_mpiv",
        "f_st_paint_min",
        "f_st_paint_max",
        "f_st_paint_seed",
    })

    def profiles_fstar_hash(self) -> str:
        """Like :meth:`profiles_hash` but also strips f_st painting parameters.

        The f_st-grid profiles depend on the physics and the grid bounds/resolution
        (f_st_grid_min/max/n), but not on how halos are sampled from that grid
        during painting.  This hash therefore stays stable across runs that differ
        only in f_st_paint_seed / sigma / distribution, allowing the expensive
        profile cube to be shared.
        """
        source_dict = {k: v for k, v in to_dict(self.source).items()
                       if k not in self._PAINT_ONLY_SOURCE_KEYS}
        d = {
            'source': source_dict,
            'cosmology': to_dict(self.cosmology),
            'redshifts': list(self.solver.redshifts),
            'fXh': self.solver.fXh,
            'halo_mass_bin_min': self.solver.halo_mass_bin_min,
            'halo_mass_bin_max': self.solver.halo_mass_bin_max,
            'halo_mass_nbin': self.solver.halo_mass_nbin,
            'halo_mass_accretion_alpha': list(self.solver.halo_mass_accretion_alpha),
            'HI_frac': self.solver.HI_frac,
            'clumping': self.solver.clumping,
            'z_decoupling': self.solver.z_decoupling,
            'z_source_start': self.solver.z_source_start,
            't_source_age': self.source.t_source_age,
            'ode_rtol': self.solver.ode_rtol,
            'ode_atol': self.solver.ode_atol,
            'ode_method': self.solver.ode_method,
        }
        return hashlib.md5(str(d).encode()).hexdigest()[:8]

    def to_yaml(self, path: Path, exclude_keys: "set[str] | None" = None) -> None:
        """Write parameters to a human-readable YAML file at *path*.

        Args:
            path: Destination file path.
            exclude_keys: Optional set of strings to omit.  Two forms are
                supported:

                - ``"section"`` — remove the entire top-level section,
                  e.g. ``{"simulation", "cosmo_sim"}``.
                - ``"section.field"`` — remove a single field within a
                  section, e.g. ``{"solver.redshifts"}``.
        """
        def _yaml_safe(obj):
            if isinstance(obj, dict):
                return {k: _yaml_safe(v) for k, v in obj.items()}
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, Path):
                return obj.as_posix()
            return obj

        safe = _yaml_safe(to_dict(self))
        if exclude_keys:
            for dotted in exclude_keys:
                section, _, key = dotted.partition('.')
                if not key:
                    safe.pop(section, None)
                elif section in safe and key in safe[section]:
                    del safe[section][key]
        with Path(path).open('w') as f:
            yaml.dump(safe, f, default_flow_style=False, sort_keys=False)

    def summary_str(self) -> str:
        """Return a concise human-readable summary of the key model parameters."""
        src = self.source
        cos = self.cosmology
        sim = self.simulation
        slv = self.solver
        cosmo_sim = self.cosmo_sim
        z_min = slv.redshifts.min()
        lines = [
            "=" * 60,
            "BEoRN model summary",
            "=" * 60,
            f"  Cosmology   : Om={cos.Om}, Ob={cos.Ob}, h0={cos.h0}, sigma_8={cos.sigma_8}",
            f"  Grid        : Ncell={sim.Ncell}, Lbox={sim.Lbox} Mpc/h",
            f"  Profile z   : z={slv.redshifts[0]:.1f} -> {slv.redshifts[-1]:.1f} ({slv.redshifts.size} steps)",
            *(
                [f"  Snapshot z  : z={cosmo_sim.snapshot_redshifts[0]:.1f} -> {cosmo_sim.snapshot_redshifts[-1]:.1f} ({cosmo_sim.snapshot_redshifts.size} snapshots)"]
                if cosmo_sim.snapshot_redshifts is not None else []
            ),
            f"  1D RT bins  : {slv.halo_mass_bin_min:.1e} - {slv.halo_mass_bin_max:.1e} Msun/h at z={z_min:.1f} ({slv.halo_mass_nbin} bins, traced back via exp. accretion)",
            f"  Source      : f_st={src.f_st}, Nion={src.Nion}, f0_esc={src.f0_esc}, pl_esc={src.pl_esc}",
            f"  X-ray       : norm={src.xray_normalisation:.2e}, E=[{src.energy_cutoff_min_xray}, {src.energy_cutoff_max_xray}] eV",
            f"  Lyman-alpha : n_phot={src.n_lyman_alpha_photons}, star-forming above {src.halo_mass_min:.1e} Msun/h",
            f"  Beorn hash  : {self.beorn_hash()}",
            "=" * 60,
        ]
        return "\n".join(lines)

    # ── KNOWN ISSUE: beorn_hash couples the output name to parallelism (review_2026-09-14 2c) ──
    # beorn_hash hashes all of to_dict(self.simulation), which includes `cores` and `fft_backend`.
    # Neither changes the painted physics, but changing either renames the
    # igm_data_<tag>_<hash> output directory, so a run interrupted and resumed with a different
    # rank layout or FFT backend silently starts again in a new directory instead of resuming.
    # OPERATIONAL RULE: do not change simulation.cores or simulation.fft_backend between the
    # launch and the resume of one run.
    # Not fixed here: excluding them renames every existing output directory, the same kind of
    # re-baseline as the fragile-hashing fix above (finding 12), so both belong in one deliberate
    # commit rather than two.
    def beorn_hash(self) -> str:
        """Short MD5 hash of BEoRN-specific parameters (source, solver, simulation).

        Cosmology is intentionally excluded — it is already encoded in the
        input data directory name (e.g. the py21cmfast subdirectory).  This
        hash therefore differentiates astrophysical models applied to the
        same underlying density/halo data.

        ``cosmo_sim`` is also excluded: it controls *which* input data is used
        but does not affect the underlying physics model — it is already encoded
        in the input_tag.
        """
        d = {
            'source': to_dict(self.source),
            'solver': to_dict(self.solver),
            'simulation': to_dict(self.simulation),
        }
        return hashlib.md5(str(d).encode()).hexdigest()[:8]


    @classmethod
    def from_dict(cls, params_dict: dict) -> 'Parameters':
        """
        Create a Parameters object from a dictionary. This is useful for loading parameters from a file.
        """
        params = cls()
        for key, value in params_dict.items():
            if type(value) is dict and hasattr(params, key):
                # Dynamically get the class from the field type annotation
                field_type = type(getattr(params, key))
                # the subparameter is a dataclass, so we can instantiate it with the dict
                child = field_type(**value)
                setattr(params, key, child)
            else:
                raise ValueError(f"Unknown parameter {key} with value {value}. Please check the parameters dictionary.")
        return params


    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'Parameters':
        """
        Create a Parameters object from a YAML file.
        """
        with yaml_path.open('r') as file:
            params_dict = yaml.safe_load(file)
        return cls.from_dict(params_dict)


    @classmethod
    def from_group(cls, group: h5py.Group) -> 'Parameters':
        """
        Create a Parameters object from an hdf5 group.
        This is useful for loading parameters from an hdf5 file.
        """
        params_dict = {}
        for param_field in fields(cls):
            field_name = param_field.name
            # check if the nested field would be a dataclass as well
            if is_dataclass(param_field.type):
                # iterate over the fields of the dataclass
                sub_group = group[field_name]
                sub_params_dict = {}
                for sub_field in fields(param_field.type):
                    sub_field_name = sub_field.name
                    if sub_field_name in sub_group.attrs:
                        sub_params_dict[sub_field_name] = sub_group.attrs[sub_field_name]
                    elif sub_field_name in sub_group:
                        # this is a dataset
                        sub_params_dict[sub_field_name] = sub_group[sub_field_name][...]
                    else:
                        # some configurations result in empty fields (e.g. the file_root might not be set when using a mock simulation)
                        logger.debug(f"Did not find field {sub_field_name} in group {field_name}.")

                params_dict[field_name] = sub_params_dict

            else:
                logger.warning(f"Not a dataclass: {field_name}. Is this expected?")
                params_dict[field_name] = group[field_name][:]

        return cls.from_dict(params_dict)



def to_dict(obj: dataclass) -> dict:
    """
    Convert a dataclass object to an hdf5-compatible dictionary.
    """
    out = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        if is_dataclass(value):
            # recursively convert dataclass to dict
            out[f.name] = to_dict(value)
        elif isinstance(value, (list, tuple)):
            # ensure the types are writable to hdf5 (finding 14: guard the empty case,
            # value[0] would otherwise IndexError).
            if len(value) and isinstance(value[0], Path):
                out[f.name] = [v.as_posix() for v in value]
            else:
                out[f.name] = value
        elif isinstance(value, Path):
            # convert Path to string
            out[f.name] = value.as_posix()
        elif callable(value):
            # convert callable to its source code
            # this is a bit of a hack but it guarantees a unique hash
            out[f.name] = inspect.getsource(value)
        else:
            out[f.name] = value

    return out
