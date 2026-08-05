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
    """Minimum mass of star forming halo. Mdark in HM. Objects below this mass are not considered during the painting process"""

    halo_mass_max: float = 1e16
    """Maximum mass of star forming halo. Objects above this mass are not considered during the painting process"""

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
    """Width parameter for the f_st sampling distribution (log-space sigma for lognormal)."""

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
    """photon escape fraction f_esc = f0_esc * (M/Mp)^pl_esc"""

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

    t_source_age: float = None
    """Maximum source age in Myr.  When set, the X-ray and ionisation integrals
    are limited to a lookback window of this duration rather than integrating
    all the way back to ``solver.z_source_start``.  This prevents unphysically
    old emission histories for halos that formed recently.  ``None`` (default)
    preserves the original behaviour (integrate back to ``solver.z_source_start``).
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
    """Minimum halo mass bin in solar masses."""

    halo_mass_bin_max: float = 1e14
    """Maximum halo mass bin in solar masses."""

    halo_mass_nbin: int = 100
    """Number of mass bins."""

    HI_frac: float = 1 - 0.08
    """HI number fraction. Only used when running H_He_Final."""

    clumping: int = 1
    """Rescale the background density. Set to 1 to get the normal 2h profile term."""

    z_decoupling: int = 135
    """Redshift at which the gas decouples from CMB and starts cooling adiabatically."""

    z_source_start: float = 35.0
    """Maximum lookback redshift for X-ray and ionisation integrals.  Sources are
    assumed to have started emitting no earlier than this redshift.  When
    ``source.source_age`` is ``None`` (default), the integral extends all the way
    back to ``z_source_start``.  When ``source_age`` is set, the window is further
    capped by the finite age — whichever limit is reached first applies."""

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
class BackendParameters:
    """Array-library backend selection for this simulation's compute-heavy
    steps: painting/mass-assignment, the LPT solver's internal FFTs, and the
    halo mass function.

    Each stage below is ``None`` by default, meaning "use :attr:`default`" —
    set a stage explicitly to override it independently. ``default`` itself
    is ``'numpy'`` so that nothing silently switches to jax/torch just
    because a GPU happens to be present; set ``default = 'auto'`` to let
    every stage below auto-select the fastest backend available instead
    (jax GPU/TPU > torch GPU > numba CPU JIT [``mass_assignment`` only] >
    numpy).
    """

    default: str = 'numpy'
    """Fallback backend used by any stage left at ``None``. ``'numpy'``
    (default) is always safe/deterministic. ``'auto'`` auto-selects the
    fastest backend available on this machine for every stage below that
    hasn't been set explicitly."""

    profile_painting: str | None = None
    """Backend for :mod:`beorn.painting.helpers`'s Fourier profile-kernel
    convolution — :class:`~beorn.painting.coordinator.PaintingCoordinator`'s
    ionization/heating/Lyman-alpha painting stage. ``None`` (default) uses
    :attr:`default`."""

    mass_assignment: str | None = None
    """Backend for :func:`beorn.particle_mapping.map_particles_to_mesh`,
    used by :meth:`beorn.lpt.LPTBase.get_density`,
    :meth:`beorn.structs.HaloCatalog.to_mesh` (always painted with the
    ``'NGP'`` scheme there — this only selects which library executes it,
    never the scheme), and the Thesan N-body loader's own density/RSD
    painting. ``None`` (default) uses :attr:`default`. Not every backend
    supports every mass-assignment scheme — see
    :func:`~beorn.particle_mapping.map_particles_to_mesh`'s support matrix."""

    lpt: str | None = None
    """Backend for :class:`~beorn.lpt.LPTBase`'s internal FFTs (δ(k)
    realisation, displacement, velocity, linear density). Its public methods
    always convert back to numpy (``LPTBackend.to_numpy``) regardless of
    this choice — a speed knob, not a differentiability one. ``None``
    (default) uses :attr:`default`."""

    hmf: str | None = None
    """Backend for :class:`~beorn.mass_function.HaloMassFunction`. Unlike
    the stages above, this genuinely changes the returned array type —
    ``'jax'``/``'torch'`` give differentiable, device-resident output with
    no conversion back to numpy. ``None`` (default) uses :attr:`default`."""

    def resolve(self, stage: str) -> str:
        """Return the explicit backend for ``stage`` if set, else :attr:`default`.

        Args:
            stage: One of ``'profile_painting'``, ``'mass_assignment'``,
                ``'lpt'``, ``'hmf'``.
        """
        value = getattr(self, stage)
        return value if value is not None else self.default


@dataclass(slots = True)
class SimulationParameters:
    """
    Parameters that are used to run the simulation. These are used in the generation of the halo profiles and when converting the halo profiles to a grid.
    """

    Ncell: int = 128
    """Number of pixels of the final grid. This is the number of pixels in each dimension. The total number of pixels will be Ncell^3."""

    Lbox: float = 100
    """Box length, in [Mpc/h] (default; see ``use_hunits``) or physical Mpc when
    ``use_hunits=False``. This is the length of the box in each dimension. The
    total volume will be Lbox^3. Internal code never reads this field directly —
    it always goes through :attr:`Parameters.Lbox_hunits`, which resolves it to
    Mpc/h regardless of ``use_hunits``."""

    store_grids: list = ('delta_b', 'Grid_Temp', 'Grid_xHII', 'Grid_xal')
    """Base grids to write to the HDF5 output file. These four fields are the independent outputs of the painting stage.
    Derived quantities such as 'Grid_dTb' are *not* stored by default because they can be recomputed on the fly
    as cached properties from the base fields (``Grid_dTb = f(delta_b, Grid_Temp, Grid_xHII, Grid_xal, z)``).
    Add 'Grid_dTb' here only if you need pre-computed access to it for very large grids where recomputation is expensive."""

    cores: int = 1
    """Number of cores used in parallelization. The computation for each redshift can be parallelized with a shared memory approach. This is the number of cores used for this. Keeping the number at 1 disables parallelization."""

    backend: BackendParameters = field(default_factory = BackendParameters)
    """Array-library backend selection for painting, mass assignment, the
    LPT solver, and the halo mass function — see :class:`BackendParameters`.
    This is a performance knob for the production pipeline only — it does
    not make :class:`~beorn.painting.coordinator.PaintingCoordinator` (or
    :meth:`~beorn.lpt.LPTBase.get_density`/:meth:`~beorn.structs.HaloCatalog.to_mesh`)
    differentiable in the sense of the ``beorn.*.differentiable`` module
    family; jax/torch's scatter-add mass assignment is also not
    bit-deterministic run-to-run on GPU. Pass ``backend='numpy'`` directly to
    a specific call (e.g. ``get_density(..., backend='numpy')``) to pin just
    that call when you need reproducibility, independent of this default."""

    spreading_pixel_threshold: int = -1
    """When spreading the excess ionization fraction, treat all the connected regions with less than "thresh_pixel" as a single connected region (to speed up). If set to a negative value, a default nonzero value will be used"""

    spreading_subgrid_approximation: bool = True
    """When spreading the excess ionization fraction and running distance_transform_edt, whether or not to do the subgrid approximation."""

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

    oversample: int = 1
    """Resolution-enhancement factor for any internal grid that gets reduced
    back down to ``Ncell`` before being used further. Opposite direction of
    :attr:`degrade_resolution`. A value of 1 (default) applies no refinement.
    Two independent consumers:

    - :meth:`beorn.lpt.LPTBase.get_density` (default, unless its own
      ``oversample`` argument is given explicitly): paints the LPT-generated
      field onto an internal ``oversample*Ncell`` mesh, then block-averages
      back down to ``(Ncell, Ncell, Ncell)`` before returning. Reduces
      mass-assignment discreteness/aliasing for real-space statistics where
      :func:`beorn.particle_mapping.deconvolve_mas` doesn't apply (issue #48).
    - :class:`~beorn.load_input_data.Py21cmFastLoader` (default, unless its
      own ``oversample`` argument is given explicitly): sets py21cmfast's
      internal grid ``DIM = Ncell * oversample`` while ``HII_DIM = Ncell``
      stays the output resolution. A larger factor resolves lower halo
      masses at the cost of more memory/compute; the minimum resolvable halo
      mass scales roughly as ``(Lbox / DIM)^3``.

    In both cases ``Ncell`` is always the resulting (coarse) grid size; the
    finer internal grid is never itself persisted."""

    use_hunits: bool = False
    """Whether ``Lbox`` is given in h-units (Mpc/h — the historical BEoRN
    convention) or physical Mpc (the default, ``False``). Internal code
    resolves ``Lbox`` via :attr:`Parameters.Lbox_hunits`, which always returns
    Mpc/h regardless of this flag — see that property's docstring for the
    full design (issue #49). Note this default means an unmodified ``Lbox``
    value now means physical Mpc, not Mpc/h as in pre-#49 scripts — a
    deliberate, non-backward-compatible choice; existing scripts/notebooks
    that set ``Lbox`` without setting ``use_hunits`` need updating. Halo-mass-
    valued quantities are not yet affected by this flag (deferred, tracked
    separately)."""

    mass_assignment: Literal['NGP', 'CIC', 'TSC', 'PCS'] = 'CIC'
    """Mass-assignment scheme used to paint particles/haloes onto the grid.
    Read as the default by :meth:`beorn.lpt.LPTBase.get_density` (the matter
    density field) and by :meth:`beorn.structs.HaloCatalog.to_mesh` (halo
    positions, as profile centers for the ionization/heating/Lyman-alpha
    painting stage) whenever their own ``mass_assignment`` argument isn't
    given explicitly. Does not affect
    ``cosmo_sim.halo_catalogs_thesan_mass_assignment``, which is a separate
    knob for the Thesan N-body loader's own particle painting (issue #48).

    Real-space mass-assignment window deconvolution (correcting the
    ``sinc^p`` suppression near k_Nyquist) is a per-call ``deconvolve``
    argument on :meth:`~beorn.lpt.LPTBase.get_density`,
    :meth:`~beorn.structs.HaloCatalog.to_mesh`,
    :func:`beorn.particle_mapping.map_particles_to_mesh`, and
    :meth:`~beorn.structs.TemporalCube.power_spectrum` — each defaulting to
    ``False`` — rather than a simulation-wide default here. Deconvolving the
    *real-space field itself* divides out the window in Fourier space, which
    amplifies noise near k_Nyquist enough to push cells below the physical
    ``δ = -1`` floor (worse at lower redshift, where small-scale power is
    larger); that fed straight into ``T_adiab_fluctu``'s ``(1 + δ)**(2/3)``,
    producing ``NaN`` cells that silently poisoned
    ``TemporalCube.global_mean``'s per-redshift average (not NaN-aware) for
    every snapshot at or below the first affected redshift. For P(k)
    analysis, prefer passing ``deconvolve=True`` directly to
    :meth:`beorn.structs.TemporalCube.power_spectrum` /
    :func:`beorn.power_spectrum.power_spectrum_1d`, which deconvolve only for
    that one measurement without ever writing the noisier field back
    (issue #48)."""

    @staticmethod
    def _kbins_from(lbox: float, ncell: int) -> np.ndarray:
        k_min = 1 / lbox
        k_max = ncell / lbox
        # TODO - explain the factor of 6
        bin_count = int(6 * np.log10(k_max / k_min))

        return np.logspace(np.log10(k_min), np.log10(k_max), bin_count, base=10)

    @property
    def kbins(self) -> np.ndarray:
        """
        Returns the k bins for the power spectrum. The bins are logarithmically spaced between k_min and k_max.
        The number of bins is determined by the size of the simulation box and the number of cells.
        """
        return self._kbins_from(self.Lbox, self.Ncell)

    def __post_init__(self):
        # ensure the items of the store_grids are strings. When loading from hdf5 they might be bytes
        self.store_grids = [s.decode() if isinstance(s, bytes) else s for s in self.store_grids]
        # Parameters.from_dict only re-instantiates its own direct dataclass
        # fields (SimulationParameters(**value)); a nested dataclass field
        # like `backend` arrives as a plain dict when loaded from YAML/HDF5.
        if isinstance(self.backend, dict):
            self.backend = BackendParameters(**self.backend)



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

    Om: float = 0.315
    Ob: float = 0.045
    Ol: float = 1 - 0.315
    rho_c: float = 2.775e11
    h0: float = 0.673
    sigma_8: float = 0.83
    ns: float = 0.96

    w0: float = -1.0
    """Dark-energy equation-of-state parameter at a=1, CPL parameterization
    w(a) = w0 + wa*(1-a). Default -1.0 (cosmological constant). Read by
    :func:`beorn.cosmo.background.E`/:func:`hubble`/:func:`hubble_per_yr`, so it
    affects the whole background expansion history (and therefore every
    quantity derived from it: growth factor, mass function, LPT, dTb)."""

    wa: float = 0.0
    """Dark-energy equation-of-state evolution parameter, CPL parameterization
    w(a) = w0 + wa*(1-a). Default 0.0 (constant w=w0). See :attr:`w0`."""

    growth_factor_method: Literal['integral', 'cpt92', 'linder2005', 'linder_cahn2007'] = 'integral'
    """Method used by :func:`beorn.cosmo.background.D` to compute the linear
    growth factor D(a), normalised to D(a=1)=1:

    - ``'integral'`` (default): exact numerical integral of the growth ODE
      (:func:`~beorn.cosmo.background.D_non_normalized`) for the CPL
      background set by :attr:`w0`/:attr:`wa`.
    - ``'cpt92'``: Carroll, Press & Turner (1992) analytic fitting formula —
      a widely used fast approximation to the exact integral, but not
      validated away from w=-1.
    - ``'linder2005'``: Linder (2005, PhRvD, 72, 043529) growth-index
      approximation (d ln D/d ln a = Omega_m(a)^gamma, single gamma
      evaluated at w(z=1)). Reduces to the classic fixed gamma=0.55
      (Omega_m(z)^0.55) approximation under the default w0=-1, wa=0.
    - ``'linder_cahn2007'``: Linder & Cahn (2007, Astropart.Phys. 28, 481)
      variant with gamma(a) tracking w(a) at every point of the integral —
      only differs from ``'linder2005'`` when wa != 0."""


@dataclass(slots = True)
class CosmoSimParameters:
    """Parameters specific to N-body/cosmo-sim inputs (py21cmfast, Thesan, PKDGrav, etc.)."""

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


    @property
    def Lbox_hunits(self) -> float:
        """``simulation.Lbox`` resolved to the internal Mpc/h representation.

        ``simulation.Lbox`` is user input whose *meaning* depends on
        ``simulation.use_hunits``: Mpc/h when True, physical Mpc when False
        (the default). Every internal consumer of the box size (LPT, mass
        function, CHMF, painting, particle mapping, tools21cm calls, ...)
        must read this property instead of ``simulation.Lbox`` directly, so
        the toggle is resolved in exactly one place rather than at each call
        site. See issue #49.
        """
        if self.simulation.use_hunits:
            return self.simulation.Lbox
        return self.simulation.Lbox * self.cosmology.h0

    def unique_hash(self) -> str:
        """
        Generates a unique hash for the current set of parameters. This can be used as a unique key when caching the computations.
        """
        dict_params = to_dict(self)
        # using the string representation of the dictionary is not optimal because it is not guaranteed to be the same for the same dictionary (if the order of the keys is different for instance)
        # but the key is that the hashes are guaranteed to be different for unique parameter sets
        dict_string = f"{dict_params}"

        return hashlib.md5(dict_string.encode()).hexdigest()

    def profiles_hash(self) -> str:
        """Short MD5 hash of parameters that affect the 1D radiation profiles.

        Covers source parameters, cosmology, solver redshifts, and the halo
        mass / accretion-rate bins.  Intentionally excludes random seed, grid
        dimensions (Ncell, Lbox, oversample), and other
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
            f"  Grid        : Ncell={sim.Ncell}, Lbox={sim.Lbox} {'Mpc/h' if sim.use_hunits else 'Mpc'} "
            f"(={self.Lbox_hunits:.3f} Mpc/h internally)",
            f"  Profile z   : z={slv.redshifts[0]:.1f} -> {slv.redshifts[-1]:.1f} ({slv.redshifts.size} steps)",
            *(
                [f"  Snapshot z  : z={cosmo_sim.snapshot_redshifts[0]:.1f} -> {cosmo_sim.snapshot_redshifts[-1]:.1f} ({cosmo_sim.snapshot_redshifts.size} snapshots)"]
                if cosmo_sim.snapshot_redshifts is not None else []
            ),
            f"  1D RT bins  : {slv.halo_mass_bin_min:.1e} - {slv.halo_mass_bin_max:.1e} Msun at z={z_min:.1f} ({slv.halo_mass_nbin} bins, traced back via exp. accretion)",
            f"  Source      : f_st={src.f_st}, Nion={src.Nion}, f0_esc={src.f0_esc}",
            f"  X-ray       : norm={src.xray_normalisation:.2e}, E=[{src.energy_cutoff_min_xray}, {src.energy_cutoff_max_xray}] eV",
            f"  Lyman-alpha : n_phot={src.n_lyman_alpha_photons}, star-forming above {src.halo_mass_min:.1e} Msun",
            f"  Beorn hash  : {self.beorn_hash()}",
            "=" * 60,
        ]
        return "\n".join(lines)

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
        return cls.from_dict(_dataclass_dict_from_group(cls, group))


def _dataclass_dict_from_group(dc_type: type, group: h5py.Group) -> dict:
    """Reconstruct a dataclass-shaped dict from an HDF5 group written by
    :func:`to_dict`/:meth:`BaseStruct._to_h5_field`.

    Recurses into any field whose declared type is itself a dataclass
    (e.g. ``SimulationParameters.backend: BackendParameters``) — a plain
    one-level loop over ``fields(dc_type)`` would try to read a nested
    dataclass's HDF5 *group* as if it were a dataset (``group[name][...]``),
    which raises ``TypeError`` (h5py groups don't support that indexing).
    """
    out = {}
    for f in fields(dc_type):
        name = f.name
        if name in group.attrs:
            out[name] = group.attrs[name]
        elif name in group:
            member = group[name]
            if isinstance(member, h5py.Group) and is_dataclass(f.type):
                out[name] = _dataclass_dict_from_group(f.type, member)
            else:
                out[name] = member[...] if hasattr(member, 'shape') else member[:]
        else:
            # some configurations result in empty fields (e.g. file_root
            # might not be set when using a mock simulation)
            logger.debug(f"Did not find field {name} in group {dc_type.__name__}.")
    return out


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
            # ensure the types are writable to hdf5
            if isinstance(value[0], Path):
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
