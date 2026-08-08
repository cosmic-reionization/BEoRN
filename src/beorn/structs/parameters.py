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

    spreading_method: Literal['exact', 'diffusion'] = 'exact'
    """Excess-ionization spreading algorithm. ``'exact'`` (default) uses
    :func:`beorn.painting.spread.spreading_excess_fast` (connected-component
    + distance-transform, exact but CPU/numpy-only — reads
    ``spreading_pixel_threshold``/``spreading_subgrid_approximation`` above).
    ``'diffusion'`` uses :func:`beorn.painting.differentiable.spreading_excess_diff`
    (FFT-diffusion surrogate, backend-generic and differentiable, approximate
    — see its docstring for the documented photon-loss tradeoff). Named for
    the specific algorithm, not just "differentiable" — other differentiable
    methods may be added as additional options later."""

    spreading_diffusion_n_iter: int = 8
    """Iteration count for :func:`~beorn.painting.differentiable.spreading_excess_diff`
    when ``spreading_method='diffusion'``."""

    spreading_diffusion_R_diffuse: float | None = None
    """Per-iteration Gaussian diffusion scale (Mpc/h) for
    :func:`~beorn.painting.differentiable.spreading_excess_diff` when
    ``spreading_method='diffusion'``. ``None`` uses its own default (2 cells)."""

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
        w0: Dark-energy equation-of-state parameter at a=1 (CPL).
        wa: Dark-energy equation-of-state evolution parameter (CPL).
        growth_factor_method: Method used to compute the linear growth factor D(a).
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
    """Parameters for however the cosmological density field is produced —
    native LPT (1LPT/2LPT/3LPT) counts as a "cosmo sim" here alongside
    external N-body inputs (py21cmfast, Thesan, PKDGrav, etc.)."""

    density_source: Literal['1LPT', '2LPT', '3LPT', 'external'] = '2LPT'
    """Which mechanism produces the density field. ``'1LPT'``/``'2LPT'``/
    ``'3LPT'`` mean native LPT at that order (:class:`~beorn.lpt.ZeldovichApproximation`/
    :class:`~beorn.lpt.SecondOrderLPT`/:class:`~beorn.lpt.ThirdOrderLPT`);
    ``'external'`` means an external N-body loader (py21cmfast/Thesan/PKDGrav)
    is used instead. This is a metadata field recording the choice for
    hashing/reproducibility — it does not itself dispatch which class gets
    instantiated (you still construct the LPT solver or loader class
    directly); keep it in sync with whichever you actually use."""

    mass_assignment: Literal['NGP', 'CIC', 'TSC', 'PCS'] = 'CIC'
    """Mass-assignment scheme used to paint the density field. Read as the
    default by :meth:`beorn.lpt.LPTBase.get_density` whenever its own
    ``mass_assignment`` argument isn't given explicitly. Does not affect halo
    catalog painting (see :attr:`HaloSimParameters.mass_assignment`) or any
    N-body loader's own particle painting, which reads this same field
    directly (e.g. the Thesan loader).

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

    upsample_density_fourier: int = 1
    """Cheap, band-limited resolution-enhancement factor for
    :meth:`beorn.lpt.LPTBase.get_density` (default, unless its own
    ``upsample_density_fourier`` argument is given explicitly): paints the
    already-solved LPT displacement field onto an internal
    ``upsample_density_fourier*Ncell`` mesh via
    :func:`~beorn.particle_mapping.upsample_field_fourier`, then
    block-averages back down to ``(Ncell, Ncell, Ncell)`` before returning.
    Reduces mass-assignment discreteness/aliasing for real-space statistics
    where :func:`beorn.particle_mapping.deconvolve_mas` doesn't apply (issue
    #48) -- but adds **no new small-scale power**: the source field is
    already band-limited at ``Ncell``'s own Nyquist, so this only resamples
    it more finely, it does not extrapolate new k-modes. A value of 1
    (default) applies no refinement. ``Ncell`` is always the resulting
    (coarse) grid size; the finer internal painting mesh is never itself
    persisted. See :attr:`field_oversample` for the alternative that *does*
    add real new modes (issue #56)."""

    field_oversample: int = 1
    """Genuine resolution-enhancement factor: solves for **real new k-modes**
    beyond ``Ncell``'s own Nyquist frequency, unlike
    :attr:`upsample_density_fourier`'s cheap resample-only refinement.
    Consumed by :class:`~beorn.load_input_data.Py21cmFastLoader` (default,
    unless its own ``field_oversample`` argument is given explicitly): sets
    py21cmfast's internal grid ``DIM = Ncell * field_oversample`` while
    ``HII_DIM = Ncell`` stays the output resolution. A larger factor resolves
    lower halo masses at the cost of more memory/compute; the minimum
    resolvable halo mass scales roughly as ``(Lbox / DIM)^3``. A value of 1
    (default) applies no refinement. ``Ncell`` is always the resulting
    (coarse) grid size; the finer internal grid is never itself persisted.

    Also read by :attr:`~beorn.structs.HaloSimParameters.field_oversample`
    (``None`` there inherits this value) to generate the CHMF's own
    conditioning field at a finer, phase-synchronized resolution before
    top-hat-smoothing it down to ``Ncell`` -- see that field's docstring for
    why the two must share the same underlying Fourier phases rather than
    each independently resolving at their own requested factor."""

    IC_seed: int = 12345
    """Seed for the density field's initial conditions: py21cmfast's own
    IC/perturb-field seed, or the seed :class:`~beorn.lpt.LPTBase` reads by
    default (its own ``seed`` constructor argument, when not given
    explicitly). Independent of :attr:`HaloSimParameters.halo_sampler_seed`
    (the halo-catalog Poisson/position-sampling seed) and
    :attr:`HaloSimParameters.IC_seed` (the density-field seed used
    specifically by :class:`~beorn.load_input_data.LPTHaloLoader`'s own
    internal LPT solver — see that field's docstring for how the two
    interact)."""

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
class HaloSimParameters:
    """Parameters for how halo catalogs are generated."""

    halo_source: Literal['CHMF', 'external'] = 'CHMF'
    """How halo catalogs are generated. ``'CHMF'`` — conditional halo mass
    function sampling on the density field (see :attr:`hmf_model` for the
    PS/ST calibration). ``'external'`` — read from an external N-body
    loader's own halo finder (py21cmfast, Thesan, PKDGrav). Kept as a
    separate axis from :attr:`hmf_model` (rather than folding PS/ST into this
    field, e.g. ``'CHMF_PS'``) so future native halo-finding methods (e.g.
    excursion-set/peak-patch, issue #26) can be added as new ``halo_source``
    values without overloading ``hmf_model``. Metadata field for
    hashing/reproducibility — does not itself dispatch which loader/sampler
    gets constructed."""

    hmf_model: Literal['PS', 'ST'] = 'ST'
    """Only meaningful when :attr:`halo_source` is ``'CHMF'``. ``'PS'`` —
    pure EPS conditional sampling (volume average = Press-Schechter).
    ``'ST'`` (default) — conditional sampling calibrated so the volume
    average matches Sheth-Tormen instead (as in 21cmFAST-family codes); see
    :attr:`chmf_recipe` for *how* that calibration is done. Read as the
    default by :class:`~beorn.lpt.chmf.CHMFSampler`/
    :class:`~beorn.load_input_data.LPTHaloLoader` whenever their own
    ``hmf_model`` argument isn't given explicitly."""

    chmf_recipe: Literal['BarkanaLoeb2004', 'MovingBarrier'] = 'BarkanaLoeb2004'
    """Only meaningful when :attr:`hmf_model` is ``'ST'`` (silently ignored
    for ``'PS'``, which has no ST-calibration route to choose between).
    Case-insensitive. ``'BarkanaLoeb2004'`` (default) --
    :meth:`~beorn.lpt.chmf.CHMF.hmf_chmf_field` (pure PS-conditional shape)
    rescaled by the unconditional ST/PS ratio (Barkana & Loeb 2004, ApJ 609,
    474, "Unusually Large Fluctuations in the Statistics of Galaxy Formation
    at High Redshift" -- not to be confused with Barkana & Loeb 2005, ApJ
    624, L65, a different paper). ``'MovingBarrier'`` --
    :meth:`~beorn.lpt.chmf.CHMF.hmf_st_movingbarrier`, the direct
    ellipsoidal-collapse moving-barrier conditional solution (Sheth & Tormen
    2002), as implemented by Davies, Mesinger & Murray (2025, 21cmFASTv4) --
    already ST-calibrated on its own, so its conditional *shape* need not
    match the Barkana & Loeb hybrid's (which inherits its delta-dependence
    entirely from the pure-PS conditional formula). Read as the default by
    :class:`~beorn.lpt.chmf.CHMFSampler` whenever its own ``chmf_recipe``
    argument isn't given explicitly."""

    delta_c: float = 1.686
    """Linear collapse threshold used by :class:`~beorn.lpt.chmf.CHMF`/
    :class:`~beorn.lpt.chmf.CHMFSampler`."""

    R_env: float | None = None
    """Environmental smoothing scale in Mpc/h for CHMF conditioning. ``None``
    (default) uses the cell size as the conditioning scale."""

    n_mass_bins: int = 40
    """Number of log-spaced mass bins for CHMF sampling."""

    halo_mass_min: float = 1e8
    """Lower bound of the CHMF sampling mass range, in M_sun. Independent of
    :attr:`SourceParameters.halo_mass_min` (the star-forming/painting cutoff
    applied downstream) — these are different concerns: this controls what's
    *generated*, source's controls what's *painted*. Same default value for
    a sane out-of-the-box match, but change them independently as needed."""

    halo_mass_max: float = 1e16
    """Soft upper bound of the CHMF sampling mass range, in M_sun — not a
    strong/binding constraint in practice. The real hard ceiling is set by
    the box itself: per-cell EPS conditioning caps sampleable mass at the
    per-cell environment mass M_env (set by :attr:`R_env` and cell size), and
    even without per-cell conditioning the box volume bounds the largest
    halo that can exist. This field only lets you cap the range *further*
    below whichever of those applies; it can never raise the effective bound
    above them. Default 1e16 is effectively a no-op cap for realistic grids."""

    halo_sampler_seed: int = 42
    """RNG seed for halo-catalog generation (Poisson draws + intra-cell
    position sampling) — independent of :attr:`CosmoSimParameters.IC_seed`
    (the density-field generation seed) and :attr:`IC_seed` (below), so none
    of the three ever conflict. Matches
    :class:`~beorn.load_input_data.LPTHaloLoader`'s existing hardcoded
    default of 42 (no default-value change)."""

    IC_seed: int | None = None
    """Seed for :class:`~beorn.load_input_data.LPTHaloLoader`'s own internal
    LPT solver, which generates the density field that
    :class:`~beorn.lpt.chmf.CHMFSampler` conditions halo sampling on. ``None``
    (default) inherits :attr:`CosmoSimParameters.IC_seed`, so the halo
    catalog's underlying density realization matches whichever density field
    is used elsewhere in the pipeline by default. Setting this to a value
    different from :attr:`CosmoSimParameters.IC_seed` deliberately
    decorrelates the two — :class:`~beorn.load_input_data.LPTHaloLoader`
    warns (``UserWarning``) when it detects this, since it means the sampled
    halos will not be spatially correlated with a density field built
    elsewhere from :attr:`CosmoSimParameters.IC_seed`."""

    field_oversample: int | None = None
    """Resolution factor for the CHMF's own conditioning field, generated at
    ``Ncell * field_oversample`` and top-hat-smoothed to :attr:`R_env` at
    that finer resolution *before* being decimated (point-sampled at the
    coincident coarse grid points -- **not** block-averaged, which would
    apply an unwanted second smoothing on top of :attr:`R_env`'s own and
    make the bias worse, not better) back down to ``Ncell`` -- reduces the
    per-cell conditioning field's residual variance
    bias (raw linear-field variance measured ~21% off the analytic
    ``sigma^2(M_env, z)``, ~6% even after issue #54's presmoothing fix,
    because ``Ncell``'s own Nyquist frequency isn't high enough to resolve
    the :attr:`R_env` top-hat window well). ``None`` (default) inherits
    :attr:`CosmoSimParameters.field_oversample` (itself ``1`` -- no
    refinement -- by default).

    Read by :class:`~beorn.load_input_data.LPTHaloLoader` whenever it builds
    its own internal density field (i.e. no explicit ``lpt_solver`` is
    passed in). The finer field is **not** independently resolved at its own
    seed -- doing so would decorrelate the sampled halos from the coarse,
    ``Ncell``-resolution density field used elsewhere in the pipeline (e.g.
    by :meth:`~beorn.load_input_data.LPTHaloLoader.load_density_field`),
    reintroducing the same kind of seed-mismatch bug already fixed for
    :attr:`IC_seed`. Instead, both resolutions are derived from a single
    Fourier-space noise realisation drawn once at the finer grid: the coarse
    field is the exact low-k truncation of that realisation (not an
    independent draw and not a real-space block-average of it), so every
    resolution any consumer requests is a phase-consistent view of the same
    underlying box -- see :func:`beorn.lpt.lpt.synchronized_white_noise`/
    :func:`beorn.lpt.lpt.extract_synced_delta_k`. Setting this to a value
    different from :attr:`CosmoSimParameters.field_oversample` is fine (only
    the CHMF's own conditioning field is affected) and does not trigger any
    warning, unlike the seed-mismatch case above -- there is no
    decorrelation risk here since both are views of the same realisation."""

    mass_assignment: Literal['NGP', 'CIC', 'TSC', 'PCS'] = 'NGP'
    """Mass-assignment scheme for painting halo *positions* onto the grid
    (:meth:`beorn.structs.HaloCatalog.to_mesh`) as profile centers for the
    ionization/heating/Lyman-alpha convolution stage. Leave at ``'NGP'`` —
    halo catalogs are discrete point sources, and the downstream
    profile-kernel convolution already does the physical smoothing; painting
    with CIC/TSC/PCS here adds an extra, unphysical pre-smoothing on top of
    that. Changing this is almost certainly not what you want."""


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
    """cosmo-sim input parameters (density-field source: native LPT order or
    external N-body loader, plus mass assignment/field_oversample/
    upsample_density_fourier/seed for it)"""
    halo_sim: HaloSimParameters = field(default_factory = HaloSimParameters)
    """halo-catalog generation parameters (CHMF vs external, HMF calibration,
    mass range, seed, mass assignment for painting halo positions)"""


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
        dimensions (Ncell, Lbox, field_oversample/upsample_density_fourier), and other
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
        in the input_tag. Exception: ``cosmo_sim.mass_assignment``/
        ``field_oversample``/``upsample_density_fourier`` *do* affect the
        painted density field, so they're added back explicitly below even
        though the rest of ``cosmo_sim`` stays excluded.
        """
        d = {
            'source': to_dict(self.source),
            'solver': to_dict(self.solver),
            'simulation': to_dict(self.simulation),
            'cosmo_sim_mass_assignment': self.cosmo_sim.mass_assignment,
            'cosmo_sim_field_oversample': self.cosmo_sim.field_oversample,
            'cosmo_sim_upsample_density_fourier': self.cosmo_sim.upsample_density_fourier,
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
