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

    mass_accretion_lookback: int = 5
    """number of snapshots to consider when computing the mass accretion rate of the halos. The halo mass grow will affect the star formation rate and thus the radiative properties of the halos."""



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
        h: Dimensionless Hubble parameter.
        sigma_8: Amplitude of the matter power spectrum on 8 Mpc/h scales.
        ns: Scalar spectral index.
    """

    # TODO - remove and set an astropy cosmology object instead of individual parameters
    Om: float = 0.31
    Ob: float = 0.045
    Ol: float = 0.68
    rho_c: float = 2.775e11
    h: float = 0.68
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
            f"  Cosmology   : Om={cos.Om}, Ob={cos.Ob}, h={cos.h}, sigma_8={cos.sigma_8}",
            f"  Grid        : Ncell={sim.Ncell}, Lbox={sim.Lbox} Mpc/h",
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
