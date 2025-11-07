"""
Global parameters for this simulation. They encompass the astrophysical parameters of the source, the cosmological parameters, the simulation parameters, the solver parameters, the excursion set parameters, and the halo mass function parameters.
Slots are used to prevent the creation of new attributes. This is useful to avoid typos and to have a clear overview of the parameters.
"""

from pathlib import Path
import importlib
import hashlib
from dataclasses import dataclass, field, is_dataclass, fields
from typing import Literal
import numpy as np
import inspect
import yaml
import h5py

from .helpers import bin_centers


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
    # TODO - rename to the redshift range
    redshifts: np.ndarray = field(default_factory=lambda: np.arange(25, 6, -0.5))
    """Array of redshifts for the simulation. This should exactly match the redshifts of the halo catalogs. This should also be monotonically decreasing."""

    fXh: Literal['constant', 'variable'] = 'constant'
    """if fXh is constant here, it will take the value 0.11. Otherwise, we will compute the free e- fraction in neutral medium and take the fit fXh = xe**0.225"""



@dataclass(slots = True)
class SimulationParameters:
    """
    Parameters that are used to run the simulation. These are used in the generation of the halo profiles and when converting the halo profiles to a grid.
    """
    halo_mass_accretion_alpha: np.ndarray = field(default_factory = lambda: np.linspace(0.1, 0.9, 10))
    """Coefficient for exponential mass accretion. Since beorn distinguishes between accretion rates a range should be specified"""

    halo_mass_bin_min: float = 1e5
    """Minimum halo mass bin in solar masses."""

    halo_mass_bin_max: float = 1e14
    """Maximum halo mass bin in solar masses."""

    halo_mass_bin_n: int = 100
    """Number of mass bins."""

    Ncell: int = 128
    """Number of pixels of the final grid. This is the number of pixels in each dimension. The total number of pixels will be Ncell^3."""

    Lbox: float = 100
    """Box length, in [Mpc/h]. This is the length of the box in each dimension. The total volume will be Lbox^3."""

    store_grids: list = ('Tk', 'bubbles', 'lyal', 'dTb')
    """List of the grids to store. Simulating only the needed grids will speed up the simulation. The available grids are: Tk, bubbles, lyal, dTb."""

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

    random_seed: int = 12345
    """Random seed for the random number generator. This is used to generate the random numbers for the halo catalogs and the density fields when using 21cmfast."""

    compute_s_alpha_fluctuations: bool = True
    """Whether or not to include the fluctuations in the suppression factor S_alpha when computing the x_al fraction."""

    compute_x_coll_fluctuations: bool = True
    """Whether or not to include the fluctuations in the collisional coupling coefficient x_coll when computing the x_tot fraction."""

    # derived properties that are directly related to the parameters
    @property
    def halo_mass_bins(self) -> np.ndarray:
        return np.logspace(np.log10(self.halo_mass_bin_min), np.log10(self.halo_mass_bin_max), self.halo_mass_bin_n, base=10)

    @property
    def halo_mass_bin_centers(self) -> np.ndarray:
        return bin_centers(self.halo_mass_bins)

    @property
    def halo_mass_accreation_alpha_bin_centers(self) -> np.ndarray:
        return bin_centers(self.halo_mass_accretion_alpha)

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


    # halo_catalogs_thesan_tree: Path = None
    # halo_catalogs_thesan_offsets: Path = None
    # TODO rename since this is actually the mass assignment used for the background density field. For assigning the halos to the grid we always use NGP
    halo_catalogs_thesan_mass_assignment: Literal['NGP', 'CIC'] = 'CIC'
    """Method used to assign the halo mass to the grid. Can be either NGP (Nearest Grid Point) or CIC (Cloud In Cell)."""

    file_root: Path = None

    def __post_init__(self):
        # ensure the the np.ndarray fields are numpy arrays
        if isinstance(self.halo_mass_accretion_alpha, list):
            self.halo_mass_accretion_alpha = np.array(self.halo_mass_accretion_alpha)

        if isinstance(self.file_root, str):
            self.file_root = Path(self.file_root)

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
        ps: Path to the input Linear Power Spectrum.
        correlation_function: Path where the corresponding correlation function will be stored.
        HI_frac: HI number fraction. Only used when running H_He_Final.
        clumping: Rescale the background density. Set to 1 to get the normal 2h profile term.
        z_decoupling: Redshift at which the gas decouples from CMB and starts cooling adiabatically.
    """

    # TODO - remove and set an astropy cosmology object instead of individual parameters
    Om: float = 0.31
    Ob: float = 0.045
    Ol: float = 0.68
    rho_c: float = 2.775e11
    h: float = 0.68
    sigma_8: float = 0.83
    ns: float = 0.96
    ps: Path = Path(importlib.util.find_spec('beorn').origin).parent / 'files' / 'PCDM_Planck.dat'
    correlation_function: Path = Path(importlib.util.find_spec('beorn').origin).parent / 'files' / 'corr_fct.dat'
    HI_frac: float = 1 - 0.08
    clumping: int = 1
    z_decoupling: int = 135




@dataclass(slots = True)
class Parameters:
    """
    Group all the parameters for the simulation.

    Attributes:
        source: SourceParameters
        solver: SolverParameters
        cosmology: CosmologyParameters
        simulation: SimulationParameters
        excursion_set: ExcursionSetParameters
        halo_mass_function: HaloMassFunctionParameters
    """
    source: SourceParameters = field(default_factory = SourceParameters)
    solver: SolverParameters = field(default_factory = SolverParameters)
    cosmology: CosmologyParameters = field(default_factory = CosmologyParameters)
    simulation: SimulationParameters = field(default_factory = SimulationParameters)


    def unique_hash(self) -> str:
        """
        Generates a unique hash for the current set of parameters. This can be used as a unique key when caching the computations.
        """
        dict_params = to_dict(self)
        # using the string representation of the dictionary is not optimal because it is not guaranteed to be the same for the same dictionary (if the order of the keys is different for instance)
        # but the key is that the hashes are guaranteed to be different for unique parameter sets
        dict_string = f"{dict_params}"

        return hashlib.md5(dict_string.encode()).hexdigest()


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
        for field in fields(cls):
            field_name = field.name
            # check if the nested field would be a dataclass as well
            if is_dataclass(field.type):
                # iterate over the fields of the dataclass
                sub_group = group[field_name]
                sub_params_dict = {}
                for sub_field in fields(field.type):
                    sub_field_name = sub_field.name
                    if sub_field_name in sub_group.attrs:
                        sub_params_dict[sub_field_name] = sub_group.attrs[sub_field_name]
                    elif sub_field_name in sub_group:
                        # this is a dataset
                        sub_params_dict[sub_field_name] = sub_group[sub_field_name][...]
                    else:
                        print(f"Did not find field {sub_field_name} in group {field_name}.")

                params_dict[field_name] = sub_params_dict

            else:
                print("no dataclass", field_name)
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
