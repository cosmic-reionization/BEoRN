"""
Global parameters for this simulation. They encompass the astrophysical parameters of the source, the cosmological parameters, the simulation parameters, the solver parameters, the excursion set parameters, and the halo mass function parameters.
"""

from pathlib import Path
import importlib
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Union


@dataclass
class SourceParameters:
    """
    Parameters for the sources of radiation. Sensible defaults are provided.
    
    Attributes:
        mass_accretion_model: mass accretion model. Can be EXP or EPS.
        mass_accretion_alpha: coefficient for exponential mass accretion
        source_type: str. source type. SED, Ghara, Ross, constant
        energy_min_sed_xray: minimum energy of normalization of xrays in eV
        energy_max_sed_xray: minimum energy of normalization of xrays in eV
        energy_cutoff_min_xray: energy cutoff for the xray band.
        energy_cutoff_max_xray: energy cutoff for the xray band.
        alS_xray: PL sed Xray part N ~ nu**-alS [nbr of photons/s/Hz]
        xray_normalisation: Xray normalization [(erg/s) * (yr/Msun)] (astro-ph/0607234 eq22)
        n_lyman_alpha_phtons: nbr of lyal photons per baryons in stars
        lyman_alpha_power_law: power law index for lyal 
        halo_mass_min: Minimum mass of star forming halo. Mdark in HM
        halo_mass_max: Maximum mass of star forming halo
        f_st: 
        Mp: TODO
        g1: TODO
        g2: TODO
        Mt: TODO
        g3: TODO
        g4: TODO
        Nion: TODO
        f0_esc: photon escape fraction f_esc = f0_esc * (M/Mp)^pl_esc
        Mp_esc: TODO
        pl_esc: TODO
        min_xHII_value: set all pixels where xHII=0 to this value
    """

    mass_accretion_model: Literal['EXP', 'EPS'] = 'EXP'
    mass_accretion_alpha: float = 0.79
    source_type: Literal['SED', 'Ghara', 'Ross', 'constant'] = 'SED'
    energy_min_sed_xray: int = 500
    energy_max_sed_xray: int = 2000
    energy_cutoff_min_xray: int = 500
    energy_cutoff_max_xray: int = 2000
    alS_xray: float = 1.00001
    xray_normalisation: float = 3.4e40
    n_lyman_alpha_photons: int = 9690
    lyman_alpha_power_law: float = 0.0
    halo_mass_min: float = 1e8
    halo_mass_max: float = 1e16
    f_st: float = 0.05
    Mp: float = 2.8e11 * 0.68
    g1: float = 0.49
    g2: float = -0.61
    Mt: float = 1e8
    g3: float = 4
    g4: float = -1
    Nion: int = 5000
    f0_esc: float = 0.2
    Mp_esc: float = 1e10
    pl_esc: float = 0.0
    min_xHII_value: int = 0



@dataclass
class SolverParameters:
    """
    Solver parameters for the simulation.
    
    Attributes:
        z_max: Starting redshift
        z_min: Only for MAR. Redshift where to stop the solver
        Nz: Array or path to a text file. TODO enforce a type
        fXh: if fXh is constant here, it will take the value 0.11. Otherwise, we will compute the free e- fraction in neutral medium and take the fit fXh = xe**0.225
    """

    z_max: int = 40
    z_min: int = 6
    Nz: Union[np.ndarray, int, Path] = 500
    fXh: Literal['constant', 'variable'] = 'constant'



@dataclass
class SimulationParameters:
    """
    Attributes:
        halo_mass_bin_min: Minimum halo mass bin in solar masses.
        halo_mass_bin_max: Maximum halo mass bin in solar masses.
        halo_mass_bin_n: Number of bins to define the initial halo mass at z_ini = solver.z.
        average_profiles_in_bin: Inside one mass bin, halos are unevenly distributed. The profile corresponding to the mass bin should hence be a weighted average.
        HR_binning: Finer mass binning used to compute accurate average profile in coarse binning.
        model_name: Name of the simulation, used to name all the files created.
        Ncell: Number of pixels of the final grid.
        Lbox: Box length, in [Mpc/h].
        halo_catalogs: Path to the directory containing all the halo catalogs.
        store_grids: Whether or not to store the grids. If not, will just store the power spectra.
        dens_field: Path and name of the input density field on a grid. Used in run.py to compute dTb maps.
        dens_field_type: Can be either 21cmFAST or pkdgrav. It adapts the format and normalization of the density field.
        Nh_part_min: Halos with less than Nh_part_min are excluded.
        cores: Number of cores used in parallelization.
        kmin: Minimum k value.
        kmax: Maximum k value.
        kbin: Either a path to a text file containing kbin edges values or an int (number of bins to measure PS).
        thresh_pixel: When spreading the excess ionization fraction, treat all the connected regions with less than "thresh_pixel" as a single connected region (to speed up). If set to None, a default nonzero value will be used.
        subgrid_approximation: When spreading the excess ionization fraction and running distance_transform_edt, whether or not to do the subgrid approximation.
        nGrid_min_heat: Stacked_T_kernel.
        nGrid_min_lyal: Stacked_lyal_kernel.
        random_seed: Random seed when using 21cmFAST 2LPT solver.
        T_saturated: If True, we will assume Tk >> Tcmb.
        reio: If False, we will assume xHII = 0.
    """

    halo_mass_bin_min: float = 1e5
    halo_mass_bin_max: float = 1e14
    halo_mass_bin_n: int = 12
    average_profiles_in_bin: bool = True
    HR_binning: int = 400
    model_name: str = 'SED'
    Ncell: int = 128
    Lbox: int = 100
    halo_catalogs: Union[str, None] = None
    store_grids: list = ('Tk', 'bubbles', 'lyal', 'dTb')
    dens_field: Union[str, None] = None
    dens_field_type: Literal['21cmFAST', 'pkdgrav'] = 'pkdgrav'
    Nh_part_min: int = 50
    cores: int = 2
    kmin: float = 3e-2
    kmax: float = 4
    kbin: Union[int, str] = 30
    thresh_pixel: Union[int, None] = None
    subgrid_approximation: bool = True
    nGrid_min_heat: int = 4
    nGrid_min_lyal: int = 16
    random_seed: int = 12345
    T_saturated: bool = False
    reio: bool = True



@dataclass
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




@dataclass
class ExcursionSetParameters:
    """
    SemiNumerical Parameters
    
    Attributes:
        R_max: Mpc/h. The scale at which we start the excursion set.
        n_rec: Mean number of recombination per baryon.
        stepping: When doing the exc set, we smooth the field over varying scales. We loop and increase this scale logarithmically (R=1.1*R).
    """
    R_max: float = 40
    n_rec: int = 3
    stepping: float = 1.1


@dataclass
class HaloMassFunctionParameters:
    """
    Parameters related to analytical halo mass function (PS formalism. Used to compute variance in EPS_MAR, and for subhalo MF in excursion set).
    
    Attributes:
        filter: tophat, sharpk or smoothk
        c: scale to halo mass relation (1 for tophat, 2.5 for sharp-k, 3 for smooth-k)
        q: q for f(nu) [0.707,1,1] for [ST,smoothk or sharpk,PS] (q = 0.8 with tophat fits better the high redshift z>6 HMF)
        p: p for f(nu) [0.3,0.3,0] for [ST,smoothk or sharpk,PS]
        delta_c: critical density
        A: A = 0.322 except 0.5 for PS Spherical collapse (to double check)
        m_min: Minimum mass
        m_max: Maximum mass
        Mbin: Number of mass bins
        z: Output redshift values. Should be a list.
    """
    filter: Literal['tophat', 'sharpk', 'smoothk'] = 'tophat'
    c: float = 1
    q: float = 0.85
    p: float = 0.3
    delta_c: float = 1.686
    A: float = 0.322
    m_min: float = 1e4
    m_max: float = 1e16
    Mbin: int = 300
    z: list = field(default_factory=lambda: [0])





@dataclass
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
    source: SourceParameters = field(default_factory = lambda: SourceParameters())
    solver: SolverParameters = field(default_factory = lambda: SolverParameters())
    cosmology: CosmologyParameters = field(default_factory = lambda: CosmologyParameters)
    simulation: SimulationParameters = field(default_factory = lambda: SimulationParameters())
    excursion_set: ExcursionSetParameters = field(default_factory = lambda: ExcursionSetParameters())
    halo_mass_function: HaloMassFunctionParameters = field(default_factory = lambda: HaloMassFunctionParameters())
