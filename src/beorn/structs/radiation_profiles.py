from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .base_struct import BaseStruct
from .parameters import Parameters


@dataclass(slots = True)
class RadiationProfiles(BaseStruct):
    """
    Flux profiles around star forming halos, computed for a range of halo masses and accretion rates (alpha). The central assumption is that each halo with given key properties (mass, alpha) produces the same radiation profile, meaning these profiles can be reused for multiple halos in the painting step of the simulation.
    """

    @classmethod
    def get_file_path(cls, directory: Path, parameters: Parameters, **kwargs) -> Path:
        """Cache key covers only the parameters that shape the 1D profiles.

        Random seed and grid dimensions (Ncell, Lbox, DIM) are excluded so
        that profiles computed for one py21cmfast realisation are reused when
        painting a different seed or grid resolution with the same physics.
        """
        return directory / f"RadiationProfiles_{parameters.profiles_hash()}.h5"

    z_history: np.ndarray
    """redshift range for which the profiles have been computed. Corresponds to the parameters.solver.redshifts parameter"""

    halo_mass_bins: np.ndarray
    """bin edges of the halo masses that the profiles have been computed for. The radiation profile at index i corresponds to the halo mass range [halo_mass_bins[i], halo_mass_bins[i+1]]"""

    # the core profiles:
    rho_xray: np.ndarray
    """X-ray profile"""

    rho_heat: np.ndarray
    """heating profile, derived from the X-ray profile"""

    rho_alpha: np.ndarray
    """rho alpha profile"""

    R_bubble: np.ndarray
    """radius of the ionized bubble around the star forming halo"""
    r_lyal: np.ndarray
    """radius of the Lyman alpha halo"""

    # radial component of the profiles
    r_grid_cell: np.ndarray
    """radial grid of the profiles"""


    def profiles_of_halo_bin(self, z_index: int, alpha_index: slice, mass_index:slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the three core profiles for a halo bin.

        Args:
            z_index (int): Redshift index.
            alpha_index (int or slice): Alpha (accretion) index or slice.
            mass_index (int or slice): Mass bin index or slice.

        Returns:
            tuple: ``(R_bubble, rho_alpha, rho_heat)`` arrays copied from the
            stored profiles for the requested bin.
        """
        return(
            self.R_bubble[mass_index, alpha_index, z_index].copy(),
            self.rho_alpha[:, mass_index, alpha_index, z_index].copy(),
            self.rho_heat[:, mass_index, alpha_index, z_index].copy(),
        )


    def validate(self):
        """Check all profile arrays for NaN/Inf values.

        Called automatically after fresh computation. When loading from disk the
        data was already validated at write time, so this is skipped to avoid
        loading the entire (potentially multi-GB) HDF5 dataset into RAM.
        Call explicitly if you need to re-validate a loaded profile.
        """
        assert np.all(np.isfinite(self.rho_xray)), "rho_xray contains invalid values"
        assert np.all(np.isfinite(self.rho_heat)), "rho_heat contains invalid values"
        assert np.all(np.isfinite(self.rho_alpha)), "rho_alpha contains invalid values"
        assert np.all(np.isfinite(self.R_bubble)), "R_bubble contains invalid values"
        assert np.all(np.isfinite(self.r_lyal)), "r_lyal contains invalid values"

    def __post_init__(self):
        BaseStruct.__post_init__(self)
        assert self.z_history.ndim == 1, "z_history must be a 1D array"
        assert self.r_grid_cell.ndim == 1, "r_grid_cell must be a 1D array"
        # Only validate when profiles are freshly computed (not when loaded from disk).
        # np.isfinite forces full materialisation of h5py.Dataset arrays into RAM, which
        # is wasteful (and slow) when the data was already validated at write time.
        if self._file_path is None:
            self.validate()


@dataclass(slots = True)
class RadiationProfilesFStarGrid(BaseStruct):
    """
    Stores radiation profiles on a (mass, alpha, f_st, z) grid. Radial profiles use (r, mass, alpha, f_st, z).
    """
    z_history: np.ndarray
    """Redshift range for which the profiles have been computed. Corresponds to the parameters.solver.redshifts parameter."""

    halo_mass_bins: np.ndarray
    """Bin edges of the halo masses that the profiles have been computed for. The radiation profile at index i corresponds to the halo mass range [halo_mass_bins[i], halo_mass_bins[i+1]]."""

    f_st_grid: np.ndarray
    """1D grid of stellar-fraction values used in the precomputation."""

    rho_xray: np.ndarray
    """X-ray profile. Radial profile with shape (r, mass, alpha, f_st, z)."""

    rho_heat: np.ndarray
    """Heating profile, derived from the X-ray profile. Shape (r, mass, alpha, f_st, z)."""

    rho_alpha: np.ndarray
    """rho alpha profile. Radial profile with shape (r, mass, alpha, f_st, z)."""

    R_bubble: np.ndarray
    """Radius of the ionized bubble around the star forming halo. Shape (mass, alpha, f_st, z)."""

    r_lyal: np.ndarray
    """Radius of the Lyman alpha halo."""

    r_grid_cell: np.ndarray
    """Radial grid of the profiles."""

    def profiles_of_halo_bin(
        self,
        z_index: int,
        alpha_index: int | slice,
        mass_index: int | slice,
        f_st_index: int | slice
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the three core profiles for a halo bin and a given f_st index.

        Args:
            z_index (int): Redshift index.
            alpha_index (int or slice): Alpha (accretion) index or slice.
            mass_index (int or slice): Mass bin index or slice.
            f_st_index (int or slice): f_st grid index or slice.

        Returns:
            tuple: ``(R_bubble, rho_alpha, rho_heat)`` arrays copied from the
            stored profiles for the requested bin and f_st value(s).
        """
        return (
            self.R_bubble[mass_index, alpha_index, f_st_index, z_index].copy(),
            self.rho_alpha[:, mass_index, alpha_index, f_st_index, z_index].copy(),
            self.rho_heat[:, mass_index, alpha_index, f_st_index, z_index].copy(),
        )

    def validate(self):
        """Check all profile arrays for NaN/Inf values."""
        assert np.all(np.isfinite(self.rho_xray)), "rho_xray contains invalid values"
        assert np.all(np.isfinite(self.rho_heat)), "rho_heat contains invalid values"
        assert np.all(np.isfinite(self.rho_alpha)), "rho_alpha contains invalid values"
        assert np.all(np.isfinite(self.R_bubble)), "R_bubble contains invalid values"
        assert np.all(np.isfinite(self.r_lyal)), "r_lyal contains invalid values"

    def __post_init__(self):
        BaseStruct.__post_init__(self)
        assert self.z_history.ndim == 1, "z_history must be a 1D array"
        assert self.r_grid_cell.ndim == 1, "r_grid_cell must be a 1D array"
        assert self.f_st_grid.ndim == 1, "f_st_grid must be a 1D array"

        if self._file_path is None:
            self.validate()
