"""Painting utilities: convert 1D radiation profiles to 3D grids.

This module provides functions that take radial radiation or
temperature profiles and 'paint' them onto a 3D grid of halo positions
using FFT-based convolution kernels. The functions modify the
provided ``output_grid`` in-place.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from astropy.convolution import convolve_fft

from .helpers import profile_to_3Dkernel, stacked_lyal_kernel, stacked_T_kernel, CONVOLVE_FFT_KWARGS

def paint_ionization_profile(
    output_grid: np.ndarray,
    radial_grid,
    x_HII_profile,
    nGrid: int,
    LBox: float,
    z: float,
    halo_grid: np.ndarray
) -> None:
    """Paint an ionization fraction profile onto a 3D grid.

    The function converts a 1D ionization profile described on
    ``radial_grid`` into a 3D convolution kernel and adds the result to
    ``output_grid`` weighted by ``halo_grid`` (halo number density).

    Args:
        output_grid (np.ndarray): 3D array modified in-place.
        radial_grid (array): Radial coordinates of the profile (comoving).
        x_HII_profile (array): Ionization fraction profile as a function of radius.
        nGrid (int): Number of grid cells per axis.
        LBox (float): Box size (Mpc/h).
        z (float): Redshift — used to convert between comoving/physical.
        halo_grid (np.ndarray): 3D halo count mesh (same shape as ``output_grid``).

    Returns:
        None: ``output_grid`` is modified in-place.
    """
    profile_xHII = interp1d(
        x = radial_grid * (1 + z),
        y = x_HII_profile,
        bounds_error = False,
        fill_value = (1, 0)
    )
    kernel_xHII = profile_to_3Dkernel(profile_xHII, nGrid, LBox)
    # self.logger.debug(f"kernel_xHII has {np.sum(np.isnan(kernel_xHII))} NaN values")

    if not np.any(kernel_xHII > 0):
        ### if the bubble volume is smaller than the grid size,we paint central cell with ion fraction value
        # kernel_xHII[int(nGrid / 2), int(nGrid / 2), int(nGrid / 2)] = np.trapz(x_HII_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / nGrid / (1 + z)) ** 3
        output_grid += halo_grid * trapezoid(x_HII_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / nGrid / (1 + z)) ** 3

    else:
        renorm = trapezoid(x_HII_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / (1 + z)) ** 3 / np.mean(kernel_xHII)
        # extra_ion = put_profiles_group(Pos_Halos_Grid[indices], kernel_xHII * 1e-7 / np.sum(kernel_xHII)) * np.sum(kernel_xHII) / 1e-7 * renorm
        output_grid += convolve_fft(
            array = halo_grid,
            kernel = kernel_xHII * 1e-7 / np.sum(kernel_xHII),
            **CONVOLVE_FFT_KWARGS
        ) * np.sum(kernel_xHII) / 1e-7 * renorm
        # bubble_volume = trapezoid(4 * np.pi * radial_grid ** 2 * x_HII_profile, radial_grid)
        # print('bubble volume is ', len(indices) * bubble_volume,'pMpc, grid volume is', np.sum(extra_ion)* (LBox /nGrid/ (1 + z)) ** 3 )
        # Grid_xHII_i += extra_ion


def paint_alpha_profile(
    output_grid: np.ndarray,
    r_lyal,
    x_alpha_prof,
    nGrid,
    LBox,
    minimum_grid_size_lyal,
    z,
    truncate,
    halo_grid: np.ndarray
) -> None:
    """Paint a Lyman-alpha coupling profile onto a 3D grid.

    The function converts a 1D Lyman-alpha coupling profile into a
    3D kernel that accounts for periodic stacking (via
    :func:`stacked_lyal_kernel`) and adds the convolved result to
    ``output_grid`` weighted by ``halo_grid``. Modifies
    ``output_grid`` in-place.

    Args:
        output_grid (np.ndarray): 3D array modified in-place.
        r_lyal (array): Radial coordinates for the Lyman-alpha profile (comoving).
        x_alpha_prof (array): Lyman-alpha coupling profile as a function of radius.
        nGrid (int): Number of grid cells per axis.
        LBox (float): Box size (Mpc/h).
        minimum_grid_size_lyal (int): Minimum grid size used when stacking kernels.
        z (float): Snapshot redshift.
        truncate (bool|float): If float, values below this physical radius are truncated.
        halo_grid (np.ndarray): 3D halo count mesh (same shape as ``output_grid``).

    Returns:
        None: ``output_grid`` is modified in-place.
    """
    # TODO - truncation should not be handled by the PAINT function

    if isinstance(truncate, float):
        # truncate below a certain radius
        x_alpha_prof[r_lyal * (1 + z) < truncate] = x_alpha_prof[r_lyal * (1 + z) < truncate][-1]

    kernel_xal = stacked_lyal_kernel(
        r_lyal * (1 + z),
        x_alpha_prof,
        LBox,
        nGrid,
        nGrid_min = minimum_grid_size_lyal
    )

    if np.any(kernel_xal > 0):
        renorm = trapezoid(x_alpha_prof * 4 * np.pi * r_lyal ** 2, r_lyal) / (LBox / (1 + z)) ** 3 / np.mean(kernel_xal)

        output_grid += convolve_fft(
            array = halo_grid,
            kernel = kernel_xal * 1e-7 / np.sum(kernel_xal),
            **CONVOLVE_FFT_KWARGS
        ) * renorm * np.sum(kernel_xal) / 1e-7
        # Avoid numerical issues when np.sum(kernel) ~ 0


def paint_temperature_profile(
    output_grid: np.ndarray,
    radial_grid,
    Temp_profile,
    nGrid,
    LBox,
    minimum_grid_size_heat,
    z,
    truncate,
    halo_grid: np.ndarray
) -> None:
    """Paint a temperature profile onto a 3D grid.

    Converts a 1D temperature profile into a 3D kernel (using
    :func:`stacked_T_kernel`) and applies it to the halo mesh. The
    function writes the temperature contribution into ``output_grid``
    in-place.

    Args:
        output_grid (np.ndarray): 3D array modified in-place.
        radial_grid (array): Radial coordinates of the temperature profile (comoving).
        Temp_profile (array): Temperature profile as a function of radius.
        nGrid (int): Number of grid cells per axis.
        LBox (float): Box size (Mpc/h).
        minimum_grid_size_heat (int): Minimum grid size used when stacking kernels for temperature.
        z (float): Snapshot redshift.
        truncate (bool|float): If float, values below this physical radius are truncated.
        halo_grid (np.ndarray): 3D halo count mesh (same shape as ``output_grid``).

    Returns:
        None: ``output_grid`` is modified in-place.
    """

    # TODO - truncation should not be handled by the PAINT function
    if isinstance(truncate, float):
        # Truncate below a certain physical radius
        Temp_profile[radial_grid * (1 + z) < truncate] = Temp_profile[radial_grid * (1 + z) < truncate][-1]

    kernel_T = stacked_T_kernel(
        radial_grid * (1 + z),
        Temp_profile,
        LBox,
        nGrid,
        nGrid_min = minimum_grid_size_heat
    )

    if np.any(kernel_T > 0):
        renorm = trapezoid(Temp_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / (1 + z)) ** 3 / np.mean(kernel_T)

        output_grid += convolve_fft(
            array = halo_grid,
            kernel = kernel_T * 1e-7 / np.sum(kernel_T),
            **CONVOLVE_FFT_KWARGS
        ) * np.sum(kernel_T) / 1e-7 * renorm
