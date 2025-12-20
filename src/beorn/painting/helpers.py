"""Helpers for the painting module.

This module provides utilities that convert 1D radial profiles into
3D convolution kernels and helpers to build stacked kernels that
account for profiles extending beyond the simulation box.
"""
import numpy as np
from scipy.interpolate import interp1d
import logging
logger = logging.getLogger(__name__)


CONVOLVE_FFT_KWARGS = {
    "boundary": "wrap",
    "normalize_kernel": False,
    "allow_huge": True
}

TQDM_KWARGS = {
    "desc": "Painting redshift snapshots",
    "unit": "snapshot",
}


def profile_to_3Dkernel(profile: callable, nGrid: int, LB: float) -> np.ndarray:
    """Embed a radial 1D profile into a centered 3D grid kernel.

    Args:
        profile (callable): Callable accepting a radius array and returning the profile values.
        nGrid (int): Number of grid cells per axis.
        LB (float): Box size in comoving units (cMpc/h) used to build the grid extents.

    Returns:
        numpy.ndarray: 3D kernel of shape ``(nGrid, nGrid, nGrid)`` with the profile centered.
    """
    x = np.linspace(-LB / 2, LB / 2, nGrid)
    # y, z will be the same.
    rx, ry, rz = np.meshgrid(x, x, x, sparse=True)
    rgrid = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    kern = profile(rgrid)
    assert np.all(np.isfinite(kern)), "Profile function returned non-finite values."
    return kern



def stacked_lyal_kernel(rr_al, lyal_array, LBox, nGrid, nGrid_min):
    """Build a stacked 3D kernel for extended Lyman-alpha profiles.

        This function paints the lyal profile on a meshgrid whose size is the size where the lyal profile reaches zeros.
    Hence it is larger than LBox. It has a lower resolution than the Grid (nGrid_min = 64). We then chunk this large box into suboxes of sizes LBox and stack them.
    This ensures that despite a small LBox, we ensure full periodic boundary conditions and account for the wide spread of lyal profiles. TODO: explain better.

    Args:
        rr_al (array): Comoving radii at which ``lyal_array`` is defined.
        lyal_array (array): Lyman-alpha profile values.
        LBox (float): Box size (Mpc/h).
        nGrid (int): Target grid resolution (cells per axis).
        nGrid_min (int): Minimum resolution used for the stacked high-res kernel.

    Returns:
        numpy.ndarray: 3D kernel at resolution ``nGrid`` covering the
        Lyman-alpha profile with periodic stacking.
    """
    profile_xal_HM = interp1d(rr_al, lyal_array, bounds_error=False, fill_value=0)  ##screening
    ind_lya_0 = np.min(np.where(lyal_array == 0))  ## indice where the lyman alpha profile gets to zero
    rr_al_max = rr_al[ind_lya_0]  ### max radius that we need to consider to fully include the lyman alpha profile
    box_extension = int(rr_al_max / (LBox / 2)) + 1

    # nGrid_min = 64
    if box_extension < 1:
        box_extension = 1

    elif box_extension % 2 == 0:
        box_extension += 1  ### this need to be even to make things work

    kernel_xal_HM = profile_to_3Dkernel(profile_xal_HM, box_extension * nGrid_min, box_extension * LBox)
    # kernel_xal_HM = profile_to_3Dkernel(profile_xal_HM, box_extension * nGrid_min, box_extension * LBox)
    # nGrid_extd = box_extension * nGrid_min
    # LBox_extd = box_extension * LBox  ## size and nbr of pix of the larger box

    stacked_xal_ker = np.zeros((nGrid_min, nGrid_min, nGrid_min))
    for ii in range(box_extension):  ## loop over the box_extension**3 subboxes and stack them
        for jj in range(box_extension):
            for kk in range(box_extension):
                stacked_xal_ker += kernel_xal_HM[
                    ii * nGrid_min:(ii + 1) * nGrid_min,
                    jj * nGrid_min:(jj + 1) * nGrid_min,
                    kk * nGrid_min:(kk + 1) * nGrid_min
                ]

    pix_lft = int(box_extension / 2) * nGrid_min  ### coordinate of the central subbox
    pix_rgth = (1 + int(box_extension / 2)) * nGrid_min
    ## remove the central box, to then add it later with full nGrid resolution
    stacked_xal_ker = stacked_xal_ker - kernel_xal_HM[pix_lft:pix_rgth, pix_lft:pix_rgth, pix_lft:pix_rgth]

    incr_rez = np.asarray(np.arange(0, nGrid) * nGrid_min / nGrid, int)  ## indices to then add

    kernel_xal_HM = profile_to_3Dkernel(profile_xal_HM, nGrid, LBox) + stacked_xal_ker[incr_rez, incr_rez, incr_rez]

    return kernel_xal_HM


def stacked_T_kernel(rr_T, T_array, LBox, nGrid, nGrid_min):
    """Build a stacked 3D kernel for extended temperature profiles.

    Identical in purpose to :func:`stacked_lyal_kernel` but tailored for temperature (X-ray heating) profiles.

    Args:
        rr_T (array): Comoving radii at which ``T_array`` is defined.
        T_array (array): Temperature profile values.
        LBox (float): Box size (Mpc/h).
        nGrid (int): Target grid resolution (cells per axis).
        nGrid_min (int): Minimum resolution used for the stacked high-res kernel.

    Returns:
        numpy.ndarray: 3D kernel at resolution ``nGrid`` covering the temperature profile with periodic stacking.
    """
    profile_T_HM = interp1d(rr_T, T_array, bounds_error=False, fill_value=0)  ##screening

    zero_K_indices = np.where(T_array < 1e-6)[0]
    if len(zero_K_indices) > 0:
        ind_T_0 = np.min(zero_K_indices)  ## indice where the T profile drops, xray haven't reached that scale
    else:
        ind_T_0 = -1  ## if T_array is always > 1e-6, we just take the whole profile...

    rr_T_max = rr_T[ind_T_0]  ### max radius that we need to consider to fully include the extended T profile
    box_extension = int(rr_T_max / (LBox / 2)) + 1

    # nGrid_min = 64
    if box_extension < 1:
        box_extension = 1

    elif box_extension % 2 == 0:
        box_extension += 1  ### this need to be even to make things work

    kernel_T_HM = profile_to_3Dkernel(profile_T_HM, box_extension * nGrid_min, box_extension * LBox)
    # nGrid_extd = box_extension * nGrid_min
    # LBox_extd = box_extension * LBox  ## size and nbr of pix of the larger box

    stacked_T_ker = np.zeros((nGrid_min, nGrid_min, nGrid_min))
    for ii in range(box_extension):  ## loop over the box_extension**3 subboxes and stack them
        for jj in range(box_extension):
            for kk in range(box_extension):
                stacked_T_ker += kernel_T_HM[ii * nGrid_min:(ii + 1) * nGrid_min, jj * nGrid_min:(jj + 1) * nGrid_min,
                                 kk * nGrid_min:(kk + 1) * nGrid_min]

    pix_lft = int(box_extension / 2) * nGrid_min  ### coordinate of the central subbox
    pix_rgth = (1 + int(box_extension / 2)) * nGrid_min
    ## remove the central box, to then add it later with full nGrid resolution
    stacked_T_ker = stacked_T_ker - kernel_T_HM[pix_lft:pix_rgth, pix_lft:pix_rgth, pix_lft:pix_rgth]

    incr_rez = np.asarray(np.arange(0, nGrid) * nGrid_min / nGrid, int)  ## indices to then add

    kernel_T_HM = profile_to_3Dkernel(profile_T_HM, nGrid, LBox) + stacked_T_ker[incr_rez, incr_rez, incr_rez]

    return kernel_T_HM
