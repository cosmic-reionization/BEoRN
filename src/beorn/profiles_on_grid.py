import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
from astropy.convolution import convolve_fft
import datetime
from scipy.interpolate import splrep, splev, interp1d
import time
from .functions import print_time
from .cloud_in_cell import CIC_coefficients
from .parameters import Parameters

def log_binning(array1, array2):
    """
     Parameters
     ----------
     array1, array2 : 2 arrays of different sizes.

     Returns
     -------
     Return the indices of the bins (array2) to which each value of input array (array1) belongs, in logarithmic scale
    """

    # np.argmin(np.abs(np.log10(Mh_[:, None] / Mh_history_HR[ind_z, :])), axis=1)
    return np.digitize(x=np.log10(array1), bins=np.log10(array2), right=False)



def bin_edges_log(array):
    """
     Parameters
     ----------
     array : array

     Returns
     -------
     An array whose bins centers correspond to the values of array1, with log scaling.
    """

    Mh_bin_array = np.concatenate(([array[0] / 2], np.sqrt(array[1:] * array[:-1]), [2 * array[-1]]))
    return Mh_bin_array



def average_profile(parameters: Parameters, profile, Mh_, ind_z, i):
    """
     Inside a given halo mass bin, computes the average profile inside that bin (by calling the HR profiles with higher bin resolution)
     If param.sim.average_profiles_in_bin is False, we skip this step and just return the profile of the halo mass bin.

     Parameters
     ----------
     profile : object from the class profiles, in compute_profiles.py
     Mh_ : array of halo masses that belong to bin i.
     ind_z  : redshift indice
     i : indice of the coarse mass bin

     Returns
     -------
     mean R bubble (float), mean lyalpha (array of size r_lyal), mean Tk profiles (array of size radial_grid)
    """
    # i : indice of the coarse mass bin
    if parameters.simulation.average_profiles_in_bin:
        Mh_history_HR = profile.Mh_history_HR
        R_bubble_HR = profile.R_bubble_HR[ind_z, :]

        # if Mh_ is digitized to M_bin_edges[i] it means it should take the value of M_Bin[i-1] (bin 0 is what's on the left...)
        # Mh_history_HR       0.   1.    2.    3.  ...
        # M_bin_edges     0.  | 1. |  2. |  3. |  4. ....
        M_bin_edges = bin_edges_log(Mh_history_HR[ind_z, :])
        HR_indexing = log_binning(Mh_,M_bin_edges )  # np.argmin(np.abs(np.log10(Mh_[:, None] / Mh_history_HR[ind_z, :])), axis=1)

        HR_indices, nbr_count = np.unique(HR_indexing, return_counts=True)
        HR_indices = HR_indices-1

        ### these are the profiles with finer resolution for each individual halo masses that are in the mass bin
        R_bubbles_in = R_bubble_HR[HR_indices]
        rho_alpha_in = profile.rho_alpha_HR[ind_z, :, HR_indices]
        Tk_in = profile.rho_heat_HR[ind_z, :, HR_indices]

        ### this is then the new mean profiles accounting for the non uniform distrib of halos inside bin.
        mean_R_bubble_in_bin = np.sum(R_bubbles_in * nbr_count) / np.sum(nbr_count)
        mean_rho_alpha_in_bin = np.sum(nbr_count[:, None] * rho_alpha_in, axis=0) / np.sum(nbr_count)  # .shape
        mean_Tk_in_bin = np.sum(nbr_count[:, None] * Tk_in, axis=0) / np.sum(nbr_count)  # .shape

    else:  # no averaging, just the usual thing with coarse M_Bin
        mean_R_bubble_in_bin = profile.R_bubble[ind_z, i]
        mean_rho_alpha_in_bin = profile.rho_alpha[ind_z, :, i]
        mean_Tk_in_bin = profile.rho_heat[ind_z, :, i]

    return mean_R_bubble_in_bin, mean_rho_alpha_in_bin, mean_Tk_in_bin


def cumulated_number_halos(parameters: Parameters, H_X, H_Y, H_Z, cic=False):
    """
     Returns the number of halo per grid pixel, for halos belonging to a mass bin.
     This function is called in paint_profile_single_snap, in run.py

     Parameters
     ----------
     H_X, H_Y, H_Z  : arrays with X,Y,Z positions of halos (in a mass bin), expressed in real spatial unit (Mpc/h)
     cic  : boolean, to turn or not the cloud in cell option.

     Returns
     -------
     unique_base_nGrid_poz : an array with all the cells occupied by halos, expressed in base nGrid
     nbr_of_halos : array of same size with the cumulated number of halos per pixel
    """
    LBox = parameters.simulation.Lbox  # Mpc/h
    nGrid = parameters.simulation.Ncell  # number of grid cells

    if not cic:
        Pos_Halos = np.vstack((H_X, H_Y, H_Z)).T  # Halo positions.
        Pos_Halos_Grid = np.array([Pos_Halos / LBox * nGrid]).astype(int)[0]
        Pos_Halos_Grid[np.where(
            Pos_Halos_Grid == nGrid)] = nGrid - 1  # we don't want Pos_Halos_Grid==nGrid. This only happens if Pos_Bubbles=LBox
        base_nGrid_position = Pos_Halos_Grid[:, 0] + nGrid * Pos_Halos_Grid[:, 1] + nGrid ** 2 * Pos_Halos_Grid[:, 2]
        unique_base_nGrid_poz, nbr_of_halos = np.unique(base_nGrid_position, return_counts=True)

    if cic:
        pixel_coordinates_output, cic_coeff_output = CIC_coefficients(parameters, H_X, H_Y, H_Z)
        base_nGrid_cic_position = pixel_coordinates_output[0] + nGrid * pixel_coordinates_output[1] + nGrid ** 2 * \
                                  pixel_coordinates_output[2]

        unique_positions = np.unique(base_nGrid_cic_position)
        unique_cic_coeff = np.unique(cic_coeff_output)
        cumulated_coeff  = np.zeros(len(unique_positions))

        print('len of unique_cic_coeff is',len(unique_cic_coeff))
        for i in range(len(unique_cic_coeff)):

            ## all the indices that have the same cic coeff
            indexes = np.where(cic_coeff_output == unique_cic_coeff[i])

            ### all the unique positions of the halos that have the same cic coeff, and how many there are per position
            unique_pos_with_same_cic_coeff, nbr_of_halos = np.unique(base_nGrid_cic_position[indexes], return_counts=True)

            mask = np.isin(unique_positions, unique_pos_with_same_cic_coeff)
            cumulated_coeff[mask] += nbr_of_halos * unique_cic_coeff[i]

        unique_base_nGrid_poz, nbr_of_halos = unique_positions, cumulated_coeff

############ this is the slower version
#        unique_positions = np.unique(base_nGrid_cic_position)
#        cumulated_coeff = np.zeros(len(unique_positions))
#        ## cumulated cic coeff for each pixel in "unique_values"
#        for i in range(len(unique_positions)):
#            indexes = np.where(base_nGrid_cic_position == unique_positions[i])
#            cumulated_coeff[i] = np.sum(cic_coeff_output[indexes])
#            # each "unique_positions" (pixel coordinate expressed in base_n_grid) has
#            # a cumulated cic coeff of "cumulated_coeff"
#        unique_base_nGrid_poz, nbr_of_halos = unique_positions, cumulated_coeff
########### this is the end of the slow version
    return unique_base_nGrid_poz, nbr_of_halos


## Creating a profile kernel
def profile_to_3Dkernel(profile, nGrid, LB):
    """
    Put profile_1D on a grid

    Parameters
    ----------
    profile  : profile_1D(r, c1=2, c2=5).
    nGrid, LB  : number of grids and boxsize (in cMpc/h) respectively

    Returns
    -------
    meshgrid of size (nGrid,nGrid,nGrid), with the profile at the center.
    """
    x = np.linspace(-LB / 2, LB / 2, nGrid)  # y, z will be the same.
    rx, ry, rz = np.meshgrid(x, x, x, sparse=True)
    rgrid = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    kern = profile(rgrid)
    return kern


def put_profiles_group(source_pos, nbr_of_halos, profile_kern, nGrid=None):
    '''
    source_pos : the position of halo centered in units number of grid cell (0..nGrid-1). shape is (3,N), with N the number of halos. (X,Y,Z)
    nbr_of_halos : the number of halos in each source_pos (>0), array of size len(source_pos)
    profile_kern : the profile to put on a grid around each source_pos pixels, multiplied by nbr_of_halos (output of profile_to_3Dkernel)

    Bin halos masses to do this. Then in a given bin all halos are assumed to have the same profile. This speeds up dramatically this step.
    '''
    if nGrid is None: nGrid = profile_kern.shape[0]
    source_grid = np.zeros((nGrid, nGrid, nGrid))
    # for i, j, k in source_pos:
    #    source_grid[i, j, k] += 1
    source_grid[source_pos[0], source_pos[1], source_pos[2]] = nbr_of_halos
    out = convolve_fft(source_grid, profile_kern, boundary='wrap', normalize_kernel=False, allow_huge=True)
    return out


def Spreading_Excess(Grid_Storage):
    """
    Spread the excess photons using scipy.measure.label and distance_transform_edt. For each connected regions, spread the photons to the first closest set of pixels, The last boundary will be filled with an equal fraction of the remaining excess x_ion.
    This is the first version of the function, where we loop over every connected region and run distance transform over the whole box everytime.
    In the next versions of the function we speed this up, but this current version remains good to test if the faster one are doing their job correctly.
    """
    Grid = np.copy(Grid_Storage)
    nGrid = Grid.shape[0]
    Binary_Grid = np.copy(Grid)
    Binary_Grid[np.where(Grid < 0.999)] = 0
    Binary_Grid[np.where(Grid >= 0.999)] = 1
    connected_regions = label(Binary_Grid)

    Nbr_regions = np.max(connected_regions) + 1

    Grid_of_1 = np.full(((nGrid, nGrid, nGrid)), 1)
    Grid_of_0 = np.zeros((nGrid, nGrid, nGrid))

    # When i = 0, the region if the full region outside the bubbles
    X_Ion_Tot_i = np.sum(Grid)
    print('initial sum of ionized fraction :', int(np.sum(Grid)))

    if X_Ion_Tot_i > Grid.size:
        print('Universe is fully ionized.')
        return 1

    else:
        for i in range(1, Nbr_regions):
            connected_indices = np.where(connected_regions == i)
            Grid_connected = np.copy(Grid_of_0)  ## Grid with the fiducial value only for the region i.
            Grid_connected[connected_indices] = Grid[connected_indices]
            ## take sub grid with only the connected region, find pixels where xion>1, sum the excess, and set these pixels to 1.
            overlap = np.where(Grid_connected > 1)

            excess_ion = np.sum(Grid_connected[overlap] - 1)
            initial_excess = excess_ion
            Grid[overlap] = 1

            Inverted_grid = np.copy(Grid_of_1)
            Inverted_grid[connected_indices] = 0

            sum_distributed_xion = 0
            if excess_ion > 1e-7:  ### small value but non zero to avoid doing that step when excess ion is very small
                dist_from_boundary = distance_transform_edt(Inverted_grid)
                dist_from_boundary[np.where(dist_from_boundary == 0)] = 2 * nGrid  ### eliminate pixels inside boundary
                dist_from_boundary[np.where(
                    Grid > 1)] = 2 * nGrid  ### eliminate pixels that already have excess x_ion (belonging to another connected regions..)
                minimum = np.min(dist_from_boundary)
                boundary = np.where(
                    dist_from_boundary == minimum)  # np.where((dist_from_boundary == minimum )& ( Grid<1))

                if np.sum(1 - Grid[boundary]) > excess_ion:  # if their is room for the excess ion,
                    #  you add in each cell a fraction of the neutral fraction available.
                    Grid[boundary] += (1 - Grid[boundary]) * excess_ion / np.sum(1 - Grid[boundary])
                    if np.any(Grid[boundary] > 1):
                        print('x_ion > 1')
                    sum_distributed_xion += excess_ion
                else:

                    while np.sum(1 - Grid[boundary]) < excess_ion:
                        # print('have to go for more than 1 layer')
                        sum_distributed_xion += np.sum(1 - Grid[boundary])
                        excess_ion = excess_ion - np.sum(1 - Grid[boundary])
                        Grid[boundary] = 1
                        dist_from_boundary[boundary] = nGrid * 2  ### exclude this layer for next step
                        minimum = np.min(dist_from_boundary)
                        boundary = np.where(
                            dist_from_boundary == minimum)  ### new closest region to fill with excess ion
                    # you go out of the *while* when np.sum(1 - Grid[boundary]) > excess_ion
                    residual_excess = (1 - Grid[boundary]) * excess_ion / np.sum(1 - Grid[boundary])
                    Grid[boundary] += residual_excess
                    sum_distributed_xion += excess_ion

                    if np.any(Grid[boundary] > 1):
                        print('x_ion > 1 at the end of the process', aaaa)
                        break

        if np.any(Grid > 1):
            print('3. x_ion > 1 ')

        print('final xion sum: ', int(np.sum(Grid)))
        X_Ion_Tot_f = np.sum(Grid)
        if int(X_Ion_Tot_f) != int(X_Ion_Tot_i):
            print('smtg is wrong when spreading xion_excess.')

    return Grid


def spreading_excess_fast(parameters: Parameters, Grid_input, plot__=False):
    """
    Last and fastest version of the function.
    Input : Grid_Storage, the cosmological mesh grid (X,X,X) with the ionized fractions, with overlap (pixels where x_ion>1). (X can be 256, 512 ..)
    A word regarding the elements of this function :
        - Binary_Grid : contains 1 where Grid_input>=1 and 0 elsewhere. This grid is used as input for scipy.measure.label. Format is (X,X,X).
        - Connected_regions : (X,X,X). Output of skimage.measure.label. Each pixel of it is labeled according to the ionized clump it belongs to.
        - x_ion_tot_i : total sum of ionizing fraction.
        - region_nbr, size_of_region : region label, and size of it . We use it to idenfity the very small regions (small_regions) with less than "pix_thresh" pixels. We treat them all together to speed up the process
        - Spread_Single :  spread the excess photons.
    """

    nGrid = len(Grid_input[0])
    Grid = np.copy(Grid_input)

    if parameters.simulation.thresh_pixel is None:
        pix_thresh = 80 * (nGrid / 256) ** 3
    else:
        pix_thresh = parameters.simulation.thresh_pixel

    Binary_Grid = np.copy(Grid)
    Binary_Grid[np.where(Grid < 0.9999999)] = 0
    Binary_Grid[np.where(Grid >= 0.9999999)] = 1

    # The first region (i=0) is the still neutral IGM, in between the bubbles
    label_image = label(Binary_Grid)

    # Periodic boundary conditions for label_image
    # assign  same label to ionised regions that are connected through left/right, up/down, front/back box boundaries
    '''''''''
    t0_PBC = time.time()
    PBC_indices = np.where(label_image[0, :, :] * label_image[-1, :, :] > 0)  # neighbor ionised indices
    label_to_change, indice = np.unique(label_image[-1, PBC_indices[0], PBC_indices[1]], return_index=True)  # labels in the upper layer that should be changed to neighbor labels in the lower layer
    replacement_label = label_image[0, PBC_indices[0], PBC_indices[1]][indice]
    for il in range(len(label_to_change)):
        label_image[label_image == label_to_change[il]] = replacement_label[il]

    PBC_indices = np.where(label_image[:, 0, :] * label_image[:, -1, :] > 0)
    label_to_change, indice = np.unique(label_image[PBC_indices[0], -1, PBC_indices[1]], return_index=True)
    replacement_label = label_image[PBC_indices[0], 0, PBC_indices[1]][indice]
    for il in range(len(label_to_change)):
        label_image[label_image == label_to_change[il]] = replacement_label[il]

    PBC_indices = np.where(label_image[:, :, 0] * label_image[:, :, -1] > 0)
    label_to_change, indice = np.unique(label_image[PBC_indices[0], PBC_indices[1], -1], return_index=True)
    replacement_label = label_image[PBC_indices[0], PBC_indices[1], 0][indice]
    for il in range(len(label_to_change)):
        label_image[label_image == label_to_change[il]] = replacement_label[il]
    print('Imposing PBC on label_image took', print_time(time.time()-t0_PBC))
    '''''''''

    x_ion_tot_i = np.sum(Grid)
    print('initial sum of ionized fraction :', round(np.sum(Grid), 3))

    if x_ion_tot_i > Grid.size:
        print('Universe is fully ionized.')
        Grid = np.array([1])

    else:
        print('Universe not fully ionized : xHII is', round(x_ion_tot_i / Grid.size, 4))

        region_nbr, size_of_region = np.unique(label_image, return_counts=True)
        print(len(region_nbr), 'connected regions.')
        label_max = np.max(label_image)

        small_regions = np.where(np.isin(label_image, region_nbr[np.where(size_of_region < pix_thresh)[
            0]]))  ## small_regions : Gridmesh indices gathering all the connected regions that have less than 10 pixels
        Small_regions_labels = region_nbr[np.where(size_of_region < pix_thresh)[
            0]]  ## labels of the small regions. Use this to exclude them from the for loop

        initial_excess = np.sum(Grid[small_regions] - 1)
        excess_ion = initial_excess

        print('there are ', len(Small_regions_labels), 'connected regions with less than ', pix_thresh,
              ' pixels. They contain a fraction ', round(excess_ion / x_ion_tot_i, 4),
              'of the total ionisation fraction.')

        Grid = spread_single(parameters, Grid, small_regions, print_time=None)  # Do the spreading for the small regions
        if np.any(Grid[small_regions] > 1):
            print('small regions not correctly spread')

        all_regions_labels = np.array(range(1, label_max + 1))  # the remaining larges overlapping ionized regions
        large_regions_labels = all_regions_labels[np.where(np.isin(all_regions_labels, Small_regions_labels) == False)[
            0]]  # indices of regions that have more than pix_thresh pixels

        # Then do the spreading individually for large regions
        for i, ir in enumerate(large_regions_labels):
            if plot__:
                if i % 100 == 0:
                    print('doing region ', i, 'over ', len(large_regions_labels), ' regions in total')
            connected_indices = np.where(label_image == ir)
            Grid = spread_single(parameters, Grid, connected_indices, print_time=None)

        if np.any(Grid > 1.):
            print('Some grid pixels are still in excess.')

        print('final xion sum: ', round(np.sum(Grid), 3))
        X_Ion_Tot_f = np.sum(Grid)
        if int(X_Ion_Tot_f) != int(x_ion_tot_i):
            print('Something is wrong when redistributing photons from the overlapping regions. See '
                  'Spreading_Excess_Fast.')

    return Grid


def spread_single(parameters: Parameters, Grid, connected_indices, print_time=None):
    """
    This spreads the excess ionizing photons for a given region.
    Input :
    - Grid : The meshgrid containing the ionizing fractions
    - Connected_indices : The indices of the ionized region from which you want to spread the overlaps. (excess_ion)
    - print_time : if it's not None, will print the time taken, along with the message contained in "print_time".

    Return : the same grid but with the excess ion fraction of the connected region spread around.

    Trick : we run distance_transform only for a sub-box centered on the connected region. This is particularly important for high resolution grids, when distance_transform_edt starts to take time (~s, but multilplied by the number of connected regions >1e4, starts taking time...)
            the size of the subbox is N_subgrid. It is called Sub_Grid.
    """

    nGrid = len(Grid[0])
    time_start = datetime.datetime.now()

    initial_excess = np.sum(Grid[connected_indices] - 1)
    Grid[connected_indices] = np.where(Grid[connected_indices] > 1, 1, Grid[connected_indices])
    excess_ion = initial_excess

    if initial_excess > 1e-8:
        ## take sub grid with only the connected region, find pixels where xion>1, sum the excess, and set these pixels to 1.
        Inverted_grid = np.full(((nGrid, nGrid, nGrid)), 1)
        Inverted_grid[connected_indices] = 0
        sum_distributed_xion = 0

        Delta_pixel = int(excess_ion ** (1. / 3) / 2) + 1

        Min_X, Max_X = np.min(connected_indices[0]), np.max(connected_indices[0])
        Min_Y, Max_Y = np.min(connected_indices[1]), np.max(connected_indices[1])
        Min_Z, Max_Z = np.min(connected_indices[2]), np.max(connected_indices[2])
        Delta_max = np.max((Max_X - Min_X + 0, Max_Y - Min_Y + 0, Max_Z - Min_Z + 0))
        Center_X, Center_Y, Center_Z = int((Min_X + Max_X) / 2), int((Min_Y + Max_Y) / 2), int((Min_Z + Max_Z) / 2)

        if parameters.simulation.subgrid_approximation:  # Is this flag is True, then you set the subgrid size
            N_subgrid = 2 * (Delta_max + 2 * Delta_pixel)  ## length of subgrid embedding the connected region
            if N_subgrid % 2 == 1:
                N_subgrid += 1  ###Nsubgrid needs to be even to make things easier

        else:  # Is approx is False, then set N_subgrid >nGrid, so that we never do the subbox trick (this is to check if the trick gives good results compared to the full)
            N_subgrid = nGrid + 1

        if N_subgrid > nGrid:
            dist_from_boundary = distance_transform_edt(Inverted_grid)
            dist_from_boundary[np.where(dist_from_boundary == 0)] = 2 * nGrid  ### eliminate pixels inside boundary
            dist_from_boundary[np.where(
                Grid > 1)] = 2 * nGrid  ### eliminate pixels that already have excess x_ion (belonging to another connected regions..)
            minimum = np.min(dist_from_boundary)
            boundary = np.where(dist_from_boundary == minimum)  # np.where((dist_from_boundary == minimum )& ( Grid<1))

            while np.sum(1 - Grid[boundary]) < excess_ion:
                sum_distributed_xion += np.sum(1 - Grid[boundary])
                excess_ion = excess_ion - np.sum(1 - Grid[boundary])
                Grid[boundary] = 1
                dist_from_boundary[boundary] = nGrid * 2  ### exclude this layer for next step
                minimum = np.min(dist_from_boundary)
                boundary = np.where(dist_from_boundary == minimum)  ### new closest region to fill with excess ion
            # you go out of the *while* when np.sum(1 - Grid[boundary]) > excess_ion
            residual_excess = (1 - Grid[boundary]) * excess_ion / np.sum(1 - Grid[boundary])
            Grid[boundary] += residual_excess
            sum_distributed_xion += excess_ion


        else:

            Sub_Grid = np.full(((N_subgrid, N_subgrid, N_subgrid)), 0)

            Sub_Grid = Sub_Grid.astype('float64')

            Sub_Grid[:] = Grid[np.max((Center_X - int(N_subgrid / 2), 0)) - np.max(
                (0, Center_X + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                (nGrid, Center_X + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_X)),
                          np.max((Center_Y - int(N_subgrid / 2), 0)) - np.max(
                              (0, Center_Y + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                              (nGrid, Center_Y + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_Y)),
                          np.max((Center_Z - int(N_subgrid / 2), 0)) - np.max(
                              (0, Center_Z + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                              (nGrid, Center_Z + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_Z))]

            while np.sum(
                    1 - Sub_Grid) < excess_ion:  ### just check if Sub_Grid has enough room for excess_ion. If not, increase its size N_subgrid.
                N_subgrid = N_subgrid + 2
                Sub_Grid = np.full(((N_subgrid, N_subgrid, N_subgrid)), 0)
                Sub_Grid = Sub_Grid.astype('float64')
                Sub_Grid[:] = Grid[np.max((Center_X - int(N_subgrid / 2), 0)) - np.max(
                    (0, Center_X + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                    (nGrid, Center_X + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_X)),
                              np.max((Center_Y - int(N_subgrid / 2), 0)) - np.max(
                                  (0, Center_Y + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                                  (nGrid, Center_Y + int(N_subgrid / 2) + 0)) + np.max(
                                  (0, int(N_subgrid / 2) - Center_Y)),
                              np.max((Center_Z - int(N_subgrid / 2), 0)) - np.max(
                                  (0, Center_Z + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                                  (nGrid, Center_Z + int(N_subgrid / 2) + 0)) + np.max(
                                  (0, int(N_subgrid / 2) - Center_Z))]

            Sub_Inverted_Grid = np.full(((N_subgrid, N_subgrid, N_subgrid)), 1)
            Sub_Inverted_Grid = Sub_Inverted_Grid.astype('float64')
            Sub_Inverted_Grid[:] = Inverted_grid[np.max((Center_X - int(N_subgrid / 2), 0)) - np.max(
                (0, Center_X + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                (nGrid, Center_X + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_X)),
                                   np.max((Center_Y - int(N_subgrid / 2), 0)) - np.max(
                                       (0, Center_Y + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                                       (nGrid, Center_Y + int(N_subgrid / 2) + 0)) + np.max(
                                       (0, int(N_subgrid / 2) - Center_Y)),
                                   np.max((Center_Z - int(N_subgrid / 2), 0)) - np.max(
                                       (0, Center_Z + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                                       (nGrid, Center_Z + int(N_subgrid / 2) + 0)) + np.max(
                                       (0, int(N_subgrid / 2) - Center_Z))]

            Sub_Grid_Initiale = np.copy(Sub_Grid)

            dist_from_boundary = distance_transform_edt(Sub_Inverted_Grid)
            dist_from_boundary[np.where(dist_from_boundary == 0)] = 2 * N_subgrid  ### eliminate pixels inside boundary
            dist_from_boundary[np.where(
                Sub_Grid >= 1)] = 2 * N_subgrid  ### eliminate pixels that already have excess x_ion (belonging to another connected regions..)
            minimum = np.min(dist_from_boundary)
            boundary = np.where(dist_from_boundary == minimum)

            excess_ion_i = excess_ion
            while np.sum(1 - Sub_Grid[boundary]) < excess_ion:
                sum_distributed_xion += np.sum(1 - Sub_Grid[boundary])
                excess_ion = excess_ion - np.sum(1 - Sub_Grid[boundary])
                Sub_Grid[boundary] = 1
                dist_from_boundary[boundary] = N_subgrid * 2  ### exclude this layer for nex
                minimum = np.min(dist_from_boundary)
                boundary = np.where(
                    dist_from_boundary == minimum)  ### new closest region to fill # you go out of the *while* when np.sum(1 - Grid[boundary]) > excess_ion

            residual_excess = (1 - Sub_Grid[boundary]) * excess_ion / np.sum(1 - Sub_Grid[boundary])

            Sub_Grid[boundary] = np.add(Sub_Grid[boundary], residual_excess)
            sum_distributed_xion += excess_ion

            Grid[
            np.max((Center_X - int(N_subgrid / 2), 0)) - np.max((0, Center_X + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                (nGrid, Center_X + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_X)),
            np.max((Center_Y - int(N_subgrid / 2), 0)) - np.max((0, Center_Y + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                (nGrid, Center_Y + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_Y)),
            np.max((Center_Z - int(N_subgrid / 2), 0)) - np.max((0, Center_Z + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                (nGrid, Center_Z + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_Z))] = Sub_Grid[:]

            if np.any(Sub_Grid[boundary] > 1) or np.any(np.isnan(Sub_Grid[boundary])):
                print('2. Thats where we trigger')

            if round(int(np.sum(Sub_Grid)) / int(np.sum(Sub_Grid_Initiale) + excess_ion_i)) != 1:
                print('loosing photons')
                exit()

    if print_time is not None:
        time_end = datetime.datetime.now()
        print('It took : ', time_end - time_start, 'to spread the overlap of', print_time)

    return Grid


def stacked_lyal_kernel(rr_al, lyal_array, LBox, nGrid, nGrid_min):
    """
    This function paints the lyal profile on a meshgrid whose size is the size where the lyal profile reaches zeros.
    Hence it is larger than LBox. It has a lower resolution than the Grid (nGrid_min = 64). We then chunk this large box into suboxes of sizes LBox and stack them.
    This ensures that despite a small LBox, we ensure full periodic boundary conditions and account for the wide spread of lyal profiles.
    rr_al : the comoving radius range
    lyal_array : the lyal profile (array)
    LBox,nGrid : the box size and grid rez of the current run.
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
                stacked_xal_ker += kernel_xal_HM[ii * nGrid_min:(ii + 1) * nGrid_min,
                                   jj * nGrid_min:(jj + 1) * nGrid_min, kk * nGrid_min:(kk + 1) * nGrid_min]

    pix_lft = int(box_extension / 2) * nGrid_min  ### coordinate of the central subbox
    pix_rgth = (1 + int(box_extension / 2)) * nGrid_min
    ## remove the central box, to then add it later with full nGrid resolution
    stacked_xal_ker = stacked_xal_ker - kernel_xal_HM[pix_lft:pix_rgth, pix_lft:pix_rgth, pix_lft:pix_rgth]

    incr_rez = np.asarray(np.arange(0, nGrid) * nGrid_min / nGrid, int)  ## indices to then add

    kernel_xal_HM = profile_to_3Dkernel(profile_xal_HM, nGrid, LBox) + stacked_xal_ker[incr_rez, incr_rez, incr_rez]

    return kernel_xal_HM


def stacked_T_kernel(rr_T, T_array, LBox, nGrid, nGrid_min):
    """
    Same as stacked_lyal_kernel but for Temperature profiles.
    rr_T : the comoving radius range
    T_array : the Temp profile (array)
    LBox,nGrid : the box size and grid rez of the current run.
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


def sigmoid_fct(r, c1=2, c2=5):  #
    """
    1D ionization profile, sigmoid function. 1 when ionized, 0 when neutral.

    Parameters
    ----------
    r  : the distance from the source [Mpc].
    c1 : shaprness of the profile (sharp = high c1)
    c2 : the ionization front [Mpc]
    """
    out = 1 - 1 / (1 + np.exp(-c1 * (np.abs(r) - c2)))
    return out

