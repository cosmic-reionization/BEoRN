import numpy as np
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
import logging
logger = logging.getLogger(__name__)

from ..structs.parameters import Parameters


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

    pix_thresh = 80 * (nGrid / 256) ** 3

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
    logger.debug(f'Initial sum of ionized fraction  {round(np.sum(Grid), 3)}')

    if x_ion_tot_i > Grid.size:
        logger.debug('Universe is fully ionized.')
        Grid = np.array([1])

    else:
        logger.info(f'Universe not fully ionized : xHII is {round(x_ion_tot_i / Grid.size, 4)}.')

        region_nbr, size_of_region = np.unique(label_image, return_counts=True)
        logger.debug(f'Found {len(region_nbr)} connected regions.')
        label_max = np.max(label_image)

        small_regions = np.where(np.isin(label_image, region_nbr[np.where(size_of_region < pix_thresh)[
            0]]))  ## small_regions : Gridmesh indices gathering all the connected regions that have less than 10 pixels
        Small_regions_labels = region_nbr[np.where(size_of_region < pix_thresh)[
            0]]  ## labels of the small regions. Use this to exclude them from the for loop

        initial_excess = np.sum(Grid[small_regions] - 1)
        excess_ion = initial_excess

        logger.debug(f'There are {len(Small_regions_labels)} connected regions with less than {pix_thresh} pixels. They contain a fraction {round(excess_ion / x_ion_tot_i, 4)} of the total ionisation fraction.')

        Grid = spread_single(parameters, Grid, small_regions, print_time=None)  # Do the spreading for the small regions
        if np.any(Grid[small_regions] > 1):
            logger.error('small regions not correctly spread')

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
            logger.error('Some grid pixels are still in excess.')

        logger.debug(f'final xion sum: {round(np.sum(Grid), 3)}')
        X_Ion_Tot_f = np.sum(Grid)
        if int(X_Ion_Tot_f) != int(x_ion_tot_i):
            logger.error('Something is wrong when redistributing photons from the overlapping regions. See Spreading_Excess_Fast.')

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

        # if parameters.simulation.subgrid_approximation:  # Is this flag is True, then you set the subgrid size
        if True:  # Is this flag is True, then you set the subgrid size
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

    return Grid
