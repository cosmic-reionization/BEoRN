import numpy as np
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
import logging
logger = logging.getLogger(__name__)

from ..structs.parameters import Parameters


def _box_slice(center: int, half: int, n: int) -> slice:
    """Return a 1-D slice for a sub-box of width 2*half centered at center within [0, n).

    Near-boundary regions are handled by shifting the window so it stays within
    the grid (same behaviour as the original max/min expression).
    """
    lo = max(center - half, 0) - max(0, center + half - n)
    hi = min(n, center + half) + max(0, half - center)
    return slice(lo, hi)


def _fill_excess_shells(grid_flat: np.ndarray, dist_flat: np.ndarray,
                        excess_ion: float, sentinel: float) -> None:
    """Distribute excess_ion into cells ordered by distance from the ionized boundary.

    Replaces the while-loop-over-shells (which calls np.where and np.min on the
    full grid each iteration) with a single argsort pass followed by searchsorted.
    Complexity: O(M log M) where M = number of valid cells, versus O(n_shells * M).

    Cells with dist_flat >= sentinel are excluded (ionized interior or already-excess
    cells from other regions). grid_flat is modified in-place.

    Args:
        grid_flat (np.ndarray): Flattened ionization grid (view of the working grid).
        dist_flat (np.ndarray): Flattened distance-transform array (same shape).
        excess_ion (float): Total excess ionization to redistribute.
        sentinel (float): Distance value used to mark excluded cells.
    """
    valid_mask = dist_flat < sentinel
    if not np.any(valid_mask):
        return

    v_idx  = np.where(valid_mask)[0]
    v_dist = dist_flat[v_idx]
    v_cap  = 1.0 - grid_flat[v_idx]          # available capacity per cell

    # Sort by distance — nearest shell first; stable so equal-distance cells stay grouped
    order  = np.argsort(v_dist, kind='stable')
    s_dist = v_dist[order]
    s_cap  = v_cap[order]
    s_orig = v_idx[order]                     # indices back into grid_flat

    # Shell boundaries: positions where the distance value changes
    shell_edges = np.concatenate([[0], np.flatnonzero(np.diff(s_dist)) + 1, [len(s_dist)]])

    # Total capacity per shell, cumulated
    shell_caps   = np.add.reduceat(s_cap, shell_edges[:-1])
    shell_cumcap = np.cumsum(shell_caps)

    # First shell whose running total meets or exceeds excess_ion
    cutoff_shell = int(np.searchsorted(shell_cumcap, excess_ion, side='left'))

    # Fill all shells before the cutoff to 1
    if cutoff_shell > 0:
        grid_flat[s_orig[:shell_edges[cutoff_shell]]] = 1.0

    # Partial fill of the cutoff shell, proportional to each cell's available capacity
    if cutoff_shell < len(shell_caps) and shell_caps[cutoff_shell] > 0:
        s0, s1 = shell_edges[cutoff_shell], shell_edges[cutoff_shell + 1]
        cumcap_before = shell_cumcap[cutoff_shell - 1] if cutoff_shell > 0 else 0.0
        remaining = excess_ion - cumcap_before
        frac = remaining / shell_caps[cutoff_shell]
        grid_flat[s_orig[s0:s1]] += s_cap[s0:s1] * frac


def spreading_excess_fast(parameters: Parameters, Grid_input, print_time=False):
    """Redistribute excess ionization from overlapping regions.

    The input grid may contain cells with ionization fraction greater
    than unity due to overlapping ionized bubbles. This function
    identifies connected ionized regions, separates small regions from
    large ones, and spreads the excess ionizing photons into nearby
    neutral volume using an efficient local distance-transform
    approach.

    Args:
        parameters (Parameters): Simulation parameters used to define
            thresholds and approximations.
        Grid_input (numpy.ndarray): 3D ionization fraction grid (values
            may be >1 where overlaps occur).
        print_time (bool, optional): If True, print simple progress
            updates for large-region processing. Defaults to False.

    Returns:
        numpy.ndarray: Grid with redistributed ionization (no values >1
        should remain).
    """
    nGrid = len(Grid_input[0])
    Grid = np.copy(Grid_input)

    pix_thresh = 80 * (nGrid / 256) ** 3

    Binary_Grid = (Grid >= 0.9999999).astype(np.int32)
    label_image = label(Binary_Grid)

    x_ion_tot_i = np.sum(Grid)
    logger.debug(f'Initial sum of ionized fraction  {round(np.sum(Grid), 3)}')

    if x_ion_tot_i > Grid.size:
        logger.debug('Universe is fully ionized.')
        return np.ones_like(Grid)

    logger.info(f'Universe not fully ionized : xHII is {round(x_ion_tot_i / Grid.size, 4)}.')

    region_nbr, size_of_region = np.unique(label_image, return_counts=True)
    logger.debug(f'Found {len(region_nbr)} connected regions.')
    label_max = np.max(label_image)

    small_mask = size_of_region < pix_thresh
    Small_regions_labels = region_nbr[small_mask]

    small_regions = np.where(np.isin(label_image, Small_regions_labels))
    logger.debug(
        f'There are {len(Small_regions_labels)} connected regions with less than {pix_thresh} pixels. '
        f'They contain a fraction {round(np.sum(Grid[small_regions] - 1) / x_ion_tot_i, 4)} of the total ionisation fraction.'
    )

    Grid = spread_single(parameters, Grid, small_regions)
    if np.any(Grid[small_regions] > 1):
        logger.error('Small regions not correctly spread')

    all_regions_labels = np.arange(1, label_max + 1)
    large_regions_labels = all_regions_labels[~np.isin(all_regions_labels, Small_regions_labels)]

    for i, ir in enumerate(large_regions_labels):
        if print_time and i % 100 == 0:
            print(f'Doing region {i} over {len(large_regions_labels)} regions in total')
        connected_indices = np.where(label_image == ir)
        Grid = spread_single(parameters, Grid, connected_indices)

    if np.any(Grid > 1.):
        logger.error('Some grid pixels are still in excess.')

    logger.debug(f'final xion sum: {round(np.sum(Grid), 3)}')
    X_Ion_Tot_f = np.sum(Grid)
    if int(X_Ion_Tot_f) != int(x_ion_tot_i):
        logger.error('Something is wrong when redistributing photons from the overlapping regions. See Spreading_Excess_Fast.')

    return Grid


def spread_single(parameters: Parameters, Grid, connected_indices):
    """Spread excess ionizing photons for a single connected region.

    This routine redistributes the excess ionizing fraction from cells
    inside ``connected_indices`` into surrounding neutral cells. To
    improve performance it works on a local sub-box around the region
    (distance-transform approach) when possible.

    Args:
        parameters (Parameters): Simulation parameters, used to set
            thresholds such as the subgrid approximation and pixel
            thresholds.
        Grid (numpy.ndarray): 3D array with ionization fractions.
        connected_indices (tuple): Tuple of index arrays (as returned
            by ``np.where(label_image == label)``) selecting the
            connected region to process.

    Returns:
        numpy.ndarray: The input ``Grid`` with the excess redistributed
        around the provided connected region.
    """
    nGrid = len(Grid[0])

    initial_excess = float(np.sum(Grid[connected_indices] - 1))
    Grid[connected_indices] = np.where(Grid[connected_indices] > 1, 1, Grid[connected_indices])

    if initial_excess <= 1e-8:
        return Grid

    # Binary mask: 1 = neutral (can receive photons), 0 = ionized region interior
    Inverted_grid = np.ones((nGrid, nGrid, nGrid), dtype=np.float64)
    Inverted_grid[connected_indices] = 0

    Min_X = int(np.min(connected_indices[0]))
    Max_X = int(np.max(connected_indices[0]))
    Min_Y = int(np.min(connected_indices[1]))
    Max_Y = int(np.max(connected_indices[1]))
    Min_Z = int(np.min(connected_indices[2]))
    Max_Z = int(np.max(connected_indices[2]))

    Delta_max  = max(Max_X - Min_X, Max_Y - Min_Y, Max_Z - Min_Z)
    Center_X   = (Min_X + Max_X) // 2
    Center_Y   = (Min_Y + Max_Y) // 2
    Center_Z   = (Min_Z + Max_Z) // 2
    Delta_pixel = int(initial_excess ** (1.0 / 3) / 2) + 1

    half = Delta_max + 2 * Delta_pixel   # half-width of initial sub-box
    if (2 * half) % 2 == 1:
        half += 1

    if 2 * half > nGrid:
        # Full-grid path: sub-box is larger than the box itself
        dist = distance_transform_edt(Inverted_grid)
        dist[connected_indices] = 2 * nGrid      # exclude region interior
        dist[Grid > 1]          = 2 * nGrid      # exclude other excess cells
        sentinel = 2 * nGrid
        _fill_excess_shells(Grid.ravel(), dist.ravel(), initial_excess, sentinel)

    else:
        # Sub-grid path: work on a local sub-box, growing it if needed
        while True:
            sx = _box_slice(Center_X, half, nGrid)
            sy = _box_slice(Center_Y, half, nGrid)
            sz = _box_slice(Center_Z, half, nGrid)
            Sub_Grid = Grid[sx, sy, sz]
            if np.sum(1.0 - Sub_Grid) >= initial_excess:
                break
            half += 1

        Sub_Grid     = Grid[sx, sy, sz].copy()
        Sub_Inv      = Inverted_grid[sx, sy, sz]

        N_sub = Sub_Grid.shape[0]
        sentinel = 2 * N_sub

        dist = distance_transform_edt(Sub_Inv)
        dist[Sub_Inv == 0]  = sentinel     # exclude region interior
        dist[Sub_Grid >= 1] = sentinel     # exclude cells already at capacity

        _fill_excess_shells(Sub_Grid.ravel(), dist.ravel(), initial_excess, sentinel)

        Grid[sx, sy, sz] = Sub_Grid

    return Grid
