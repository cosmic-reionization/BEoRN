import numpy as np
import logging
logger = logging.getLogger(__name__)

def bin_centers(bins: np.ndarray) -> np.ndarray:
    """
    Compute the bin centers from the bin edges. Automatically detects whether bins are linear or logarithmic.

    Parameters
    ----------
    bins : np.ndarray
        1D array of bin edges.

    Returns
    -------
    np.ndarray
        1D array of bin centers with n-1 elements, where n is the number of bin edges.
    """
    if np.all(np.diff(bins) == bins[1] - bins[0]):
        # Linear bins
        return 0.5 * (bins[:-1] + bins[1:])
    else:
        logger.debug("Logarithmic bins detected.")
        return np.sqrt(bins[:-1] * bins[1:])
