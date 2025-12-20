import numpy as np
import logging
logger = logging.getLogger(__name__)

def bin_centers(bins: np.ndarray) -> np.ndarray:
    """Compute bin centers from bin edges.

    Automatically detects whether the provided ``bins`` are linear or
    logarithmic and returns the appropriate center values.

    Args:
        bins (numpy.ndarray): 1D array of bin edges.

    Returns:
        numpy.ndarray: 1D array of bin centers with length ``len(bins)-1``.
    """
    if np.all(np.diff(bins) == bins[1] - bins[0]):
        # Linear bins
        return 0.5 * (bins[:-1] + bins[1:])
    else:
        logger.debug("Logarithmic bins detected.")
        return np.sqrt(bins[:-1] * bins[1:])
