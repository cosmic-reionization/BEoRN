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

    Note:
        The uniform-spacing test is ``np.allclose``, not exact equality.  Exact
        equality is never satisfied by ``np.linspace`` output for more than two
        edges (e.g. ``np.diff(np.linspace(0, 5, 26))`` is 0.2 only to within
        float rounding), which used to send every uniform grid down the
        logarithmic branch — putting the first alpha bin centre at
        ``sqrt(0 * 0.2) = 0`` and the second at 0.283 instead of 0.1 and 0.3.
        Log-spaced grids are unaffected: their successive differences span many
        orders of magnitude, so they remain far from ``allclose``.
    """
    spacings = np.diff(bins)
    if np.allclose(spacings, spacings[0]):
        # Linear bins
        return 0.5 * (bins[:-1] + bins[1:])
    else:
        logger.debug("Logarithmic bins detected.")
        return np.sqrt(bins[:-1] * bins[1:])
