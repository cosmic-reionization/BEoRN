import numpy as np
import logging
logger = logging.getLogger(__name__)


def vectorized_alpha_fit(redshifts: np.ndarray, mass_history: np.ndarray) -> np.ndarray:
    """
    Vectorized fit of exponential mass-accretion parameter alpha.

    Fits the exponent ``alpha`` in the model
    ``M(z) = M0 * exp(alpha * (z0 - z))`` using a least-squares fit on
    the logarithm of the mass history. The implementation is fully
    vectorised over mass histories.

    Args:
        redshifts (np.ndarray): 1D array of redshifts in ascending order (current -> past).
        mass_history (np.ndarray): Array with shape ``(n_objects, n_redshifts)`` containing the mass history for each object.

    Returns:
        np.ndarray: 1D array of fitted ``alpha`` values for each
        object.
    """
    z0 = redshifts[0]
    dz = z0 - redshifts[:]
    assert np.all(dz <= 0), "Redshift values must be ascending (now -> past)."
    assert np.all(mass_history > 0), "Mass history must be non-negative."


    # since mass history is given in order of ascending redshift, the later mass is subtracted from the earlier mass
    # => mass_diff should always be negative
    mass_history_diff = np.diff(mass_history, axis=1)
    logger.debug(f"{np.sum(np.any(mass_history_diff > 0, axis=1))} Negative growth in mass history detected.")

    # since the mass history decreases with lookback, we will need to flip the sign before returning
    y = np.log(mass_history[:, 0][:, np.newaxis]) - np.log(mass_history)

    # Now for each pair of zs and ys we can compute the alpha minimizing the least squares error
    alphas = np.sum(y * dz[np.newaxis, :], axis=1) / np.sum(dz[np.newaxis, :] ** 2, axis=1)

    return -1 * alphas
