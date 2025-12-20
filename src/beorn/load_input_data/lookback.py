import numpy as np
from ..structs import Parameters


def get_lookback_range(parameters: Parameters, redshifts: np.ndarray) -> np.ndarray:
    """Compute the lookback redshift range used when assembling mass histories.

    The returned array is in ascending order (current redshift first,
    earlier/past redshifts later) to match the expectations of the
    accretion-fitting routines.

    Args:
        parameters (Parameters): Simulation parameters providing the requested lookback length.
        redshifts (np.ndarray): array of available redshifts in descending order.

    Returns:
        numpy.ndarray: Subset of ``redshifts`` with the requested
        lookback range, ordered current -> past.
    """
    assert np.all(np.diff(redshifts) < 0), "Redshifts must be in descending order."
    # the lookback should cover a fixed time but since the snapshots are spaced in redshift space we need to calculate the range for each redshift
    # the characteristic time for the accretion is determined by the size of the resulting profile
    # For a profile a around 200 comoving Mpc this corresponds to a causal time of about 600 Myr
    lookback_time = 600 # Myr
    # depending on the current redshift, the corresponding number of snapshots is different

    # TODO - hardcoded for now:
    mass_accretion_lookback = parameters.source.mass_accretion_lookback

    # TODO - lookback at early times should be "0" since there are no bubbles yet.
    #     mass_accretion_lookback = parameters.source.mass_accretion_lookback
    lookback_index = max(0, redshifts.size - mass_accretion_lookback)

    redshifts = redshifts[lookback_index:]
    return np.flip(redshifts)

