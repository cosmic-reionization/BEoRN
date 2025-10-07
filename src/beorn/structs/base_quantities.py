import numpy as np
from dataclasses import dataclass

import logging
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GridBasePropertiesMixin:
    """
    Mixin class to represent the fundamental properties that each grid snapshot should have.
    """

    delta_b: np.ndarray
    """Density field (delta_b) of the snapshot."""

    Grid_Temp: np.ndarray
    """Temperature field (T) of the snapshot."""

    Grid_xHII: np.ndarray
    """Ionization fraction (xHII) of the snapshot."""

    Grid_xal: np.ndarray
    """Lyman alphy fraction (xal) of the snapshot."""
