from abc import ABC, abstractmethod
import numpy as np
import logging

from ..structs import Parameters, HaloCatalog

class BaseLoader(ABC):
    """
    Base class for data loaders. This class provides a common interface for loading data and ensures that derived classes implement the required methods.
    Beorn requires the following input data:
    - a halo catalog containing the halo properties (position, mass)
    - a grid containing the associated baryonic density field
    - a grid containing the RSD density field

    The simulation can be run without some of this information but some refined properties will not be available.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, parameters: Parameters):
        """
        Initialize the loader with the given parameters.

        :param parameters: Parameters object containing simulation settings.
        """
        self.parameters = parameters

    @abstractmethod
    def load_halo_catalog(self, redshift_index: int) -> HaloCatalog:
        """
        Load the halo catalog for the redshift index.
        """
        pass

    @abstractmethod
    def load_density_field(self, redshift_index: int) -> np.ndarray:
        """
        Load the baryonic density field for the redshift index.
        """
        pass

    @abstractmethod
    def load_rsd_fields(self, redshift_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the RSD density field for the redshift index. A separate field is loaded for each of the three velocity components (x, y, z).
        """
        pass

    @property
    @abstractmethod
    def redshifts(self) -> np.ndarray:
        """
        Loads the redshifts available to the loader.
        """
        pass

    def redshift_index(self, redshift: float) -> int:
        """
        Returns the index of the given redshift in the loader's redshifts array.
        If the redshift is not found, raises a ValueError.
        """
        indices = np.where(self.redshifts == redshift)[0]
        if indices.size == 0:
            raise ValueError(f"Redshift {redshift} not found in loader's redshifts.")
        return indices[0]
