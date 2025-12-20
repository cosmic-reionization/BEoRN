from abc import ABC, abstractmethod
import numpy as np
import logging

from ..structs import Parameters, HaloCatalog

class BaseLoader(ABC):
    """Abstract base class for data loaders.

    Subclasses must implement methods to provide the following data:
    - halo catalogs containing all relevant halo properties,
    - baryonic density fields,
    - redshift-space-distortion (RSD) (optional)

    Implementations are expected to expose a ``redshifts`` property describing available snapshots.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, parameters: Parameters):
        """Initialize the loader with simulation ``parameters``.

        Args:
            parameters (Parameters): Parameters object containing simulation settings.
        """
        self.parameters = parameters

    @abstractmethod
    def load_halo_catalog(self, redshift_index: int) -> HaloCatalog:
        """Load the halo catalog for a given snapshot index.

        Args:
            redshift_index (int): Snapshot index to load.

        Returns:
            HaloCatalog: Loaded halo catalog for the snapshot.
        """
        pass

    @abstractmethod
    def load_density_field(self, redshift_index: int) -> np.ndarray:
        """Load the baryonic density field for a given snapshot.

        Args:
            redshift_index (int): Snapshot index to load.

        Returns:
            numpy.ndarray: 3D density field array (shape ``Ncell^3``).
        """
        pass

    @abstractmethod
    def load_rsd_fields(self, redshift_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load the three RSD (velocity-weighted) fields for the snapshot.

        Args:
            redshift_index (int): Snapshot index to load.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The three
            velocity-component meshes (vx, vy, vz) mapped to the grid.
        """
        pass

    @property
    @abstractmethod
    def redshifts(self) -> np.ndarray:
        """Array of available redshifts for this loader.

        Returns:
            numpy.ndarray: 1D array of redshift values (ascending order: current->past).
        """
        pass

    def redshift_index(self, redshift: float) -> int:
        """Return the index of ``redshift`` in the loader's grid.

        Args:
            redshift (float): Redshift value to look up.

        Returns:
            int: Index into :pyattr:`redshifts` corresponding to ``redshift``.

        Raises:
            ValueError: If the requested redshift is not available.
        """
        indices = np.where(self.redshifts == redshift)[0]
        if indices.size == 0:
            raise ValueError(f"Redshift {redshift} not found in loader's redshifts.")
        return indices[0]
