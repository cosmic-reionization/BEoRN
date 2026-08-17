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
        self._degrade_factor = 1
        self._native_ncell = None

    def _apply_degrade_resolution(self, native_ncell: int) -> None:
        """Validate ``parameters.simulation.degrade_resolution`` and reduce ``Ncell``.

        Call this once a subclass knows the *native* grid size its underlying
        data actually uses (e.g. peeked from a file, as
        :class:`~beorn.load_input_data.nbody_base.NBodyLoader` does, or the
        ``Ncell`` a loader was told to generate/read data at, as
        :class:`~beorn.load_input_data.cosmo_sim_py21cmfast.Py21cmFastLoader`
        does). Sets ``parameters.simulation.Ncell = native_ncell // degrade_resolution``
        so painting/FFT code downstream operates on the reduced grid, and
        stores both the factor (:attr:`_degrade_factor`) and the native size
        (:attr:`_native_ncell`) for the subclass's own use — e.g. a loader
        whose on-disk halo coordinates are indices into the native grid must
        keep using :attr:`_native_ncell` (not the now-reduced
        ``parameters.simulation.Ncell``) to convert them to physical units.

        Must be called with the true native size — constructing a second
        loader against a ``Parameters`` object a prior call already degraded
        would otherwise misinterpret the reduced ``Ncell`` as native.

        Args:
            native_ncell (int): The grid size the underlying data actually
                uses, before any degradation.

        Raises:
            ValueError: If ``degrade_resolution`` is not an integer >= 1, or
                ``native_ncell`` is not evenly divisible by it.
        """
        degrade_resolution = self.parameters.simulation.degrade_resolution
        if not isinstance(degrade_resolution, int) or degrade_resolution < 1:
            raise ValueError(
                f"parameters.simulation.degrade_resolution must be an integer >= 1, "
                f"got {degrade_resolution!r}."
            )
        if native_ncell % degrade_resolution != 0:
            raise ValueError(
                f"Native grid size {native_ncell} is not evenly divisible by "
                f"degrade_resolution={degrade_resolution}."
            )
        self._degrade_factor = degrade_resolution
        self._native_ncell = native_ncell
        effective_n = native_ncell // degrade_resolution
        self.parameters.simulation.Ncell = effective_n
        if degrade_resolution > 1:
            self.logger.info(
                f"Resolution degradation enabled: native {native_ncell}³ data "
                f"block-averaged to {effective_n}³ (factor {degrade_resolution}). "
                f"parameters.simulation.Ncell set to {effective_n}."
            )

    def _coarsen_density_field(self, delta: np.ndarray) -> np.ndarray:
        """Block-average ``delta`` by :attr:`_degrade_factor`, if greater than 1.

        Thin wrapper around :func:`beorn.particle_mapping.coarsen_field` for
        any loader that calls :meth:`_apply_degrade_resolution`.

        Args:
            delta (np.ndarray): 3D field at the native grid resolution.

        Returns:
            numpy.ndarray: ``delta`` unchanged if :attr:`_degrade_factor` is 1,
            otherwise block-averaged down by that factor.
        """
        if self._degrade_factor <= 1:
            return delta
        from ..particle_mapping import coarsen_field
        return coarsen_field(delta, self._degrade_factor)

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
