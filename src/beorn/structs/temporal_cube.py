"""Global description of the 3d data computed over multiple redshifts."""
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
import h5py
import numpy as np
import logging
import tools21cm as t2c
from tqdm.auto import tqdm
logger = logging.getLogger(__name__)

from .base_struct import BaseStruct
from .coeval_cube import CoevalCube
from .base_quantities import GridBasePropertiesMixin
from .derived_quantities import GridDerivedPropertiesMixin
from .parameters import Parameters


@dataclass
class TemporalCube(BaseStruct, GridBasePropertiesMixin, GridDerivedPropertiesMixin):
    """Collection of grid data over multiple redshifts.

    Unlike other :class:`BaseStruct` subclasses, a :class:`TemporalCube` is
    stored as a **directory** of per-redshift HDF5 files rather than a single
    monolithic file:

    .. code-block:: text

        igm_data_{input_tag}_{beorn_hash}/
            CoevalCube_z7.000.h5
            CoevalCube_z7.500.h5
            ...
            CoevalCube_z15.000.h5

    Each ``CoevalCube_z{z:.3f}.h5`` file is a standard :class:`CoevalCube` HDF5 file
    containing ``delta_b``, ``Grid_Temp``, ``Grid_xHII``, and ``Grid_xal``
    for that redshift.  This layout means:

    - Snapshots can be added incrementally without touching existing files.
    - The directory doubles as a resume checkpoint — painting skips redshifts
      whose file already exists.
    - There is no separate ``CoevalCube_*_z_index=N.h5`` cache file.
    """

    z: np.ndarray = None
    """Array of redshifts for which the grid data is available."""

    # ------------------------------------------------------------------ #
    # File-path helpers                                                    #
    # ------------------------------------------------------------------ #

    @classmethod
    def get_file_path(cls, directory: Path, parameters: Parameters, input_tag: str = None, **kwargs) -> Path:
        """Return the output **directory** path ``igm_data_{input_tag}_{beorn_hash}``.

        Args:
            input_tag (str, optional): Human-readable identifier for the upstream
                input data (e.g. ``'py21cmfast_N128_D384_L100_seed12345_9b1f5a85'``).
        """
        beorn_hash = parameters.beorn_hash()
        dir_name = f"igm_data_{input_tag}_{beorn_hash}" if input_tag else f"igm_data_{beorn_hash}"
        return directory / dir_name

    def snapshot_path(self, z: float) -> Path:
        """Return the path for the per-redshift file ``CoevalCube_z{z:.3f}.h5``."""
        return self._file_path / f"CoevalCube_z{z:.3f}.h5"

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def create_empty(cls, parameters: Parameters, directory: Path, snapshot_number: int = None, **kwargs) -> "TemporalCube":
        """Create an empty :class:`TemporalCube` output directory.

        Args:
            parameters (Parameters): Simulation parameters used to build the directory name.
            directory (Path): Parent directory under which the output folder is created.
            snapshot_number: Ignored — kept for API compatibility. Per-z files are
                written on demand so no pre-allocation is needed.
            **kwargs: Forwarded to :meth:`get_file_path` (e.g. ``input_tag``).

        Returns:
            TemporalCube: New instance whose ``_file_path`` points to the created directory.
        """
        path = cls.get_file_path(directory, parameters, **kwargs)
        path.mkdir(parents=True, exist_ok=True)
        ret = cls(z=None, parameters=parameters, delta_b=None, Grid_Temp=None, Grid_xHII=None, Grid_xal=None)
        ret._file_path = path
        return ret

    # ------------------------------------------------------------------ #
    # Per-snapshot I/O                                                     #
    # ------------------------------------------------------------------ #

    def append(self, grid_snapshot: CoevalCube, index: int) -> None:
        """Write a :class:`CoevalCube` snapshot as ``z{z:.3f}.h5`` inside the directory.

        Args:
            grid_snapshot (CoevalCube): Snapshot to write.
            index (int): Unused — kept for API compatibility. The file name is
                derived from ``grid_snapshot.z``.
        """
        if not isinstance(grid_snapshot, CoevalCube):
            raise TypeError("grid_snapshot must be an instance of CoevalCube")
        if self._file_path is None:
            raise ValueError("Output directory is not set. Call create_empty() first.")

        path = self.snapshot_path(grid_snapshot.z)
        grid_snapshot.write(file_path=path)

    # ------------------------------------------------------------------ #
    # Loading                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def read(cls, file_path: Path = None, directory: Path = None, parameters: Parameters = None, **kwargs):
        """Load a :class:`TemporalCube` from its output directory.

        Scans the directory for ``z*.h5`` files (sorted by redshift), loads
        each one, and stacks the fields into 4D ``(n_z, Ncell, Ncell, Ncell)``
        numpy arrays.

        Args:
            file_path (Path): Direct path to the output directory.
            directory / parameters / kwargs: Alternatively, derive the path via
                :meth:`get_file_path`.
        """
        if file_path is not None and (directory or kwargs):
            raise ValueError("Provide either file_path or directory+parameters, not both.")
        if file_path is None:
            file_path = cls.get_file_path(directory, parameters, **kwargs)

        z_files = sorted(file_path.glob("CoevalCube_z*.h5"), key=lambda p: float(p.stem[len("CoevalCube_z"):]))
        if not z_files:
            logger.warning(f"No per-redshift files found in {file_path}. Returning empty TemporalCube.")
            ret = cls(z=None, parameters=parameters, delta_b=None, Grid_Temp=None, Grid_xHII=None, Grid_xal=None)
            ret._file_path = file_path
            return ret

        grid_fields = ['delta_b', 'Grid_Temp', 'Grid_xHII', 'Grid_xal']
        stacks = {name: [] for name in grid_fields}
        z_values = []
        loaded_parameters = parameters

        for path in z_files:
            with h5py.File(path, 'r') as f:
                z_values.append(f.attrs['z'])
                for name in grid_fields:
                    if name in f:
                        stacks[name].append(f[name][:])
                if loaded_parameters is None and 'parameters' in f:
                    loaded_parameters = Parameters.from_group(f['parameters'])

        ret = cls(
            z=np.array(z_values),
            parameters=loaded_parameters,
            delta_b=np.stack(stacks['delta_b'])   if stacks['delta_b']   else None,
            Grid_Temp=np.stack(stacks['Grid_Temp']) if stacks['Grid_Temp'] else None,
            Grid_xHII=np.stack(stacks['Grid_xHII']) if stacks['Grid_xHII'] else None,
            Grid_xal=np.stack(stacks['Grid_xal'])  if stacks['Grid_xal']  else None,
        )
        ret._file_path = file_path
        logger.info(f"Loaded {len(z_files)} snapshots from {file_path}.")
        return ret

    def __post_init__(self):
        # Directory-backed TemporalCubes are loaded explicitly via read().
        # Skip BaseStruct.__post_init__() which expects an HDF5 file.
        pass

    def write(self, **kwargs):
        # TemporalCube has no monolithic file to write; snapshots are written
        # individually via append().  This is a deliberate no-op.
        pass

    # ------------------------------------------------------------------ #
    # Analysis                                                             #
    # ------------------------------------------------------------------ #

    def power_spectrum(self, quantity: np.ndarray, parameters: Parameters) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D power spectra for a given grid quantity over all z.

        Args:
            quantity (np.ndarray): Array shaped (z, nx, ny, nz) to analyse.
            parameters (Parameters): Simulation parameters providing `kbins` and `Lbox`.

        Returns:
            tuple: ``(power_spectrum, bins)`` where ``power_spectrum`` has shape
            (n_z, n_k) and ``bins`` are the k-bin edges.
        """
        bin_number = parameters.simulation.kbins.size
        box_dims = parameters.simulation.Lbox
        power_spectrum = np.zeros((self.z.size, bin_number))

        delta_quantity = quantity[:] / np.mean(quantity, axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis] - 1

        for i, z in enumerate(tqdm(self.z, desc='Power spectrum', unit='snapshot')):
            power_spectrum[i, ...], bins = t2c.power_spectrum.power_spectrum_1d(delta_quantity[i, ...], box_dims=box_dims, kbins=bin_number)

        return power_spectrum, bins


    def redshift_of_reionization(self, ionization_fraction: float = 0.5) -> int:
        """Return the redshift index where the volume-averaged ionization crosses a threshold.

        Args:
            ionization_fraction (float): Threshold volume-averaged ionization fraction. Default 0.5.

        Returns:
            int: Index in the time/redshift array corresponding to the crossing.
        """
        if self.Grid_xHII is None:
            raise ValueError("Grid_xHII is not available.")

        xHII_mean = np.mean(self.Grid_xHII, axis=(1, 2, 3))
        reionization_index = np.argmin(np.abs(xHII_mean - ionization_fraction))
        return reionization_index
