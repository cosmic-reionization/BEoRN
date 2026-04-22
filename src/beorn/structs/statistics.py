"""Statistical estimators for :class:`~beorn.structs.TemporalCube` outputs.

Usage::

    stats = StatisticsEstimator(multi_z_quantities, parameters)
    stats.save()                    # writes igm_stats_<tag>.h5 next to the cube dir
    stats.save('my_stats.h5')       # explicit path, or format='pickle'

    # reuse without re-computing (auto-loads from default path if file exists)
    stats2 = StatisticsEstimator(multi_z_quantities, parameters)
    results = stats2.results        # loads from disk; falls back to compute()

    # load raw arrays without a cube object
    data = StatisticsEstimator.load('igm_stats_<tag>.h5')

Extend by subclassing and appending to ``_estimators``::

    class MyStats(StatisticsEstimator):
        _estimators = StatisticsEstimator._estimators + ['variance_signals']

        def variance_signals(self) -> dict:
            return {'var_dTb': np.var(self.cube.Grid_dTb[:], axis=(1, 2, 3))}
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .temporal_cube import TemporalCube
    from .parameters import Parameters


class StatisticsEstimator:
    """Compute and persist summary statistics from a :class:`~beorn.structs.TemporalCube`.

    Each entry in ``_estimators`` names a method on this class.  ``compute()``
    calls them in order and merges their ``dict`` returns into a single results
    dict.  Subclasses add new estimators by extending the list and implementing
    the corresponding method.

    ``results`` is a lazy property: on first access it tries to load from
    :meth:`default_path`; if no file exists there it calls :meth:`compute`.

    Args:
        cube: Multi-redshift simulation output.
        parameters: Simulation parameters (needed for k-bins, box size, etc.).
    """

    _estimators: list[str] = ['global_signals', 'power_spectra']

    def __init__(self, cube: TemporalCube | None = None, parameters: Parameters | None = None) -> None:
        self.cube = cube
        self.parameters = parameters
        self._results: dict | None = None

    @classmethod
    def from_file(cls, path: str | Path, format: str = 'hdf5') -> 'StatisticsEstimator':
        """Load a previously saved statistics file without a TemporalCube.

        The returned instance has ``cube=None`` and its :attr:`results` are
        pre-populated from disk.  It can be passed directly to the
        ``draw_*`` plotting functions.

        Args:
            path: File written by :meth:`save`.
            format: ``'hdf5'`` or ``'pickle'``.  Inferred from extension when omitted.

        Returns:
            StatisticsEstimator: Instance with results loaded from *path*.

        Example::

            stats = MyStats.from_file('output/igm_stats_py21cmfast_N128_….h5')
            beorn.plotting.draw_dTb_signal(ax, stats, color='k')
        """
        inst = cls.__new__(cls)
        inst.cube = None
        inst.parameters = None
        inst._results = cls.load(path, format=format)
        return inst

    # ------------------------------------------------------------------
    # Default file path
    # ------------------------------------------------------------------

    def default_path(self, format: str = 'hdf5') -> Path | None:
        """Return the default statistics file path next to the cube directory.

        The name is derived from the cube directory by replacing the
        ``igm_data_`` prefix with ``igm_stats_`` and appending ``.h5`` or
        ``.pkl``.  Returns ``None`` when the cube has no ``_file_path``.

        Example:
            cube dir  → ``output/igm_data_py21cmfast_N128_…_4024dbb5``
            stats file → ``output/igm_stats_py21cmfast_N128_…_4024dbb5.h5``
        """
        cube_path = getattr(self.cube, '_file_path', None)
        if cube_path is None:
            return None
        stem = Path(cube_path).name
        stats_stem = stem.replace('igm_data_', 'igm_stats_', 1) if stem.startswith('igm_data_') else f'igm_stats_{stem}'
        ext = '.pkl' if format == 'pickle' else '.h5'
        return Path(cube_path).parent / (stats_stem + ext)

    # ------------------------------------------------------------------
    # Lazy results property
    # ------------------------------------------------------------------

    @property
    def results(self) -> dict:
        """Cached statistics dict.  Loads from :meth:`default_path` if the file
        exists; otherwise calls :meth:`compute` and caches the result."""
        if self._results is None:
            path = self.default_path()
            if path is not None and path.exists():
                self._results = StatisticsEstimator.load(path)
            else:
                self._results = self.compute()
        return self._results

    # ------------------------------------------------------------------
    # Built-in estimators
    # ------------------------------------------------------------------

    def global_signals(self) -> dict:
        """Global (volume-averaged) signal histories.

        Returns:
            dict with keys ``z``, ``mean_dTb``, ``mean_Temp``, ``mean_xHII``, ``mean_xal``.
        """
        return {
            'z':         self.cube.z[:],
            'mean_dTb':  self.cube.global_mean('dTb'),
            'mean_Temp': self.cube.global_mean('Temp'),
            'mean_xHII': self.cube.global_mean('xHII'),
            'mean_xal':  self.cube.global_mean('xal'),
        }

    def power_spectra(self) -> dict:
        """Dimensionless 21-cm power spectrum :math:`\\Delta^2_{21}(k, z)`.

        Returns:
            dict with keys ``k`` (shape ``(n_k,)``) and
            ``ps_dTb`` (shape ``(n_z, n_k)``).
        """
        ps, k = self.cube.power_spectrum(self.cube.Grid_dTb, self.parameters)
        mean_dTb = self.cube.global_mean('dTb')
        delta2 = ps * k[np.newaxis, :] ** 3 * mean_dTb[:, np.newaxis] ** 2 / (2 * np.pi ** 2)
        return {
            'k':      k,
            'ps_dTb': delta2,
        }

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def compute(self) -> dict:
        """Run all registered estimators and return a merged results dict."""
        results: dict = {}
        for name in self._estimators:
            results.update(getattr(self, name)())
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None, format: str = 'hdf5') -> Path:
        """Compute statistics and write them to *path*.

        If *path* is omitted, :meth:`default_path` is used.  The computed
        results are also cached in :attr:`_results` so a subsequent call to
        :attr:`results` does not re-compute.

        Args:
            path: Output file path, or ``None`` to use the default.
            format: ``'hdf5'`` (default) or ``'pickle'``.

        Returns:
            Path: The file that was written.

        Raises:
            ValueError: If *path* is ``None`` and the cube has no ``_file_path``.
        """
        if path is None:
            path = self.default_path(format=format)
            if path is None:
                raise ValueError(
                    "Cannot derive a default path because the cube has no _file_path. "
                    "Pass an explicit path to save()."
                )
        path = Path(path)
        results = self.compute()
        self._results = results
        if format == 'hdf5':
            import h5py
            with h5py.File(path, 'w') as f:
                for key, val in results.items():
                    f.create_dataset(key, data=np.asarray(val))
        elif format == 'pickle':
            with open(path, 'wb') as fh:
                pickle.dump(results, fh)
        else:
            raise ValueError(f"Unknown format '{format}'. Supported: 'hdf5', 'pickle'.")
        return path

    @staticmethod
    def load(path: str | Path, format: str = 'hdf5') -> dict:
        """Load previously saved statistics from *path*.

        The format is inferred from the file extension when *format* is not
        given explicitly: ``.h5`` / ``.hdf5`` → ``'hdf5'``; ``.pkl`` /
        ``.pickle`` → ``'pickle'``.

        Args:
            path: File written by :meth:`save`.
            format: ``'hdf5'`` or ``'pickle'``.  Inferred from extension when omitted.

        Returns:
            dict mapping statistic names to numpy arrays.
        """
        path = Path(path)
        if path.suffix in ('.pkl', '.pickle'):
            format = 'pickle'
        elif path.suffix in ('.h5', '.hdf5'):
            format = 'hdf5'
        if format == 'hdf5':
            import h5py
            with h5py.File(path, 'r') as f:
                return {key: f[key][:] for key in f}
        elif format == 'pickle':
            with open(path, 'rb') as fh:
                return pickle.load(fh)
        else:
            raise ValueError(f"Unknown format '{format}'. Supported: 'hdf5', 'pickle'.")
