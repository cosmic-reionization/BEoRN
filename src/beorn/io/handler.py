"""Convenience helpers for reading and writing project objects to disk.

This module provides :class:`Handler`, a thin wrapper around the
``BaseStruct`` read/write API that centralises the file-root used for
persistence and exposes helper methods for loading, saving and clearing
the persistence directory.
"""
from pathlib import Path
import logging
from typing import TypeVar
import shutil

from ..structs import Parameters
from ..structs.base_struct import BaseStruct

# define a typing variable to represent the fact that the return type of the read method is a subclass of BaseStruct
BaseStructDerived = TypeVar("BaseStructDerived", bound = BaseStruct)


class Handler:
    """Manage a persistence directory and delegate read/write calls.

    The handler wraps the read/write methods implemented by classes
    deriving from :class:`beorn.structs.base_struct.BaseStruct`, storing
    a common ``file_root`` directory and optional default
    ``write_kwargs`` that are passed to write operations.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, file_root: Path, input_tag: str = None, clear: bool = False, write_kwargs: dict = None):
        """Create a new :class:`Handler` bound to ``file_root``.

        Args:
            file_root (Path): Root directory for persistence. The directory will be created if it does not already exist.
            input_tag (str, optional): Human-readable identifier for the upstream
                input data (e.g. ``loader.input_tag``).  When set it is embedded
                in the output filename so that the provenance of each
                ``igm_cubes_*.h5`` file is immediately visible on disk.
            clear (bool, optional): If True the directory will be
                removed and recreated. Useful for tests or clean runs.
                Defaults to ``False``.
            write_kwargs (dict, optional): Default keyword arguments to
                include on write calls. If provided these are merged
                into each call to :meth:`write_file`.
        """
        self.file_root = Path(file_root)
        self.file_root.mkdir(exist_ok=True)
        self.write_kwargs = write_kwargs if write_kwargs is not None else {}
        if input_tag is not None:
            self.write_kwargs['input_tag'] = input_tag
        self.logger.info(f"Using persistence directory at {self.file_root}")
        if clear:
            self.clear()


    def write_file(self, parameters: Parameters, obj: BaseStructDerived, **kwargs) -> None:
        """Write ``obj`` to the handler's persistence directory.

        This convenience wrapper calls the :meth:`BaseStruct.write`
        implementation of ``obj`` with the configured ``file_root`` and
        merges any provided ``kwargs`` with the handler's
        ``write_kwargs``.

        Args:
            parameters (Parameters): Parameters instance used to create or uniquely identify the object.
            obj (BaseStructDerived): Instance providing a ``write`` method.
            **kwargs: Additional keyword arguments forwarded to
                ``obj.write``. These are often used to distinguish file
                names or control writing behaviour.

        Returns:
            None
        """
        obj.write(directory=self.file_root, parameters=parameters, **kwargs, **self.write_kwargs)


    def load_file(self, parameters: Parameters, cls: type[BaseStructDerived], **kwargs) -> BaseStructDerived:
        """Load an instance of ``cls`` from the persistence directory.

        This convenience wrapper calls :meth:`BaseStruct.read` on the
        provided class with the handler's ``file_root`` and returns the
        instantiated object.

        Args:
            parameters (Parameters): Parameters instance used to identify the file to load.
            cls (type[BaseStructDerived]): Class implementing ``read`` that returns an instance of :class:`BaseStruct`.
            **kwargs: Additional keyword arguments forwarded to``cls.read``.

        Returns:
            BaseStructDerived: Loaded instance of ``cls``.
        """
        return cls.read(directory=self.file_root, parameters=parameters, **kwargs, **self.write_kwargs)


    def clear(self):
        """Remove and recreate the handler's persistence directory.

        This deletes all files under ``file_root``. Use with caution.

        Returns:
            None
        """
        self.logger.info(f"Clearing persistence directory at {self.file_root}")
        shutil.rmtree(self.file_root)
        self.file_root.mkdir()


    def save_logs(self, parameters: Parameters) -> None:
        """Configure a file handler to save application logs.

        The log file is named to match the output ``igm_cubes_*.h5`` file,
        using the same ``input_tag`` and ``beorn_hash`` so logs and output
        sit side-by-side and are trivially associated.

        Args:
            parameters (Parameters): Parameters object used to derive the log file name.

        Returns:
            None
        """
        input_tag = self.write_kwargs.get('input_tag')
        beorn_hash = parameters.beorn_hash()
        log_stem = f"logs_{input_tag}_{beorn_hash}" if input_tag else f"logs_{beorn_hash}"
        log_path = self.file_root / f"{log_stem}.log"

        # add a file handler to the global logging config
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(message)s', datefmt='%H:%M:%S')
        )
        logging.getLogger().addHandler(file_handler)
