"""Convenience class for reading and writing objects to a persistence directory. The main benefit of this class is that it calls the read/write methods of BaseStruct classes autonomously."""
from pathlib import Path
import logging
from typing import TypeVar
import shutil

from ..structs import Parameters
from ..structs import BaseStruct

# define a typing variable to represent the fact that the return type of the read method is a subclass of BaseStruct
BaseStructDerived = TypeVar("BaseStructDerived", bound = BaseStruct)


class Handler:
    """
    Convenience class for reading and writing objects to a persistence directory. The main benefit of this class is that it calls the read/write methods of BaseStruct classes autonomously.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, file_root: Path, clear: bool = False, write_kwargs: dict = None):
        """
        Args:
            file_root (Path): The root directory for the persistence files. This directory will be created if it does not exist.
            clear (bool): If True, the directory will be cleared before writing. This is useful for testing purposes.
        """
        self.file_root = file_root
        self.file_root.mkdir(exist_ok = True)
        self.logger.info(f"Using persistence directory at {self.file_root} and kwargs {write_kwargs}")
        self.write_kwargs = write_kwargs if write_kwargs is not None else {}
        if clear:
            self.clear()


    def write_file(self, parameters: Parameters, obj: BaseStructDerived, **kwargs) -> Path:
        """
        Writes the object to the persistence directory. This is a convenience method that replaces the need to call cls.write() directly.
        Args:
            parameters (Parameters): The parameters object that was used to create the instance.
            obj (BaseStructDerived): The object to write.
            **kwargs: Additional keyword arguments that are converted to a string and appended to the file name.
        """
        obj.write(directory=self.file_root, parameters=parameters, **kwargs, **self.write_kwargs)


    def load_file(self, parameters: Parameters, cls: type[BaseStructDerived], **kwargs) -> BaseStructDerived:
        """
        Loads the instance of the class from the persistence directory. This is a convenience method that replaces the need to call cls.read() directly.
        Args:
            parameters (Parameters): The parameters object that was used to create the instance.
            cls (BaseStructDerived): The class of the object to load.
            **kwargs: Additional keyword arguments that are converted to a string and appended to the file name.
        """
        return cls.read(directory=self.file_root, parameters=parameters, **kwargs, **self.write_kwargs)


    def clear(self):
        """
        Clears the persistence directory. This is useful for testing purposes.
        """
        self.logger.info(f"Clearing persistence directory at {self.file_root}")
        shutil.rmtree(self.file_root)
        self.file_root.mkdir()


    def save_logs(self, parameters: Parameters) -> None:
        """
        Registers as a logging handler to save logs to a file in the currently configured file root.
        """
        log_file = f"logs_{parameters.unique_hash()}.log"
        log_path = self.file_root / log_file

        # add a file handler to the global logging config
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(message)s', datefmt='%H:%M:%S')
        )
        logging.getLogger().addHandler(file_handler)
