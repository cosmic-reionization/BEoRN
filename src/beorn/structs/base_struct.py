"""Base data structure for derived data classes containing data required at different stages of the simulation."""
from pathlib import Path
from abc import ABC
from functools import cached_property
from dataclasses import dataclass, fields
import h5py
import numpy as np
import logging
logger = logging.getLogger(__name__)

from .parameters import Parameters, to_dict


# kw_only allows use to specify an optional field even though subclasses have required fields
@dataclass(kw_only=True)
class BaseStruct(ABC):
    """
    Base class for derived data classes containing data required at different stages of the simulation.
    The implementation is such that "loading" this object creates attributes but keeps the actual data on on the disk to be loaded on demand.
    """

    _file_path: Path = None

    parameters: Parameters = None

    @classmethod
    def get_file_path(cls, directory: Path, parameters: Parameters, **kwargs) -> Path:
        """
        Returns the file path for the HDF5 file associated with this object. The file name is generated based on the class name, parameters, and any additional keyword arguments.
        """
        prefix = cls.__name__
        kwargs_string = "_".join([f"{key}={value}" for key, value in kwargs.items()])
        file_name = f"{prefix}_{parameters.unique_hash()}_{kwargs_string}.h5"
        return directory / file_name


    def __post_init__(self):
        """
        Dynamically add attributes to the class for type checking and code completion. All the available hdf5 datasets are now available as attributes
        """
        if self._file_path is not None:
            # Do not use a context manager here, because we want to keep the file open
            hdf5_file = h5py.File(self._file_path, 'r')
            for dataset_name in hdf5_file.keys():
                if dataset_name == "parameters":
                    attribute = Parameters.from_group(hdf5_file[dataset_name])
                elif isinstance(getattr(type(self), dataset_name, None), property):
                    continue  # skip properties, they are not writable
                else:
                    attribute = hdf5_file[dataset_name]

                logger.debug(f"Adding dataset {dataset_name} as attribute to {self.__class__.__name__}")
                setattr(self, dataset_name, attribute)
            for field in hdf5_file.attrs.keys():
                setattr(self, field, hdf5_file.attrs[field])
            logger.debug(f"Read data from {self._file_path}")


    def write(self, file_path: Path = None, directory: Path = None, parameters: Parameters = None, **kwargs):
        """
        Write the content of this dataclass into an HDF5 file. Can be called without any arguments to write to the default file path, or with a specific file path, or with additional keyword arguments to customize the file name.
        """
        if self._file_path:
            logger.debug(f"Using existing file path: {self._file_path=}, ignoring arguments.")
        else:
            if file_path and (directory or parameters or kwargs):
                raise ValueError("Either provide a file path or a directory and parameters, but not both.")
            if file_path is None:
                file_path = self.get_file_path(directory, parameters, **kwargs)
                file_path.parent.mkdir(parents=True, exist_ok=True)

            self._file_path = file_path

        # the fields to write:
        # all attributes of the dataclass
        with h5py.File(self._file_path, 'w') as h5file:
            for field in self._writable_fields():
                value = getattr(self, field)
                self._to_h5_field(h5file, field, value)

        file_size = self._file_path.stat().st_size
        if file_size >= 1024 ** 3:
            human_readable_size = f"{file_size / (1024 ** 3):.2f} GB"
        elif file_size >= 1024 ** 2:
            human_readable_size = f"{file_size / (1024 ** 2):.2f} MB"
        else:
            human_readable_size = f"{file_size / 1024:.2f} KB"

        logger.info(f"Data written to {self._file_path} ({human_readable_size})")


    @classmethod
    def read(cls, file_path: Path = None, directory: Path = None, parameters: Parameters = None, **kwargs):
        """
        Reads the content of a specified HDF5 file and populate the dataclass. Can be called with a specific file path, or with a base directory and parameters to infer the file path.
        """
        if file_path and (directory or parameters or kwargs):
            raise ValueError("Either provide a file path or a directory and parameters, but not both.")
        if file_path is None:
            file_path = cls.get_file_path(directory, parameters, **kwargs)

        # initialize the dataclass with dummy values and the file path
        # -> most values will be populated in the __post_init__ method by reading the HDF5 file
        data = {f.name: None for f in fields(cls)}
        data['parameters'] = parameters
        data['_file_path'] = file_path
        return cls(**data)


    def _writable_fields(self):
        """
        Returns a list of fields that can be written to the HDF5 file. Children of the BaseStruct class will have dataclass fields and properties that can be written.
        """
        writable_fields = []
        for field in fields(self):
            writable_fields.append(field.name)

        if self.parameters is not None:
            for field in self.parameters.simulation.store_grids:
                if field in writable_fields:
                    continue

                # if the field is not a dataclass field, but a property, we can still write it
                if isinstance(getattr(type(self), field, None), property):
                    writable_fields.append(field)
                elif isinstance(getattr(type(self), field, None), cached_property):
                    writable_fields.append(field)
                else:
                    logger.warning(f"Field {field} not found in {self.__class__.__name__}. It will not be written to the HDF5 file.")
        return writable_fields


    def _to_h5_field(self, file, attr: str, value: object):
        """
        Write the content of the dataclass into an HDF5 file.
        """
        # TODO this might fail with ipython auto-reload

        # simple types can be stored as attributes
        if isinstance(value, (int, float, str)):
            file.attrs[attr] = value

        # lists, tuples and arrays can be stored as datasets
        elif isinstance(value, (list, tuple)) or hasattr(value, 'shape'):
            # except if the elements are strings, then we need additional conversion
            value = np.asarray(value)
            if value.dtype.kind == 'U':  # Unicode string
                shape = value.shape
                # convert back to list because h5py complains otherwise
                value = [str(v) for v in value]
                file.create_dataset(attr, shape, data=value, dtype=h5py.string_dtype())
            else:
                file.create_dataset(attr, data=value)

        # finally dataclasses and dictionaries can be stored as groups and subgroups
        elif isinstance(value, Parameters):
            data = to_dict(value)
            self._to_h5_field(file, attr, data)

        elif isinstance(value, dict):
            sub_group = file.create_group(attr)
            for k, v in value.items():
                self._to_h5_field(sub_group, k, v)

        elif isinstance(value, Path):
            file.attrs[attr] = str(value)

        elif value is None:
            # skip None values
            pass

        else:
            raise TypeError(f"Unsupported data type for attribute '{attr}': {type(value)}")
