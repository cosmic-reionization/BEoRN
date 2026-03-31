from dataclasses import dataclass, fields
import h5py

from .base_struct import BaseStruct
from .base_quantities import GridBasePropertiesMixin
from .derived_quantities import GridDerivedPropertiesMixin

@dataclass(slots = True)
class CoevalCube(BaseStruct, GridBasePropertiesMixin, GridDerivedPropertiesMixin):
    """
    Grid data for a single redshift snapshot. All grid data properties are implemented as base properties and derived properties in mixin classes. They contain the fundamental grids computed during the painting of the simulation as well as derived quantities computed from them.
    """

    z: float
    """Redshift of the snapshot."""


    def to_arrays(self) -> None:
        """Ensure all fields are plain numpy arrays.

        When a :class:`CoevalCube` is loaded from HDF5 its datasets are
        ``h5py.Dataset`` objects which are not picklable across MPI
        processes. This helper converts such datasets to numpy arrays in
        place.
        """
        open_files = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, h5py.Dataset):
                continue

            try:
                open_files[id(value.file)] = value.file
            except Exception:
                pass

            setattr(self, field.name, value[:])

        for file_handle in open_files.values():
            try:
                file_handle.close()
            except Exception:
                pass
