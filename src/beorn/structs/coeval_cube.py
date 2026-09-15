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

        ``BaseStruct.__post_init__`` now reads every dataset eagerly and closes the file, so a
        :class:`CoevalCube` loaded from HDF5 already holds numpy values and this is a no-op for
        it. It is kept for callers that assign ``h5py.Dataset`` objects to fields themselves
        (those are not picklable across MPI processes) and converts any such field in place
        (review_2026-09-14 Phase 5).
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

            if value.shape == ():
                setattr(self, field.name, value[()])
            else:
                setattr(self, field.name, value[:])

        for file_handle in open_files.values():
            try:
                file_handle.close()
            except Exception:
                pass
