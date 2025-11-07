from dataclasses import dataclass
import numpy as np

from .base_struct import BaseStruct
from .base_quantities import GridBasePropertiesMixin
from .derived_quantities import GridDerivedPropertiesMixin

@dataclass(slots = True)
class GridData(BaseStruct, GridBasePropertiesMixin, GridDerivedPropertiesMixin):
    """
    Grid data for a single redshift snapshot. All grid data properties are implemented as base properties and derived properties in mixin classes. They contain the fundamental grids computed during the painting of the simulation as well as derived quantities computed from them.
    """

    z: float
    """Redshift of the snapshot."""
