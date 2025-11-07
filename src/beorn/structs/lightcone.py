"""Wrapper class for building lightcones using tools21cm"""
import numpy as np
import tools21cm as t2c
from dataclasses import dataclass
from .parameters import Parameters
from .global_profiles import GridDataMultiZ

@dataclass
class Lightcone:
    """Class to handle lightcone data generated from a series of simulation snapshots."""

    data: np.ndarray
    """The lightcone data array containing the desired quantity (e.g., brightness temperature) with spatial and redshift dimensions."""

    redshifts: np.ndarray
    """Array of redshifts corresponding to the slices in the lightcone data."""

    parameters: Parameters
    """The parameters of the simulation."""

    quantity: str
    """The quantity represented in the lightcone (e.g., 'dTb' for brightness temperature)."""


    @classmethod
    def build(cls, parameters: Parameters, grid: GridDataMultiZ, quantity='dTb') -> "Lightcone":
        # prepare the data to a format readable by tools21cm
        try:
            grid_data = getattr(grid, quantity)
        except KeyError:
            raise ValueError(f"Quantity '{quantity}' not found in grid data.")

        # data_dict = {grid.z[i]: grid_data[i, ...] for i in range(len(grid.z))}
        def reading_function(i):
            gd = grid_data[i, ...]
            # cleanup nans TODO - remove
            gd[np.isnan(gd)] = 0.0
            return gd

        scale_factors = 1 / (grid.z[:] + 1)

        lightcone_data, lightcone_redshifts = t2c.make_lightcone(
            filenames = range(grid.z.size),
            # file_redshifts = grid.z[:],
            file_redshifts = scale_factors,
            reading_function = reading_function,
            los_axis = 2,
            raw_density = False,
            box_length_mpc = parameters.simulation.Lbox,
        )
        # assert lightcone_redshifts == grid.z, "Redshifts in lightcone do not match grid redshifts."

        return Lightcone(
            parameters = parameters,
            data = lightcone_data,
            redshifts = lightcone_redshifts,
            quantity = quantity,
        )
