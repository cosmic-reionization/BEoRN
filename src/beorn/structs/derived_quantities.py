import numpy as np
import tools21cm as t2c
from functools import cached_property

from ..cosmo import dTb_fct
from ..couplings import x_coll, S_alpha
from .. import constants

import logging
logger = logging.getLogger(__name__)


class GridDerivedPropertiesMixin:
    """
    Mixin class to add properties for derived quantities from the grid.
    This is used in the GridData and GridDataMultiZ classes to provide easy access to derived quantities like the brightness temperature and post processed fluctuations.
    """
    loader = None

    def _is_grid_data_multi_z(self) -> bool:
        """
        Check if the current instance is a GridData or GridDataMultiZ instance.
        This is used to ensure that the properties are added correctly.
        """
        # if z is an array that means we are dealing with a multi-z grid and all the fields are 4D: redshift, x, y, z
        # If z is not an array, we are dealing with a single redshift snapshot and all the fields are 3D: x, y, z
        return hasattr(self, "z") and hasattr(self.z, "size") and self.z.size > 1


    def _compute_dTb(self, temp, x_tot, delta_b, x_HII):
        """
        Compute the brightness temperature fluctuations given the base properties of the grid.
        This function is reused multiple times, where the fluctuations of the input field are intentionally set to zero.
        This is a helper function to avoid code duplication in the properties.
        """
        if self._is_grid_data_multi_z():
            z = self.z[:]
        else:
            z = self.z
        grid = dTb_fct(z=z, Tk=temp[:], xtot=x_tot[:], delta_b=delta_b[:], x_HII=x_HII[:], parameters=self.parameters)

        return grid


    @cached_property
    def Grid_dTb(self):
        temp = getattr(self, "Grid_Temp")
        x_tot = self.Grid_xtot
        x_ii = getattr(self, "Grid_xHII")
        delta_b = getattr(self, "delta_b")

        return self._compute_dTb(temp, x_tot, delta_b, x_ii)

    @cached_property
    def Grid_dTb_no_reio(self):
        """
        TODO explain that in this case we assume that x_HII = 0
        """
        temp = getattr(self, "Grid_Temp")
        delta_b = getattr(self, "delta_b")
        x_tot = self.Grid_xtot

        return self._compute_dTb(temp, x_tot, delta_b, 0)

    @cached_property
    def Grid_dTb_T_sat(self):
        """
        TODO explain that in this case we assume that Tk = 1e50
        """
        x_ii = getattr(self, "Grid_xHII")
        delta_b = getattr(self, "delta_b")

        return self._compute_dTb(1e50, 1e50, delta_b, x_ii)

    @cached_property
    def Grid_dTb_RSD(self):
        """
        Use tools21cm to apply Redshift Space Distortion to the dTb field.
        """
        # TODO - differentiate between single-z and multi-z data
        if self.loader is None:
            raise ValueError("Redshift Space Distortion (RSD) require additional information from the loader, but no loader is set.")
        try:
            redshift_index = self.loader.redshift_index(self.z)
            v_x, v_z, v_z = self.loader.load_rsd_fields(redshift_index)
        except NotImplementedError:
            logger.error("Redshift Space Distortion fields are not available for the currently selected input data loader.")
            raise

        rsd = np.array((v_x, v_z, v_z))
        # TODO make sure that the line of sight is the same as the one used later in the spectra
        t2c.conv.LB = self.parameters.simulation.Lbox
        dT_rsd = t2c.get_distorted_dt(self.Grid_dTb, rsd, self.z, los_axis=0, velocity_axis=0, num_particles=20)
        return dT_rsd



    @cached_property
    def coef(self):
        """
        TODO - explain what this coefficient is
        TODO - rename function to something more descriptive
        """
        z = getattr(self, "z")
        if self._is_grid_data_multi_z():
            z = z[:]

        return constants.rhoc0 * self.parameters.cosmology.h ** 2 * self.parameters.cosmology.Ob * (1 + z) ** 3 * constants.M_sun / constants.cm_per_Mpc ** 3 / constants.m_H


    @cached_property
    def Grid_xcoll(self):
        temp = getattr(self, "Grid_Temp")
        x_ii = getattr(self, "Grid_xHII")
        delta_b = getattr(self, "delta_b")
        z = getattr(self, "z")

        if self._is_grid_data_multi_z():
            grid = np.zeros_like(temp)
            for i, z_i in enumerate(z):
                grid[i] = x_coll(z=z_i, Tk = temp[i, ...], xHI = (1 - x_ii[i, ...]), rho_b = (delta_b[i, ...] + 1) * self.coef[i, ...])
        else:
            grid = x_coll(z=z, Tk = temp[:], xHI = (1 - x_ii[:]), rho_b = (delta_b[:] + 1) * self.coef)

        return grid


    @cached_property
    def Grid_xcoll_mean(self):
        # TODO - is this the same as np.mean(self.Grid_xcoll, axis=0)?
        temp = getattr(self, "Grid_Temp")
        x_ii = getattr(self, "Grid_xHII")
        z = getattr(self, "z")

        if self._is_grid_data_multi_z():
            grid = np.zeros_like(temp)
            for i, z_i in enumerate(z):
                grid[i] = x_coll(z=z_i, Tk = np.mean(temp[i, ...]), xHI = (1 - np.mean(x_ii[i, ...])), rho_b = self.coef[i, ...])
        else:
            grid = x_coll(z=z, Tk = np.mean(temp), xHI = (1 - np.mean(x_ii)), rho_b = self.coef)

        return grid


    @cached_property
    def Grid_xtot(self):
        parameters = getattr(self, "parameters")
        grid_xal = getattr(self, "Grid_xal")
        if parameters.simulation.compute_x_coll_fluctuations:
            logger.debug('Including xcoll fluctuations in dTb')
            grid = grid_xal + self.Grid_xcoll

        else:
            logger.debug('NOT including xcoll fluctuations in dTb')
            grid = grid_xal + self.Grid_xcoll_mean

        return grid
