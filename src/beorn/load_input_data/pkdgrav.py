import numpy as np

from .base import BaseLoader
from ..structs import HaloCatalog



class PKDGravLoader(BaseLoader):
    """
    Loader for the PKDGrav data format.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_root = self.parameters.simulation.file_root


        logs = np.loadtxt(self.file_root / "CDM_200Mpc_2048.log")
        catalogs = (self.file_root / "haloes").glob("*.fof.txt")
        density_paths = (self.file_root / "grids" / "nc256").glob("*.den.256.0")

        # get the available redshift_range
        redshifts = logs[:, 1]
        # The correspondence is now given by:
        # redshift -> CDM_200Mpc_2048.<index of that redshift>.fof.txt

        # sort the snapshots
        catalogs = sorted(catalogs, key=lambda x: str(x))
        density_paths = sorted(density_paths, key=lambda x: str(x))

        # There is a mismatch between the logs and the saved snapshots - the last ~20 snapshots were not saved (too large)

        # Now reduce the redshift range to the one requested
        indices = np.where((redshifts >= self.parameters.solver.redshifts[0]) & (redshifts <= self.parameters.solver.redshifts[-1]))[0]
        self._redshifts = np.array(redshifts[indices])
        self.logger.debug(f"Reducing available redshift range to the restriction imposed by the {len(self._redshifts)} snapshots.")
        self.catalogs = [catalogs[i] for i in indices]
        self.density_paths = [density_paths[i] for i in indices]

        self.remove_duplicates()

        self.logger.info(f"Initialized PKDGrav data loader - reading files from {self.file_root}.")


    @property
    def redshifts(self):
        return self._redshifts


    def remove_duplicates(self) -> None:
        # The particular PKDGrav data we have has some duplicate redshifts due to restarted transfers - we want to discard them
        z_initial = np.inf
        redshifts_copy = []
        catalogs_copy = []
        density_paths_copy = []

        for i in range(self.redshifts.size):
            if self.redshifts[i] >= z_initial:
                self.logger.debug(f"Removing {i}th snapshot from the list")
            else:
                # keep the snapshot
                redshifts_copy.append(self.redshifts[i])
                catalogs_copy.append(self.catalogs[i])
                density_paths_copy.append(self.density_paths[i])
                z_initial = self.redshifts[i]

        self._redshifts = np.array(redshifts_copy)
        self.catalogs = catalogs_copy
        self.density_paths = density_paths_copy

        self.logger.debug(f"After removing duplicates, {self.redshifts.size} snapshots remain - range: {self.redshifts[0]} to {self.redshifts[-1]}")


    def load_halo_catalog(self, redshift_index: int) -> HaloCatalog:
        catalog_path = self.catalogs[redshift_index]
        catalog_array = np.loadtxt(catalog_path)
        if catalog_array.shape == (0,):
            catalog_array = np.ndarray((0, 4))

        return HaloCatalog(
            # masses should be in Msun
            masses = catalog_array[:, 0] * self.parameters.cosmology.h,
            # shift to center the box
            positions = catalog_array[:, 1:] + self.parameters.simulation.Lbox / 2,
            parameters = self.parameters,
        )


    def load_density_field(self, redshift_index: int) -> np.ndarray:
        Ncell = self.parameters.simulation.Ncell

        density_path = self.density_paths[redshift_index]
        density_array = np.fromfile(density_path, dtype=np.float32)

        assert Ncell**3 == density_array.size, f"Density file dimension {density_array.size} does not match simulation {Ncell**3 = }"

        # Put into the right shape
        density = density_array.reshape(Ncell, Ncell, Ncell)
        density = density.T
        # V_total = LBox ** 3
        # V_cell = (LBox / nGrid) ** 3
        # mass  = (pkd * rhoc0 * V_total).astype(np.float64)
        # rho_m = mass / V_cell
        # delta_b = (rho_m) / np.mean(rho_m, dtype=np.float64) - 1
        # since we divide by the mean, we can skip the mass calculation
        delta_b = density / np.mean(density, dtype=np.float64) - 1
        return delta_b


    def load_rsd_fields(self, redshift_index: int):
        raise NotImplementedError("RSD fields are not implemented for PKDGrav data format")
