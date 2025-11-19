try:
    import py21cmfast as p21c
except:
    raise ImportError('To use the density information and halo catalogs from 21cmfast, beorn requires additional dependencies. Either install beorn with extended package set: `pip install beorn[extra]` or see https://github.com/21cmfast/21cmFAST/.')

import time
import h5py
import logging
logger = logging.getLogger(__name__)

from .cosmo import *
from .structs.parameters import Parameters
from .io import Handler



def generate_haloes_and_density(
    parameters: Parameters,
    handler: Handler = None,
):
    """Generate halo catalogs and density fields using 21cmFast."""
    start_time = time.process_time()
    logger.info('Simulating matter evolution with 21cmFast')

    # Convert the parameters to the format required by py21cmfast
    user_params = p21c.UserParams(
        HII_DIM = parameters.simulation.Ncell,
        DIM = parameters.simulation.Ncell * 3,
        BOX_LEN = parameters.simulation.Lbox / parameters.cosmology.h,
        USE_INTERPOLATION_TABLES = True,
        # FIXED_IC = True,
        N_THREADS = parameters.simulation.cores,
    )

    cosmo_params = p21c.CosmoParams(
        SIGMA_8 = parameters.cosmology.sigma_8,
        hlittle = parameters.cosmology.h,
        OMm = parameters.cosmology.Om,
        OMb = parameters.cosmology.Ob,
        POWER_INDEX = parameters.cosmology.ns,
    )

    global_params = {
        "INITIAL_REDSHIFT": 300,
        "CLUMPING_FACTOR": 2.0,
    }

    halo_file_list = []
    dens_file_list = []

    Lbox = parameters.simulation.Lbox
    logger.debug(f'{parameters.simulation.Lbox=} in Mpc/h. Halo catalogs catalogs have masses in Msol/h and positions in Mpc/h.')

    IC = p21c.initial_conditions(
        user_params = user_params,
        cosmo_params = cosmo_params,
        random_seed = parameters.simulation.random_seed,
    )

    with p21c.global_params.use(**global_params):
        for redshift in parameters.solver.redshifts:

            # reuse the handler caching logic
            file_root = handler.file_root / f"21cmfast_{parameters.unique_hash()}"
            file_root.mkdir(parents=True, exist_ok=True)

            halo_fname = f'haloes_z{redshift}.h5'
            field_fname = f'densities_z{redshift}.h5'

            f = h5py.File(file_root / halo_fname, 'w')
            f.close()
            f = h5py.File(file_root / field_fname, 'w')
            f.close()

            perturbed_field = p21c.perturb_field(
                redshift = redshift,
                # TODO directly pass it the redshift list
                init_boxes = IC,
                # user_params = user_params,
                # cosmo_params = cosmo_params,
                # astro_params = astro_params,
                # random_seed = random_seed,
            )
            halo_list = p21c.perturb_halo_list(
                redshift = redshift,
                # TODO directly pass it the redshift list
                init_boxes = IC,
                # user_params = user_params,
                # cosmo_params = cosmo_params,
                # astro_params = astro_params,
                # random_seed = random_seed,
            )

            halo_list.write(direc=file_root, fname = halo_fname)
            perturbed_field.write(direc=file_root, fname = field_fname)

            halo_file_list.append(file_root / halo_fname)
            dens_file_list.append(file_root / field_fname)


            # TODO - return slightly different format when reading
            # halo_list = {
            #     'X': halo_list.halo_coords[:, 0] * Lbox / user_params.HII_DIM,
            #     'Y': halo_list.halo_coords[:, 1] * Lbox / user_params.HII_DIM,
            #     'Z': halo_list.halo_coords[:, 2] * Lbox / user_params.HII_DIM,
            #     'M': halo_list.halo_masses * parameters.cosmology.h,
            #     'z': redshift,
            #     'Lbox': Lbox
            # }

    logger.info(f"Finished generating halos in {time.process_time() - start_time} seconds")
    return halo_file_list, dens_file_list
