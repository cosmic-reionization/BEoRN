"""
Basic functions to load and save profiles, 3D maps etc...
"""

import pickle
import numpy as np

def load_f(file):
    import pickle
    prof = pickle.load(open(file, 'rb'))
    return prof


def save_f(file, obj):
    import pickle
    pickle.dump(file=open(file, 'wb'), obj=obj)


def load_halo(param, z_str):
    """
    Load a halo catalog. Should be a pickle dictionnary. With 'M', 'X', 'Y', 'Z', and redshift 'z'.
    The halo catalogs should be in param.sim.halo_catalogs and end up with z_string
    z is the redshift of the snapshot (outupt of z_string_format)
    """
    catalog_dir = param.sim.halo_catalogs
    catalog = catalog_dir + z_str
    halo_catalog = load_f(catalog)
    return halo_catalog


def load_grid(param, z, type=None):
    """
    Parameters
    ----------
    param : Bunch
    z : redshift, float.
    type : str.

    Returns
    ----------
    3D map of the desired "type", at redshift z
    """

    dir_name = './grid_output/'
    out_name = param.sim.model_name
    z_str = z_string_format(z)
    nGrid: str = str(param.sim.Ncell)

    if type == 'dTb':
        return load_f(dir_name + 'dTb_Grid' + nGrid + out_name + '_snap' + z_str)
    elif type == 'lyal':
        return load_f(dir_name + 'xal_Grid' + nGrid + out_name + '_snap' + z_str)
    elif type == 'Tk':
        return load_f(dir_name + 'T_Grid' + nGrid + out_name + '_snap' + z_str)
    elif type == 'exc_set':
        return load_f(dir_name + 'xHII_exc_set_' + nGrid + '_' + out_name + '_snap' + z_str)
    elif type == 'sem_num':
        return load_f(dir_name + 'xHII_Sem_Num_' + nGrid + '_' + out_name + '_snap' + z_str)
    elif type == 'bubbles':
        return load_f(dir_name + 'xHII_Grid' + nGrid + out_name + '_snap' + z_str)
    else:
        print('grid type should be dTb, lyal, Tk, exc_set, sem_num, or bubbles. Abort')
        exit()


def save_grid(param, z, grid, type=None):
    """
    Parameters
    ----------
    param : Bunch
    z : redshift, float.
    type : str.
    grid : 3D meshrgrid

    Returns
    ----------
    Nothing. Save the grid in pickle file with the relevant name (corresponding to "type")
    """
    dir_name = './grid_output/'
    out_name = param.sim.model_name
    z_str = z_string_format(z)
    nGrid: str = str(param.sim.Ncell)

    if type == 'dTb':
        save_f(file=dir_name + 'dTb_Grid' + nGrid + out_name + '_snap' + z_str, obj=grid)
    elif type == 'lyal':
        save_f(file=dir_name + 'xal_Grid' + nGrid + out_name + '_snap' + z_str, obj=grid)
    elif type == 'Tk':
        save_f(file=dir_name + 'T_Grid' + nGrid + out_name + '_snap' + z_str, obj=grid)
    elif type == 'exc_set':
        save_f(file=dir_name + 'xHII_exc_set_' + nGrid + out_name + '_snap' + z_str, obj=grid)
    elif type == 'sem_num':
        save_f(file=dir_name + 'xHII_Sem_Num_' + nGrid + out_name + '_snap' + z_str, obj=grid)
    elif type == 'bubbles':
        save_f(file=dir_name + 'xHII_Grid' + nGrid + out_name + '_snap' + z_str, obj=grid)
    else:
        print('grid type should be dTb, lyal, exc_set, sem_num, or bubbles. Abort')
        exit()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.argmin(np.abs(array - value))
    return array[idx], idx


def z_string_format(zz):
    """
    Parameters
    ----------
    zz : Float

    Returns
    ----------
    string : The same redshift but written in format 00.00.
    """
    txt = "{:.2f}".format(zz)
    return txt.zfill(5)




def def_redshifts(param):
    """
    Parameters
    ----------
    param:Bunch

    Returns
    ----------
    The input redshifts where profiles will be computed. It should correspond to some input density fields and halo catalogs.
    """
    if isinstance(param.solver.Nz, int):
        print('param.solver.Nz is given as an integer. We define z values in linspace from ', param.solver.z_max, 'to ', param.solver.z_min)
        z_arr = np.linspace(param.solver.z_max, param.solver.z_min, param.solver.Nz)
    elif isinstance(param.solver.Nz, list):
        print('param.solver.Nz is given as a list.')
        z_arr = np.array(param.solver.Nz)
    elif isinstance(param.solver.Nz, np.ndarray):
        print('param.solver.Nz is given as a np array.')
        z_arr = param.solver.Nz
    elif isinstance(param.solver.Nz, str):
        z_arr = np.loadtxt(param.solver.Nz)
        print('param.solver.Nz is given as a string. We read z values from ', param.solver.Nz)
    else:
        print('param.solver.Nz should be a string, list, np array, or an int.')
    return z_arr


