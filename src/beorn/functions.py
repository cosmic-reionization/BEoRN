"""
Basic functions to load and save profiles, 3D maps etc...
"""

import pickle
import numpy as np
from .constants import rhoc0, Tcmb0

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
    indices = np.where(halo_catalog['M'] > param.source.M_min)

    # remove halos not forming stars
    halo_catalog['M'] = halo_catalog['M'][indices]
    halo_catalog['X'] = halo_catalog['X'][indices]
    halo_catalog['Y'] = halo_catalog['Y'][indices]
    halo_catalog['Z'] = halo_catalog['Z'][indices]

    return halo_catalog




def format_file_name(param,dir_name,z,qty):
    """
    Parameters
    ----------
    param : Bunch
    z : redshift, float.
    qty : str. The quantity of interest. Can be dTb, Tk, xal, xHII
    dir : str. The directory where we store the grids.

    Returns
    ----------
    The name of the pickle file containing the 3D maps of quantity qty.
    """
    out_name = param.sim.model_name
    z_str = z_string_format(z)
    nGrid: str = str(param.sim.Ncell)
    return dir_name + qty + '_' + nGrid + '_' + out_name + '_z' + z_str


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

    out_name = param.sim.model_name
    dir_name = './grid_output/'
    z_str = z_string_format(z)
    nGrid: str = str(param.sim.Ncell)

    if type == 'dTb':
        return load_f(dir_name + 'dTb_' + nGrid + '_' + out_name + '_z' + z_str)
    elif type == 'lyal':
        return load_f(dir_name + 'xal_' + nGrid + '_' + out_name + '_z' + z_str)
    elif type == 'Tk':
        return load_f(dir_name + 'Tk_' + nGrid + '_' + out_name + '_z' + z_str)
    elif type == 'exc_set':
        return load_f(dir_name + 'xHII_exc_set_' + nGrid + '_' + out_name + '_z' + z_str)
    elif type == 'sem_num':
        return load_f(dir_name + 'xHII_Sem_Num_' + nGrid + '_' + out_name + '_z' + z_str)
    elif type == 'bubbles':
        grid = load_f(dir_name + 'xHII_' + nGrid + '_' + out_name + '_z' + z_str)
        if np.all(grid == 0):
            Ncell = param.sim.Ncell
            return np.zeros((Ncell, Ncell, Ncell))
        elif np.all(grid == 1):
            Ncell = param.sim.Ncell
            return np.zeros((Ncell, Ncell, Ncell)) + 1
        else :
            return grid
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
        save_f(file=dir_name + 'dTb_' + nGrid + '_' + out_name + '_z' + z_str, obj=grid)
    elif type == 'lyal':
        save_f(file=dir_name + 'xal_' + nGrid + '_' + out_name + '_z' + z_str, obj=grid)
    elif type == 'Tk':
        save_f(file=dir_name + 'Tk_' + nGrid + '_' + out_name + '_z' + z_str, obj=grid)
    elif type == 'exc_set':
        save_f(file=dir_name + 'xHII_exc_set_' + nGrid  + '_' + out_name + '_z' + z_str, obj=grid)
    elif type == 'sem_num':
        save_f(file=dir_name + 'xHII_Sem_Num_' + nGrid + '_' + out_name + '_z' + z_str, obj=grid)
    elif type == 'bubbles':
        save_f(file=dir_name + 'xHII_' + nGrid  + '_' + out_name + '_z' + z_str, obj=grid)
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






def Beta(zz,PS,qty='Tk'):
    if qty=='Tk':
        Tcmb = Tcmb0 *  (1 + zz)
        beta_T = Tcmb / (PS['Tk'] - Tcmb)
        return beta_T
    elif qty == 'lyal':
        x_al = PS['x_al']
        x_tot = x_al+ PS['x_coll']
        return x_al / x_tot / (1 + x_tot)
    elif qty=='reio':
        return -PS['x_HII']/(1-PS['x_HII'])
    else:
        print('qty should be either Tk, lyal, or reio.')



from datetime import timedelta

def print_time(delta_t):
    """
    Parameters
    ----------
    delta_t : output of time.time()

    Returns
    ----------
    A clean time in hh:mm:ss
    """
    return "{:0>8}".format(str(timedelta(seconds=round(delta_t))))




def load_pkdgrav_density_field(file,LBox,nGrid):
    """
    Parameters
    ----------
    file : String. Path to the pkdgrav density field
    LBox : Float, box size in Mpc/h
    nGrid : Float, number of grid pixels

    Returns
    ----------
    delta = rho_m/rho_mean-1
    3-D mesh grid. Size (nGrid,nGrid,nGrid)
    """
    dens = np.fromfile(file, dtype=np.float32)
    pkd = dens.reshape(nGrid, nGrid, nGrid)
    pkd = pkd.T  ### take the transpose to match X_ion map coordinates
    V_total = LBox ** 3
    V_cell = (LBox / nGrid) ** 3
    mass = pkd * rhoc0 * V_total
    rho_m = mass / V_cell
    delta_b = (rho_m) / np.mean(rho_m) - 1
    return delta_b