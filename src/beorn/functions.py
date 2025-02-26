"""
Basic functions to load and save profiles, 3D maps etc...
"""

import pickle
import numpy as np
from .constants import rhoc0, Tcmb0
import tools21cm as t2c
import os
from os.path import exists
from .parameters import Parameters
def load_f(file):
    import pickle
    prof = pickle.load(open(file, 'rb'))
    return prof


def save_f(file, obj):
    import pickle
    pickle.dump(file=open(file, 'wb'), obj=obj)


def load_halo(parameters: Parameters, z_str):
    """
    Load a halo catalog. Should be a pickle dictionnary. With 'M', 'X', 'Y', 'Z', and redshift 'z'.
    The halo catalogs should be in param.sim.halo_catalogs and end up with z_string
    z is the redshift of the snapshot (outupt of z_string_format)
    """
    if not isinstance(z_str,str):
        z_str = z_string_format(z_str)
    catalog_dir = parameters.simulation.halo_catalogs
    catalog = catalog_dir + z_str
    halo_catalog = load_f(catalog)
    indices = np.intersect1d(np.where(halo_catalog['M'] > parameters.source.halo_mass_min),np.where(halo_catalog['M'] < parameters.source.halo_mass_max))

    for dim in ['X','Y','Z']:
        # in case you want to do High rez on a sub box of your Nbody simulation
        indices = np.intersect1d(indices,np.where(halo_catalog[dim] < parameters.simulation.Lbox))

    # remove halos not forming stars
    halo_catalog['M'] = halo_catalog['M'][indices]
    halo_catalog['X'] = halo_catalog['X'][indices]
    halo_catalog['Y'] = halo_catalog['Y'][indices]
    halo_catalog['Z'] = halo_catalog['Z'][indices]

    return halo_catalog


def format_file_name(parameters: Parameters, dir_name, z, qty):
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
    out_name = parameters.simulation.model_name
    z_str = z_string_format(z)
    nGrid: str = str(parameters.simulation.Ncell)
    return dir_name + qty + '_' + nGrid + '_' + out_name + '_z' + z_str

def def_k_bins(parameters: Parameters):
    """
    The k-bins used to measure the power spectrum.
    If param.sim.kbin is given as an int, you need to specify kmin and kmax.
    If given as a string, it will read in the boundary of the kbins.
    """
    if isinstance(parameters.simulation.kbin, int):
        kbins = np.logspace(np.log10(parameters.simulation.kmin), np.log10(parameters.simulation.kmax), parameters.simulation.kbin, base=10)  # h/Mpc
    elif isinstance(parameters.simulation.kbin, str):
        kbins = np.loadtxt(parameters.simulation.kbin)
    else:
        print(
            'param.sim.kbin should be either a path to a text files containing kbins edges values or it should be an int.')
        exit()
    return kbins




def load_delta_b(parameters: Parameters, zz):
    """
    Parameters
    ----------
    param:Bunch
    zz : str. Output of fct z_string_format,

    Returns
    ----------
    3D meshgrid of delta_b = rho/mean_rho-1
    """

    LBox = parameters.simulation.Lbox
    nGrid = parameters.simulation.Ncell
    dens_field = parameters.simulation.dens_field

    if parameters.simulation.dens_field_type == 'pkdgrav':
        if dens_field is not None:
            print('reading pkdgrav density field....')
            delta_b = load_pkdgrav_density_field(dens_field + zz, LBox)
        else:
            print('no density field provided. Return 0 for delta_b.')
            delta_b = np.array([0])  # rho/rhomean-1 (usual delta here..)

    elif parameters.simulation.dens_field_type == '21cmFAST':
        delta_b = load_f(dens_field + zz + '.0')
    elif parameters.simulation.dens_field_type == 'array':
        delta_b = np.loadtxt(dens_field + zz)
    else:
        print('param.sim.dens_field_type should be either 21cmFAST or pkdgrav.')

    if nGrid != delta_b.shape[0] and dens_field is not None:
        delta_b = reshape_grid(delta_b,nGrid)

    return delta_b



def reshape_grid(grid, N):
    """
    Parameters
    ----------
    grid : (a,a,a) a 3D meshgrid.
    new_shape : int. the nbr of pixel per grid for the reshaped array

    Returns
    ----------
    3D meshgrid of shape (N,N,N)
    """
    N_ini = grid.shape[0]

    if (N_ini/N) % 1 != 0 and (N/N_ini) % 1 != 0 :
        print('Your param.sim.Ncell should be a mutiple of a divider of your input density field shape.')
        exit()

    else :
        new_shape = (N, N, N)
        if N < N_ini:
            print('Downsampling the density field to a shape ({},{},{})'.format(N, N, N))
            # Downsample by taking the mean of block_size x block_size x block_size blocks
            block_size = int(N_ini / N)
            # Reshape grid into blocks and take the mean
            arr2 = grid.reshape(new_shape[0], block_size, new_shape[1], block_size, new_shape[2], block_size).mean(axis=(1, 3, 5))

        else:
            print('Oversampling the density field to a shape ({},{},{})'.format(N, N, N))
            # Create arr2 by indexing and expanding grid
            arr2 = grid[np.arange(new_shape[0])[:, None, None] // 2, np.arange(new_shape[1])[None, :, None] // 2, np.arange(
            new_shape[2])[None, None, :] // 2]

    return  arr2

def load_grid(parameters: Parameters, z, type=None):
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

    out_name = parameters.simulation.model_name
    dir_name = './grid_output/'
    z_str = z_string_format(z)
    nGrid: str = str(parameters.simulation.Ncell)

    if type == 'dTb':
        grid = load_f(dir_name + 'dTb_' + nGrid + '_' + out_name + '_z' + z_str)
    elif type == 'lyal':
        grid = load_f(dir_name + 'xal_' + nGrid + '_' + out_name + '_z' + z_str)
    elif type == 'Tk':
        grid = load_f(dir_name + 'Tk_' + nGrid + '_' + out_name + '_z' + z_str)
    elif type == 'exc_set':
        grid = load_f(dir_name + 'xHII_exc_set_' + nGrid + '_' + out_name + '_z' + z_str)
    elif type == 'sem_num':
        grid = load_f(dir_name + 'xHII_Sem_Num_' + nGrid + '_' + out_name + '_z' + z_str)
    elif type == 'bubbles':
        grid = load_f(dir_name + 'xHII_' + nGrid + '_' + out_name + '_z' + z_str)
    elif type=='matter':
        grid = load_delta_b(parameters, z_str)
    else:
        print('grid type should be dTb, lyal, Tk, matter, exc_set, sem_num, or bubbles. Abort')
        exit()

    if grid.shape == (1,):
        Ncell = parameters.simulation.Ncell
        grid =  np.full((Ncell, Ncell, Ncell),grid[0])

    return grid


def save_grid(parameters: Parameters, z, grid, type=None):
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
    out_name = parameters.simulation.model_name
    z_str = z_string_format(z)
    nGrid: str = str(parameters.simulation.Ncell)

    if type == 'dTb':
        save_f(file=dir_name + 'dTb_' + nGrid + '_' + out_name + '_z' + z_str, obj=grid)
    elif type == 'lyal':
        save_f(file=dir_name + 'xal_' + nGrid + '_' + out_name + '_z' + z_str, obj=grid)
    elif type == 'Tk':
        save_f(file=dir_name + 'Tk_' + nGrid + '_' + out_name + '_z' + z_str, obj=grid)
    elif type == 'exc_set':
        save_f(file=dir_name + 'xHII_exc_set_' + nGrid + '_' + out_name + '_z' + z_str, obj=grid)
    elif type == 'sem_num':
        save_f(file=dir_name + 'xHII_Sem_Num_' + nGrid + '_' + out_name + '_z' + z_str, obj=grid)
    elif type == 'bubbles':
        save_f(file=dir_name + 'xHII_' + nGrid + '_' + out_name + '_z' + z_str, obj=grid)
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


def def_redshifts(parameters: Parameters):
    """
    Parameters
    ----------
    param:Bunch

    Returns
    ----------
    The input redshifts where profiles will be computed. It should correspond to some input density fields and halo catalogs.
    """
    # TODO implement as init of the data class directly
    if isinstance(parameters.solver.Nz, int):
        print('param.solver.Nz is given as an integer. We define z values in linspace from ', parameters.solver.z_max, 'to ',
              parameters.solver.z_min)
        z_arr = np.linspace(parameters.solver.z_max, parameters.solver.z_min, parameters.solver.Nz)
    elif isinstance(parameters.solver.Nz, list):
        print('param.solver.Nz is given as a list.')
        z_arr = np.array(parameters.solver.Nz)
    elif isinstance(parameters.solver.Nz, np.ndarray):
        print('param.solver.Nz is given as a np array.')
        z_arr = parameters.solver.Nz
    elif isinstance(parameters.solver.Nz, str):
        z_arr = np.loadtxt(parameters.solver.Nz)
        print('param.solver.Nz is given as a string. We read z values from ', parameters.solver.Nz)
    else:
        print('param.solver.Nz should be a string, list, np array, or an int.')
    return z_arr


def Beta(zz, PS, qty='Tk'):
    if qty == 'Tk':
        Tcmb = Tcmb0 * (1 + zz)
        beta_T = Tcmb / (PS['Tk'] - Tcmb)
        return beta_T
    elif qty == 'lyal':
        x_al = PS['x_al']
        x_tot = x_al + PS['x_coll']
        return x_al / x_tot / (1 + x_tot)
    elif qty == 'reio':
        return -PS['x_HII'] / (1 - PS['x_HII'])
    else:
        print('qty should be either Tk, lyal, or reio.')


def cross_PS(arr1, arr2, box_dims, kbins):
    return t2c.power_spectrum.cross_power_spectrum_1d(arr1, arr2, box_dims=box_dims, kbins=kbins)


def auto_PS(arr1, box_dims, kbins):
    return t2c.power_spectrum.power_spectrum_1d(arr1, box_dims=box_dims, kbins=kbins)


from datetime import timedelta


def delta_fct(grid):
    """
    grid : np.array, meshgrid.
    returns : grid/mean(grid)-1
    """
    return grid / np.mean(grid) - 1



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


def load_pkdgrav_density_field(file, LBox):
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
    nGrid = round(dens.shape[0]**(1/3))
    pkd = dens.reshape(nGrid, nGrid, nGrid)
    pkd = pkd.T  ### take the transpose to match X_ion map coordinates
    V_total = LBox ** 3
    V_cell = (LBox / nGrid) ** 3
    mass  = (pkd * rhoc0 * V_total).astype(np.float64)
    rho_m = mass / V_cell
    delta_b = (rho_m) / np.mean(rho_m, dtype=np.float64) - 1
    return delta_b



def pixel_position(X,Y,Z,LBox,nGrid):
    """
    Parameters
    ----------
    X,Y,Z : floats, positions in cMpc/h
    LBox : Float, box size in Mpc/h
    nGrid : Float, number of grid pixels

    Returns
    ----------
    Coordinates expressed in grid pixel unit (between 0 and nGrid-1)
    """
    Pos_Halos = np.vstack((X,Y,Z)).T  # Halo positions.
    Pos_Halos_Grid = np.array([Pos_Halos / LBox * nGrid]).astype(int)[0]%nGrid
    return Pos_Halos_Grid


def Gaussian(d,mean,S):
    return 1/np.sqrt(2*np.pi*S)*np.exp(-(d-mean)**2/2/S)


def smooth_field(field,Rsmoothing,Lbox, nGrid):
    """
    Parameters
    ----------
    field : 3d meshgrid with nGrid pixel per dim, in box size Lbox (Mpc/h).
    Lbox : float, box size in Mpc/h
    nGrid : int, number of grid pixels per dim
    Rsmoothing : float (Mpc/h), smoothing scale

    Returns
    ----------
    smoothed_field over a tophat kernel with radius Rsmoothing
    """

    from .excursion_set import profile_kern
    from astropy.convolution import convolve_fft

    x = np.linspace(-Lbox / 2, Lbox / 2, nGrid)  # y, z will be the same.
    rx, ry, rz = np.meshgrid(x, x, x, sparse=True)
    rgrid = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    kern = profile_kern(rgrid, Rsmoothing)
    del rgrid
    smoothed_field = convolve_fft(field, kern, boundary='wrap', normalize_kernel=True, allow_huge=True)
    del kern
    del field

    return smoothed_field


def print_halo_distribution(M_bin,nbr_halos):
    """
    Parameters
    ----------
    M_bin : halo mass bin
    nbr_halos : array of size (M_bin) containing for each value M_bin[i],
                the number of halos with mass M_bin[i] (closer to).

    Returns
    ----------
    Prints the halo distribution.
    """
    indices = np.where(nbr_halos>0)

    min_bin, max_bin = np.min(indices), np.max(indices)
    Mh_min, Mh_max = M_bin[min_bin], M_bin[max_bin]

    print(f'Halos are distributed between mass bin {min_bin} and {max_bin} ({Mh_min:.2e}, {Mh_max:.2e}). Here is the histogram:',nbr_halos[indices])




def initialise_mpi4py(parameters: Parameters):
    """
    Parameters
    ----------
    Will read in param the number of cores to use.

    Returns
    ----------
    Initialise the mpi4py parallelisation. Returns the rank, size, and com.
    """

    if parameters.simulation.cores > 1:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        import mpi4py.MPI
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()
    else:
        comm = None
        rank = 0
        size = 1

    return comm, rank, size

def Barrier(comm):
    """
    Parameters
    ----------
    comm : Either None, either MPI.COMM_WORLD.

    Returns
    ----------
    Just to avoid error when running BEoRN on a laptop without mpi4py
    """
    if comm is not None:
        comm.Barrier()




def format_grid_for_PS_measurement(Grid_Temp,Grid_xHII,Grid_xal,nGrid) :
    """
    Parameters
    ----------
    Grid_Temp,Grid_xHII,Grid_xal : the grids as we store them.
    nGrid : param.code.Ncell. Nbr of grid pixel per dim.

    Returns
    ----------
    If a grid is just a number (e.g. xHII = np.array([1]) when the whole universe is ionised), returns an array of one.
    This is to measure power and crosses spectra..
    """

    if Grid_Temp.size == 1:  ## to avoid error when measuring power spectrum
        Grid_Temp = np.full((nGrid, nGrid, nGrid), 1)
    if Grid_xHII.size == 1:
        if Grid_xHII == np.array([0]):
            Grid_xHII = np.full((nGrid, nGrid, nGrid), 0)  ## to avoid div by zero
        elif Grid_xHII == np.array([0]):
            Grid_xHII = np.full((nGrid, nGrid, nGrid), 1)  ## to avoid div by zero
    if Grid_xal.size == 1:
        Grid_xal = np.full((nGrid, nGrid, nGrid), 0)
    return Grid_Temp,Grid_xHII,Grid_xal



def gather_files(parameters: Parameters, path, z_arr, Ncell, remove=True):
    """
    Parameters
    ----------
    path : str
    z_arr : list of redshift to loop over

    Returns
    ----------
    Nothing. Loops over files named <<path + str(Ncell) + '_' + param.sim.model_name + '_' + z_str + '.pkl'>>,
    and gather their data into a single dictionnary.
    """

    from collections import defaultdict
    dd = defaultdict(list)

    for ii, z in enumerate(z_arr):
        z_str = z_string_format(z)
        file  = path + str(Ncell) + '_' + parameters.simulation.model_name + '_' + z_str + '.pkl'
        if exists(file):
            data_z = load_f(file)
            for key, value in data_z.items():
                dd[key].append(value)
            if remove:
                os.remove(file)

    for key, value in dd.items():  # change lists to numpy arrays
        dd[key] = np.array(value)

    if 'k' in dd:
        dd['k'] = data_z['k']

    save_f(file= path + str(Ncell) + '_' + parameters.simulation.model_name + '.pkl', obj=dd)







