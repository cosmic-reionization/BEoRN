"""
Generate the xHII field from the non linear density field using the excursion set formalism.  See 1403.0941 (2.3.2), 21cmFAST original paper, Zahn et al..
"""


import numpy as np
from .constants import *; from.cosmo import *
from astropy.convolution import convolve_fft
import dmcosmo as dm  ## This is the HMF package. Can be dl here : https://github.com/timotheeschaeffer/dmcosmo.git
from .run import load_delta_b
import datetime
from os.path import exists
from .astro import f_esc, f_star_Halo
import warnings
from .compute_profiles import Ngdot_ion
from .functions import *

def run_excursion_set(param):
    """
    Produces xHII map using the excursion set formalism from Furlanetto 2004.

    Parameters
    ----------
    param: Bunch
            The parameter file created using the beorn.par().
    Returns
    -------
            Nothing.
    """

    start_time = datetime.datetime.now()
    LBox = param.sim.Lbox  # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells
    catalog_dir = param.sim.halo_catalogs
    model_name = param.sim.model_name

    if catalog_dir is None :
        print('You should specify param.sim.halo_catalogs. Should be a file containing the halo catalogs.')
    print('Applying excursion set formalism to produce ionisation maps, with', nGrid,'pixels per dim. Box size is',LBox ,'cMpc/h.')


    if param.sim.cores > 1:
        import mpi4py.MPI
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()
    else:
        rank = 0
        size = 1

    for ii, filename in enumerate(os.listdir(catalog_dir)):
        if rank == ii % size:
            print('Core nbr',rank,'is taking care of snap',filename[4:-5])
            if exists('./grid_output/xHII_exc_set_' + str(nGrid)+ '_' + model_name + '_snap' + filename[4:-5]):
                print('xHII map for snapshot ',filename[4:-5],'already painted. Skiping.')
            else:
                print('----- Excursion set for snapshot nbr :', filename[4:-5], '-------')
                excursion_set(filename,param)
                print('----- Snapshot nbr :', filename[4:-5], ' is done -------')

    end_time = datetime.datetime.now()
    print('DONE. Stored the xHII grid. It took in total: ',end_time-start_time,'to do the excursion set.')
    print('  ')



def excursion_set(filename,param):
    """
    Produces xHII map with excursion set formalism for a single snapshot

    Parameters
    ----------
    param: Beorn parameter dictionnary.
    filename : halo catalog name. Will be called by load_delta_b in run.py

    Returns
    -------
    Nothing
    """


    nGrid       = param.sim.Ncell
    Lbox        = param.sim.Lbox
    catalog_dir = param.sim.halo_catalogs
    catalog     = catalog_dir + filename
    model_name = param.sim.model_name
    halo_catalog = load_f(catalog)
    z = halo_catalog['z']
    delta_field = load_delta_b(param,filename) # load the overdensity field delta = rho/rho_bar-1
    Mmin = param.source.M_min
    M0 = np.logspace(np.log10(Mmin), 15, 100, base=10)
    n_rec = param.exc_set.n_rec
    Nion = param.source.Nion

    fcoll_ion_ST = Nion * f_coll_norm(param,Mmin,z)/n_rec
    #fcoll_PS = f_coll_PS(param,Mmin,z)
    #renorm = fcoll_ST / fcoll_PS
    #print('renorm is : ', round(renorm, 5))

    pixel_size = Lbox / nGrid
    x = np.linspace(-Lbox / 2, Lbox / 2, nGrid)  # y, z will be the same.
    rx, ry, rz = np.meshgrid(x, x, x, sparse=True)
    rgrid = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

    ion_map  = np.zeros((nGrid, nGrid, nGrid)) # This will be our output xHII map.
    stepping = pixel_size * param.exc_set.stepping
    Rsmoothing_Max = param.exc_set.R_max  # Mpc/h, max distance to which we smooth

    Var_M0, dVarM0_dM = Variance(param, M0)
    Var_min = np.interp(Mmin, M0, Var_M0)

    dc_z = delta_c(z, param)

    print('pixel size is:', round(pixel_size,4), 'cMpc/h. With a stepping of ',param.exc_set.stepping  ,'pixel.')

    Rsmoothing_Max = 50  # Mpc/h, max distance to which we smooth
    #Smoothing_scales = np.flip(np.arange(pixel_size, Rsmoothing_Max, stepping))
    Rsmoothing = pixel_size #Rsmoothing_Max
    while Rsmoothing < Rsmoothing_Max: #pixel_size:
        kern = profile_kern(rgrid, Rsmoothing)
        smooth_rho_ov_rhobar = convolve_fft(delta_field + 1, kern, boundary='wrap', normalize_kernel=True,allow_huge=True)  ## rho_smooth/rho_bar
        Msmooth = M_of_R(Rsmoothing, param)

        Var_Rsmooth = np.interp(Msmooth, M0, Var_M0)
        ## Now to the smoothed density field.
        ## nion_arr collapsed fraction of matter in stars (or actually more of matter in "ionising photons")
        min_rho = np.min(smooth_rho_ov_rhobar) #rho/rho_bar
        max_rho = np.max(smooth_rho_ov_rhobar)
        rho_norm_array = np.linspace(min_rho,max_rho,100)
        ind_max = np.argmin(np.abs(M0 - Msmooth))  # integrate only up to the Mass contained in Rsmooth
        M_range = M0[:ind_max]

                # to avoid computing the integral for each grid pixel, we do this interpolation trick
        nion_arr = np.trapz(Nion * np.nan_to_num(f_esc(param,M_range) * f_star_Halo(param,M_range) * np.abs(dVarM0_dM[:ind_max]) * f_Conditional(dc_z, Var_M0[:ind_max], ((rho_norm_array[:,None]-1)/D(1/(1+z),param)),Var_Rsmooth)),M_range)

        ## when nion_arr turns negative, it means that delta_rho/D(z) is larger than delta(z).
        ## hence the cell has collapsed in a halo of mass at least M(Rsmooth)
        ## the value of nion is hence fstar(Msmooth) * fesc * Nion
        nion_arr[np.where(nion_arr < 0)] = Nion * np.interp(Msmooth, M0, f_esc(param, M0) * f_star_Halo(param, M0))
        nion_arr = nion_arr / n_rec
        ###set the max value of nion to be 1 (equivalent of fcoll, but for the ionisation fraction)
        ### --> AFTER this you can normalize to ShethTormen fcoll_ion !!
        nion_arr = nion_arr.clip(max=1)
        nion_grid = np.interp(smooth_rho_ov_rhobar, rho_norm_array, nion_arr)

        #renorm = fcoll_ST / (np.mean(nion_grid) / Nion) ## renormalize to the collapsed fraction given by ST HMF (including fstar and fesc..)
        #nion_grid = renorm * nion_grid

        nion_grid = nion_grid / np.mean(nion_grid) * fcoll_ion_ST
        ion_map[np.where(nion_grid >= 1)] = 1

        if len(np.where(nion_grid >= 1)[0])==0:
            print('Stopping at Rsmoothing  =',np.round(Rsmoothing,3),'Mpc/h')
            break  ## when there is no more ionized cell, we stop the loop
        Rsmoothing = param.exc_set.stepping * Rsmoothing

    #### Now without smoothing : just replace Msmooth by the mass in a cell,
    #### and the smoothed rho/rhobar by the full unsmoothed density field
    Mcell = 4 * rhoc0 * param.cosmo.Om * np.pi * (Lbox / nGrid) ** 3 / 3
    Var_cell = np.interp(Mcell, M0, Var_M0)

    min_rho = np.min(delta_field + 1)  # rho/rho_bar
    max_rho = np.max(delta_field + 1)
    rho_norm_array = np.linspace(min_rho, max_rho, 100)
    nion_arr = np.trapz(Nion * np.nan_to_num(f_esc(param, M0) * f_star_Halo(param, M0) * np.abs(dVarM0_dM) * f_Conditional(dc_z, Var_M0, ( (rho_norm_array[:, None] - 1) / D(1 / (1 + z), param)), Var_cell)), M0)
    nion_arr[np.where(nion_arr < 0)] = Nion * np.interp(Mcell, M0, f_esc(param, M0) * f_star_Halo(param, M0))
    nion_arr = nion_arr/n_rec
    nion_arr = nion_arr.clip(max=1)
    nion_grid = np.interp(delta_field + 1, rho_norm_array, nion_arr)
    nion_grid = nion_grid / np.mean(nion_grid) * fcoll_ion_ST

    ion_map[np.where(nion_grid >= 1)] = 1
    nion_grid = nion_grid.clip(max=1)
    ion_map[np.where(ion_map == 0)] = nion_grid[np.where(ion_map == 0)]
    print('Done for z = ',z,', xHII = ',np.mean(ion_map))

    save_f(file = './grid_output/xHII_exc_set_' + str(nGrid)+ '_' + model_name + '_snap' + filename[4:-5], obj = ion_map)




def profile_kern(r, size):
    """
    We use this kernel to smooth the density field over certain scales.
        Kernel  is 1 for r < size and 0 outside
        Size : comoving size in cMpc/h (== the smoothing scale R)
    """
    return 1 - 1 / (1 + np.exp(-10000 * (np.abs(r) - size)))


#   return 1-1/(1+np.exp(-10000*(np.abs(r)-size)))
def profile_kern_sharpk(r, size):
    ## sharpk filter smoothing kernel
    return W_tophat(np.abs(r / size))



def f_coll_norm(param,Mmin,z):
    """
    param : beorn param file
    Fraction of total matter that "collapsed" into ionising photons.
    We use this to normalize the exc set results to the Sheth Tormen HMF (or the HMF that fits the halo catalog given by the user).
    """
    par = HMF_par(param)
    par.code.z = [z]
    par.PS.A = 0.322
    HMF = dm.HMF(par)
    HMF.generate_HMF(par)
    ind_min = np.argmin(np.abs(HMF.tab_M - Mmin))
    #fcoll_ST = np.trapz(f_esc(param, HMF.tab_M[ind_min:]) * f_star_Halo(param, HMF.tab_M[ind_min:]) * HMF.HMF[0][ind_min:],HMF.tab_M[ind_min:]) / param.cosmo.Om / rhoc0  # integral of dndlnM dM
    fcoll_ion_ST = np.trapz(f_esc(param, HMF.tab_M[ind_min:]) * f_star_Halo(param, HMF.tab_M[ind_min:]) * HMF.HMF[0][ind_min:],HMF.tab_M[ind_min:]) / param.cosmo.Om / rhoc0

    return fcoll_ion_ST

def f_coll_PS(param,Mmin,z):
    """
    param : beorn param file
    Fraction of total matter that "collapsed" into ionising photons assuming a PS HMF (i.e p=0,q=1)
    We use this to normalize the exc set results.
    """
    ### Collapsed fraction using PS HMF.
    par = HMF_par(param)
    par.code.z = [z]
    par.PS.p = 0
    par.PS.q = 1
    par.PS.A = 0.5
    HMF = dm.HMF(par)
    HMF.generate_HMF(par)
    ind_min = np.argmin(np.abs(HMF.tab_M - Mmin))
    fcoll_PS = np.trapz( f_esc(param, HMF.tab_M[ind_min:]) * f_star_Halo(param, HMF.tab_M[ind_min:]) * HMF.HMF[0][ind_min:],HMF.tab_M[ind_min:]) / param.cosmo.Om / rhoc0  # integral de dndlnM dM
    return fcoll_PS

def W_tophat(x):
    """
    Tophat filter in Fourier space.
    """
    return 3*(np.sin(x)-x*np.cos(x))/x**3

def Variance(param,mm):
    """
    param : beorn param file
    Sigma^2 at z=0. Used to compute the barrier.
    """
    ps = read_powerspectrum(param)
    kk_ = ps['k']
    PS_ = ps['P']
    R_ = ((3 * mm / (4 * rhoc0 * param.cosmo.Om * np.pi)) ** (1. / 3))
    Var = np.trapz(kk_ ** 2 * PS_ * W_tophat(kk_ * R_[:, None]) ** 2 / (2 * np.pi ** 2), kk_, axis=1)
    dVar_dM = np.gradient(Var, mm)
    return Var, dVar_dM


def f_Conditional(dc,S,dc0,S0):
    """
    Conditional first crossing distribution. Neede for the subhalo mass function in the fcoll formula.
    """

    warnings.filterwarnings("ignore")
    return (dc-dc0)/np.sqrt(2*np.pi*(S-S0)**3)*np.exp(-(dc-dc0)**2/2/(S-S0))

delta_c0 = 1.686
def delta_c(z,par):
    return par.exc_set.delta_c / D(1/(z+1),par)

def R_of_M(M,par):
    return (3 * M / (4 * rhoc0 * par.cosmo.Om * np.pi)) ** (1. / 3)
def M_of_R(R,par):
    return 4 * rhoc0 * par.cosmo.Om * np.pi * R**3/3


def HMF_par(param):
    """
    Create a dict for the dmcosmo python package, with the Halo Mass function parameters updated from the one chosen by the user.

    Parameters
    ----------
    param: Beorn parameter dictionnary.

    Returns
    -------
    A dmcosmo parameter dictionnary
    """

    par = dm.par()
    par.cosmo.Om   = param.cosmo.Om
    par.cosmo.Ob   = param.cosmo.Ob
    par.cosmo.Ol   = 1 - param.cosmo.Om
    par.cosmo.h    = param.cosmo.h
    par.PS.c       = param.exc_set.c
    par.PS.delta_c = param.exc_set.delta_c
    par.PS.p       = param.exc_set.p
    par.PS.filter  = param.exc_set.filter
    par.PS.q       = param.exc_set.q
    par.PS.A       = param.exc_set.A
    return par




from beorn.astro import f_star_Halo, f_esc

def Nion_(Mh,param):
    """
    Number of ionising photons for a given halo. This function is used for the Sem Num method of Majumdar 2014.

    Parameters
    ----------
    param: Beorn parameter dictionnary.
    Mh : Halo Mass in Msol/h

    Returns
    -------
    The total number of ionising photons produced by Mh.
    """
    Nion, Om, Ob, h0 = param.source.Nion, param.cosmo.Om,param.cosmo.Ob,param.cosmo.h
    if param.source.type == 'Ghara':
        print('CAREFUL, Ghara source type is chosen, Nion becomes just a fine tuning multiplicative factor')
        return param.source.Nion * 1.33 * 1e43 * Mh/h0 * 1e7 * sec_per_year ### multiplying by 10 Myr expressed in seconds
    else :
        return f_star_Halo(param,Mh) * f_esc(param,Mh) * Ob/Om * Mh/h0 / m_p_in_Msun * Nion



def Nion_new(Mh,z,param):
    """
    NOT USED !!
    Consistent way of computing this for a non flat fesc and fstar (integral of Nion_dot)
    Number of ionising photons for a given halo. This function is used for the Sem Num method of Majumdar 2014.

    Parameters
    ----------
    param: Beorn parameter dictionnary.
    Mh : Halo Mass in Msol/h

    Returns
    -------
    The total number of ionising photons produced by Mh.
    """
    zmax = 30
    zz   = np.arange(z,zmax,0.2)
    Mh_z = Mh * np.exp(-param.source.alpha_MAR*(zz-z))
    dNion_dz = Ngdot_ion(param, zz, Mh_z)/((zz + 1) * Hubble(zz, param)  )*sec_per_year
    return np.trapz(dNion_dz,zz)




def run_Sem_Num(param):
    """
    Run the Sem Num method (Majumdar 2014) to produce ionisation map.

    Parameters
    ----------
    param: Beorn parameter dictionnary.

    Returns
    -------
    Nothing
    """

    start_time = datetime.datetime.now()
    LBox = param.sim.Lbox  # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells
    catalog_dir = param.sim.halo_catalogs
    model_name = param.sim.model_name

    if catalog_dir is None :
        print('You should specify param.sim.halo_catalogs. Should be a file containing the halo catalogs.')
    print('Applying Sem Numerical Method on top of halos to produce ionisation maps, with', nGrid,'pixels per dim. Box size is',LBox ,'cMpc/h.')

    if param.sim.cores > 1:
        import mpi4py.MPI
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()
    else:
        rank = 0
        size = 1

    for ii, filename in enumerate(os.listdir(catalog_dir)):
        if rank == ii % size:
            print('Core nbr',rank,'is taking care of snap',filename[4:-5])
            if exists('./grid_output/xHII_Sem_Num_' + str(nGrid)+ '_' + model_name + '_snap' + filename[4:-5]):
                print('xHII map for snapshot ',filename[4:-5],'already painted. Skiping.')
            else:
                print('----- SemNum for snapshot nbr :', filename[4:-5], '-------')
                Sem_Num(filename,param)
                print('----- Snapshot nbr :', filename[4:-5], ' is done -------')

    end_time = datetime.datetime.now()
    print('DONE. Stored the xHII grid. It took in total: ',end_time-start_time,'to do the Semi Num method.')
    print('  ')





def Sem_Num(filename,param):
    """
    Produces xHII map with Sem Num method for a single snapshot.

    Parameters
    ----------
    param: Beorn parameter dictionnary.
    filename : halo catalog name. Will be called by load_delta_b in run.py

    Returns
    -------
    Nothing
    """

    start_time = datetime.datetime.now()
    nGrid       = param.sim.Ncell
    Lbox        = param.sim.Lbox
    z_start     = param.solver.z_max
    halo_catalog = load_f(param.sim.halo_catalogs + filename)
    model_name = param.sim.model_name
    H_Masses, H_X, H_Y, H_Z, z = halo_catalog['M'], halo_catalog['X'], halo_catalog['Y'], halo_catalog['Z'], halo_catalog['z']

    delta_field = load_delta_b(param,filename) # load the overdensity field delta = rho/rho_bar-1

    n_rec = param.exc_set.n_rec
    Nion, Om, Ob, h0 = param.source.Nion, param.cosmo.Om, param.cosmo.Ob, param.cosmo.h

    pixel_size = Lbox / nGrid
    x = np.linspace(-Lbox / 2, Lbox / 2, nGrid)  # y, z will be the same.
    rx, ry, rz = np.meshgrid(x, x, x, sparse=True)
    rgrid = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

    ion_map  = np.zeros((nGrid, nGrid, nGrid))  # This will be our final xHII map.
    Rsmoothing_Max = param.exc_set.R_max        # Mpc/h, max distance to which we smooth

    M_Bin = np.logspace(np.log10(param.sim.Mh_bin_min), np.log10(param.sim.Mh_bin_max), param.sim.binn, base=10)
    Mh_bin_z = M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))
    ##Bin the halo masses to produce the Nion grid quicker.
    Indexing = np.argmin(np.abs(np.log10(H_Masses[:, None] / Mh_bin_z)), axis=1)
    print('There are', H_Masses.size, 'halos at z=', z, )

    if H_Masses.size == 0:
        print('There aint no sources')
        ion_map = np.array([0])

    else:
        Pos_Bubles = np.vstack((H_X, H_Y, H_Z)).T  # Halo positions.
        Pos_Bubbles_Grid = np.array([Pos_Bubles / Lbox * nGrid]).astype(int)[0]
        Pos_Bubbles_Grid[np.where(Pos_Bubbles_Grid == nGrid)] = nGrid - 1

        Nion_grid = np.zeros((nGrid, nGrid, nGrid))  # grid containing the value of Nion (see Majumdar14 eq.3)


        for ih in range(len(Mh_bin_z)):
            if param.source.M_min<Mh_bin_z[ih]:
               # source_grid = np.zeros((nGrid, nGrid, nGrid))
                indices = np.where(Indexing == ih)

                base_nGrid_position = Pos_Bubbles_Grid[indices][:, 0] + nGrid * Pos_Bubbles_Grid[indices][:, 1] + nGrid ** 2 * Pos_Bubbles_Grid[ indices][:, 2]
                unique_base_nGrid_poz, nbr_of_halos = np.unique(base_nGrid_position, return_counts=True)

                ZZ_indice = unique_base_nGrid_poz // (nGrid ** 2)
                YY_indice = (unique_base_nGrid_poz - ZZ_indice * nGrid ** 2) // nGrid
                XX_indice = (unique_base_nGrid_poz - ZZ_indice * nGrid ** 2 - YY_indice * nGrid)
                Nion_grid[XX_indice, YY_indice, ZZ_indice] += Nion_(Mh_bin_z[ih],param) * nbr_of_halos

                #for i, j, k in Pos_Bubbles_Grid[indices]:
                #   source_grid[i, j, k] += 1
                #Nion_grid += source_grid * Nion_(Mh_bin_z[ih], param)
            #Nion_grid += source_grid * Nion_new(Mh_bin_z[ih],z, param)


        print('Ion Fraction should be  ',round(np.sum(Nion_grid)/(rhoc0*Ob/h0/m_p_in_Msun * Lbox**3)/n_rec, 3)) #theoretically expected value (Nion_to/N_H_tot)

        Rsmoothing = pixel_size
        ii = 0
        nbr_ion_pix = 1 # arbitraty value larger than 0
        while Rsmoothing < Rsmoothing_Max and nbr_ion_pix > 0 and np.mean(ion_map) < 1: # to stop the while loop earlier if there is no more ionisation.
            kern = profile_kern(rgrid, Rsmoothing)
            smooth_delta = convolve_fft(delta_field, kern, boundary='wrap', normalize_kernel=True,allow_huge=True)  # Smooth the density field
            nbr_H = (smooth_delta + 1) * rhoc0 * Ob / h0 / m_p_in_Msun * pixel_size ** 3 # Grid with smoothed number of H atom per pixel.
            Nion_grid_smoothed = convolve_fft(Nion_grid, kern, boundary='wrap', normalize_kernel=True,allow_huge=True)  ## Grid with smoothed the nbr of ionising photons
            ion_map[np.where(Nion_grid_smoothed / nbr_H / n_rec >= 1)] = 1  # compare Nion and nH in each pixel. Ionised when Nion/N_H/n_rec >= 1
            Rsmoothing = Rsmoothing * 1.1
            nbr_ion_pix = len(np.where(Nion_grid_smoothed / nbr_H / n_rec >= 1)[0])
            if ii % 5 == 0:
                print('Rsmoothing is', Rsmoothing, 'there are ', len(np.where(Nion_grid_smoothed / nbr_H / 1.5 >= 1)[0]),'ionisations.', 'mean Nion is')
            ii += 1

        ##partial ionisations
        nbr_H = (delta_field + 1) * rhoc0 * Ob / h0 / m_p_in_Msun * pixel_size ** 3
        ion_map[np.where(Nion_grid / nbr_H/n_rec >= 1)] = 1
        xHII_partial = Nion_grid / nbr_H/n_rec
        indices_partial = np.where(ion_map < 1)
        ion_map[indices_partial] = xHII_partial[indices_partial]

    if np.mean(ion_map) == 1:
        ion_map = np.array([1])

    end_time = datetime.datetime.now()
    print('Done with z=', z,': xHII =', np.mean(ion_map),'it took :',end_time-start_time)
    save_f(file = './grid_output/xHII_Sem_Num_' + str(nGrid)+ '_' + model_name + '_snap' + filename[4:-5], obj = ion_map)