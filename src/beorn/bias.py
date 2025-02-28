import os
import math
import numpy as np
import time

from scipy.interpolate import splrep,splev
from scipy.optimize import curve_fit
from .cosmo import D, rhoc_of_z
from .profiles_on_grid import log_binning,bin_edges_log, cumulated_number_halos
from .constants import *
from .functions import *

# TODO don't hardcode this
delta_c = 1.686

def delt_c(z,param):
    """
    Redshift dependent critical density.
    """
    delta_c = param.hmf.delta_c
    return delta_c/D(1/(1+z), param)


def wf_sharpk(y):
    return np.heaviside(1 - y, 0)

def wf_tophat(x):
    return 3 * (np.sin(x) - x * np.cos(x)) / (x) ** 3


def read_powerspectrum(param):
    """
    Linear power spectrum from file
    """
    names= 'k, P'
    PS = np.genfromtxt(param.cosmo.ps,usecols=(0,1),comments='#',dtype=None, names=names)
    return PS

def Variance_tophat(param,mm):
    """
    Sigma**2 at z=0 computed with a tophat filter. Used to compute the barrier.
    Output : Var and dlnVar_dlnM
    We reintroduce little h units to be consistent with Power spec units.
    """
    ps = read_powerspectrum(param)
    kk_ = ps['k']
    PS_ = ps['P']
    rhoc = 2.775e11 ### with h
    R_ = ((3 * mm / (4 * rhoc * param.cosmo.Om * np.pi)) ** (1. / 3))
    #Var = np.trapz(kk_ ** 2 * PS_ * wf_tophat(kk_ * R_[:, None]) ** 2 / (2 * np.pi ** 2), kk_, axis=-1)
    Var = np.trapz(kk_ ** 2 * PS_ * wf_tophat(kk_ * R_) ** 2 / (2 * np.pi ** 2), kk_)
   # dlnVar_dlnM = np.gradient(np.log(Var), np.log(mm) )
    return Var #, dlnVar_dlnM


def bias(z,param,Mass = None,bias_type='Tinker'):
    M = Mass
    if bias_type == 'ST':
        #ST bias
        q = 0.707 # sometimes called a
        p = 0.3
        dcz = delt_c(z,param)
        var = Variance_tophat(param,M)
        nu = dcz ** 2.0 / var
        # cooray and sheth
        e1 = (q * nu - 1.0) / delta_c
        E1 = 2.0 * p / dcz / (1.0 + (q * nu) ** p)
        bias = 1.0 + e1 + E1

    elif (bias_type == 'Tinker'):
        # tinker bias
        dcz = delt_c(z,param)
        dc = delta_c
        var = Variance_tophat(param,M)
        nu = dcz ** 2.0 / var
        y = np.log10(200)
        A = 1 + 0.24 * y * np.exp(-(4 / y) ** 4)
        a = 0.44 * y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4)
        c = 2.4
        bias = 1 - A * nu ** (a / 2) / (nu ** (a / 2) + dc ** a) + B * nu ** (b / 2) + C * nu ** (c / 2)
    return bias



def rho_2h(bias_, cosmo_corr_ ,param, z):
    """
    Large scale 2halo profile, matter density around a halo. In Msol/cMpc**3
    """
    return (bias_ * cosmo_corr_ * D(1/(z+1)**2,param) * param.cosmo.profile + 1.0) * param.cosmo.Om * rhoc_of_z(param, 0) * param.cosmo.clumping


def rhoNFW_fct(param,rbin,z,Mvir,cvir):
    """
    NFW density profile. We do not use it in the code.
    """

    rvir = (3.0*Mvir/(4.0 * np.pi * 200*rhoc_of_z(param,z)))**(1.0/3.0)
    rho0 = 200*rhoc_of_z(param,z)*cvir**3.0/(3.0*np.log(1.0+cvir)-3.0*cvir/(1.0+cvir))
    x = cvir*rbin/rvir
    return rho0/(x * (1.0+x)**2.0)

def R_halo(M_halo,z,param):
    """
    M_halo in Msol.
    Output : Halo radius is physical unit [pMpc]
    """
    return (3*M_halo/(4*math.pi*200*rhoc_of_z(param,z)*(1+z)**3))**(1.0/3)



def profile(bias_,cosmo_corr_,param, z):
    """
    Global profile of total matter, in Msol/Mpc**3
    This is a comoving density [(Msol)/(cMpc)**3] as a function of comoving radius [cMpc/h]
    """
    return rho_2h(bias_, cosmo_corr_, param, z)     #+ rhoNFW_fct(rbin,param)



def bar_density_2h(rgrid,param,z,Mass):
    """
    2h profiles in nbr of [baryons /co-cm**3]
    rgrid : radial coordinates in physical units [pMpc/h].
    Mass : Halo mass in Msol/h. Used to compute the bias.
    """
    # Profiles
    cosmofile = param.cosmo.corr_fct
    vc_r, vc_corr = np.loadtxt(cosmofile, usecols=(0, 1), unpack=True)
    corr_tck = splrep(vc_r, vc_corr, s=0)
    cosmo_corr = splev(rgrid * (1 + z) , corr_tck)  # r_grid * (1+self.z)  in cMpc/h --> To reach the correct scales and units for the correlation fucntion
    halo_bias = bias(z, param,Mass)
    # baryonic density profile in [co-cm**-3]
    norm_profile = profile(halo_bias, cosmo_corr, param, z) * param.cosmo.Ob / param.cosmo.Om * M_sun / (cm_per_Mpc) ** 3 / m_H

    return norm_profile


#### SCRIPT TO MEASURE THE HALO BIAS :
#### SCRIPT TO MEASURE THE HALO BIAS :

from beorn.functions import *


def compute_bias(param, tab_M=None,dir='',zmax = 100,cross=False,fit=False,remove=True):
    # zmax : will not compute bias for z> zmax.
    import os

    import time

    if not os.path.isdir(dir+'./Halo_bias'):
        os.mkdir(dir+'./Halo_bias')

    start_time = time.time()
    print('Computing halo bias.')

    if param.sim.cores > 1:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        import mpi4py.MPI
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()
    else:
        rank = 0
        size = 1

    kbins = def_k_bins(param)
    z_arr = def_redshifts(param)
    Ncell = param.sim.Ncell
    nGrid = Ncell
    Lbox = param.sim.Lbox

    for ii, z in enumerate(z_arr):
        z = np.round(z, 2)
        if rank == ii % size:
            if not cross:
                measure_halo_bias(param, z, nGrid, tab_M=tab_M, kbins=kbins,dir=dir,zmax=zmax,fit=fit)

            elif cross :
                measure_halo_bias_with_cross(param, z, nGrid, tab_M=tab_M, kbins=kbins, dir=dir, zmax=zmax,fit=fit)

            else :
                print('Cross parameter in compute bias should either be true or false.')

    if param.sim.cores > 1:
        comm.Barrier()


    if remove and rank == 0:
        gather_bias(param,cross=cross,dir=dir)

    end_time = time.time()
    print('Finished computing halo bias. It took in total: ', end_time - start_time)
    print('  ')



def gather_bias(param,cross,dir):
    if not cross:
        name = 'halo_bias_B'
    elif cross:
        name = 'halo_bias_with_cross_B'

    z_arr = def_redshifts(param)
    Ncell = param.sim.Ncell
    nGrid = Ncell
    Lbox = param.sim.Lbox
    from collections import defaultdict
    dd = defaultdict(list)

    for ii, z in enumerate(z_arr):
        z_str = z_string_format(z)
        file = dir + './Halo_bias/' + name + str(Lbox) + '_' + str(nGrid) + 'grid_z' + z_str + '.pkl'

        if os.path.exists(file):
            bias__ = load_f(file)
            for key, value in bias__.items():
                dd[key].append(value)
            os.remove(file)

    for key, value in dd.items():  # change lists to numpy arrays
        dd[key] = np.array(value)

    dd['k'] = bias__['k']

    save_f(file=dir + './Halo_bias/' + name + str(Lbox) + '_' + str(nGrid) + '.pkl', obj=dd)





def measure_halo_bias(param, z, nGrid, tab_M=None, kbins=None, name='',dir='',zmax=100):


    M_bin, kbin, Nm, Nk = def_tab_M_and_kbin(tab_M,kbins)

    PS_h_m_arr = np.zeros((Nm, Nk))
    PS_h_h_arr = np.zeros((Nm, Nk))
    Shot_Noise = np.zeros((Nm))
    Bias = np.zeros((Nm))
    Lbox = param.sim.Lbox
    z_str = z_string_format(z)
    Vcell = (Lbox / nGrid) ** 3

    if z<zmax:

        halo_catalog = load_halo(param, z_str)
        H_Masses, H_X, H_Y, H_Z, z_catalog = halo_catalog['M'], halo_catalog['X'], halo_catalog['Y'], halo_catalog['Z'], \
                                             halo_catalog['z']

        if round(z_catalog / z, 2) != 1:
            print('ERROR. Redshifts do not match between halo catalog name and data in catalog.')
            exit()

        if len(H_Masses) > 0:
            delta_rho = load_delta_b(param, z_str)
            print('at z = ', z, 'shape of delta_rho is ', delta_rho.shape)

            Pos_Halos_Grid = pixel_position(H_X, H_Y, H_Z, Lbox, nGrid)
            Indexing = log_binning(H_Masses, bin_edges_log(M_bin))
            Indexing = Indexing - 1

            PS_m_m = auto_PS(delta_rho, box_dims=Lbox, kbins=kbin)
            kk = PS_m_m[1]
            for im in range(len(M_bin)):
                if im == len(M_bin):
                    PS_h_m_arr[im, :] = np.zeros((Nk))
                    PS_h_h_arr[im, :] = np.zeros((Nk))

                indices = np.where(Indexing == im)[0]
                if len(indices) > 0:
                    print('mass bin', im, 'over', Nm, 'has', len(indices), 'halos')
                    unique_base_nGrid_poz, nbr_of_halos = cumulated_number_halos(param, H_X[indices], H_Y[indices],
                                                                             H_Z[indices], cic=False)
                    t1 = time.time()
                    delta_h = delta_halo(unique_base_nGrid_poz, nbr_of_halos, Lbox, nGrid)


                    PS_h_m = t2c.power_spectrum.cross_power_spectrum_1d(delta_h, delta_rho, box_dims=Lbox, kbins=kbin)

                    # PS_h_h   = t2c.power_spectrum.power_spectrum_1d(delta_h,box_dims = Lbox,kbins=kbin)

                    PS_h_m_arr[im, :] = PS_h_m[0]

                    # PS_h_h_arr[im,:] = PS_h_h[0]
                    Shot_Noise[im] = 1 / (len(indices) / Lbox ** 3)

                    bias__ = PS_h_m[0] / PS_m_m[0]
                    ind_to_average = np.intersect1d(np.where(kk < 0.3), np.where(~np.isnan(bias__)))
                    Bias[im] = np.mean(bias__[ind_to_average])
                    print('bias is', Bias[im])

                else:
                    Bias[im] = 0
                ## below 0.5 we found that the bias is well converged between 128^3 and 256^3 grids, so we take the average to define scale independent bias
                ## also, we are intersted in large scale bias in a first step.


        else :
            delta_rho = np.zeros((nGrid,nGrid,nGrid))
            PS_m_m = t2c.power_spectrum.power_spectrum_1d(delta_rho, box_dims=Lbox, kbins=kbin)
            kk = PS_m_m[1]



    Dict = {}
    Bias = Bias.clip(min=0)
    Dict['Mh'] = M_bin
    Dict['z'] = round(z, 2)
    Dict['k'] = kk
    Dict['PS_h_m'] = PS_h_m_arr
    Dict['PS_m_m'] = PS_m_m[0]
    Dict['PS_h_h'] = PS_h_h_arr
    Dict['Shot_Noise'] = Shot_Noise
    Dict['Bias'] = Bias

    save_f(file=dir+'./Halo_bias/halo_bias_B' + str(Lbox) + '_' + str(nGrid) + 'grid_z' + z_str + '.pkl',obj=Dict)




def def_tab_M_and_kbin(tab_M,kbins):
    if tab_M is None:
        min_M = np.min(H_Masses)
        max_M = np.max(H_Masses)
        print('Min and Max : {:.2e}, {:.2e}'.format(min_M, max_M))
        M_bin = np.logspace(np.log10(min_M), np.log10(max_M), int(2 * np.log10(max_M / min_M)), base=10)
    else:
        M_bin = tab_M
    Nm = len(M_bin)

    if kbins is None:
        Nk = 20
        kbin = Nk

    else:
        kbin = kbins
        Nk = len(kbins) - 1

    return M_bin, kbin, Nm, Nk


def delta_halo(unique_base_nGrid_poz,nbr_of_halos,Lbox,nGrid):
    """
    Parameters
    ----------
    unique_base_nGrid_poz, nbr_of_halos : output of cumulated_number_halos.
    Lbox : float, Box size in Mpc/h
    nGrid : int, nbr of grid pixels

    Returns
    ----------
    Halo overdensity field : Meshgrid (nGrid,nGrid,nGrid)
    """

    Vcell = (Lbox / nGrid) ** 3
    ZZ_indice = unique_base_nGrid_poz // (nGrid ** 2)
    YY_indice = (unique_base_nGrid_poz - ZZ_indice * nGrid ** 2) // nGrid
    XX_indice = (unique_base_nGrid_poz - ZZ_indice * nGrid ** 2 - YY_indice * nGrid)
    t2 = time.time()

    Grid_halo_field = np.zeros((nGrid, nGrid, nGrid))
    Grid_halo_field[XX_indice, YY_indice, ZZ_indice] += 1 / Vcell * nbr_of_halos
    delta_h = Grid_halo_field / np.mean(Grid_halo_field) - 1
    return delta_h



def measure_halo_bias_with_cross(param, z, nGrid, tab_M=None, kbins=None, name='',dir='',zmax=100,fit=False):

    ### same as above, we just measure b(M1,M2)

    M_bin, kbin, Nm, Nk = def_tab_M_and_kbin(tab_M,kbins)
    PS_h_m_arr = np.zeros((Nm,Nm, Nk))
    PS_h_h_arr = np.zeros((Nm,Nm, Nk))
    Nbr_Halos = np.zeros((Nm))
    Nbr_Pixels = np.zeros((Nm))
    Bias = np.zeros((Nm,Nm))
    Non_lin_Bias = np.zeros((Nm,Nm,Nk))


    Lbox = param.sim.Lbox
    z_str = z_string_format(z)
    Vcell = (Lbox / nGrid) ** 3

    PS_m_m = t2c.power_spectrum.power_spectrum_1d(np.zeros((nGrid, nGrid, nGrid)), box_dims=Lbox, kbins=kbin)
    kk = PS_m_m[1]

    if z<zmax:

        halo_catalog = load_halo(param, z_str)
        H_Masses, H_X, H_Y, H_Z, z_catalog = halo_catalog['M'], halo_catalog['X'], halo_catalog['Y'], halo_catalog['Z'], \
                                             halo_catalog['z']

        if round(z_catalog / z, 2) != 1:
            print('ERROR. Redshifts do not match between halo catalog name and data in catalog.')
            exit()

        if len(H_Masses) > 0:
            delta_rho = load_delta_b(param, z_str)
            PS_m_m = auto_PS(delta_rho, box_dims=Lbox, kbins=kbin)
            kk = PS_m_m[1]
            print('at z = ', z, 'shape of delta_rho is ', delta_rho.shape)

            Indexing = log_binning(H_Masses, bin_edges_log(M_bin))
            Indexing = Indexing - 1

            Dict_halo_unique_poz = {}
            for im in range(len(M_bin)):
                indices_im = np.where(Indexing == im)[0]
                unique_base_nGrid_poz, nbr_of_halos = cumulated_number_halos(param, H_X[indices_im], H_Y[indices_im],
                                                                             H_Z[indices_im], cic=False)
                Dict_halo_unique_poz[str(im)] = np.array((unique_base_nGrid_poz, nbr_of_halos))

                Nbr_Halos[im] = len(indices_im)  # total number of halos in this mass bin
                Nbr_Pixels[im] = len(unique_base_nGrid_poz)  # total number of occupied pixel in this mass bin

            print_halo_distribution(M_bin,Nbr_Halos)

            for im in range(len(M_bin)):
                indices_im = np.where(Indexing == im)[0]
                for jm in range(im, len(M_bin)):
                    if len(Dict_halo_unique_poz[str(im)][1]) > 0 and len(Dict_halo_unique_poz[str(jm)][1]) > 0:

                        unique_base_nGrid_poz, nbr_of_halos = Dict_halo_unique_poz[str(im)]
                        t1 = time.time()
                        delta_h_i =  delta_halo(unique_base_nGrid_poz,nbr_of_halos,Lbox,nGrid)
                        t3 = time.time()

                        unique_base_nGrid_poz, nbr_of_halos = Dict_halo_unique_poz[str(jm)]
                        delta_h_j = delta_halo(unique_base_nGrid_poz,nbr_of_halos,Lbox,nGrid)

                        t4 = time.time()

                        # PS_h_m   = t2c.power_spectrum.cross_power_spectrum_1d(delta_h_i,delta_rho,box_dims = Lbox,kbins=kbin)
                        # PS_m_m   = t2c.power_spectrum.power_spectrum_1d(delta_rho,box_dims = Lbox,kbins=kbin)
                        PS_h_h = cross_PS(delta_h_i, delta_h_j, box_dims=Lbox, kbins=kbin)

                        t5 = time.time()

                        PS_h_h_arr[im, jm, :] = PS_h_h[0]
                        bias__ = np.sqrt(PS_h_h[0] / PS_m_m[0])

                        ind_to_average = np.intersect1d(np.where(kk < 0.1), np.where(~np.isnan(bias__)))
                        ### we take 0.15 here because we found that sqrt(Phh/Pmm-shot noise) is not so well converged to Phm/Pmm
                        Bias[im, jm] = np.mean(bias__[ind_to_average])


                        if im == jm:
                            PS_h_m = t2c.power_spectrum.cross_power_spectrum_1d(delta_h_i, delta_rho,
                                                                                box_dims=Lbox, kbins=kbin)
                            bias__ = PS_h_m[0] / PS_m_m[0]

                            ind_to_average = np.intersect1d(np.where(kk < 0.3), np.where(~np.isnan(bias__)))
                            Bias[im, im] = np.mean(bias__[ind_to_average])
                            PS_h_m_arr[im, jm, :] = PS_h_m[0]

                        Non_lin_Bias[im, jm] = bias__

                        if fit :
                            indices_ = bias__ > 0 # just fit values that are positive and not nans.
                            if len(bias__[indices_]) > 0:
                                try :
                                    param_fit, covariance = fit_bias(Bias[im, jm], kk[indices_], bias__[indices_])
                                    fitted_bias = bias_fit(Bias[im, jm],kk,param_fit[0],param_fit[1])

                                except RuntimeError as re:
                                    print(f"Caught a RuntimeError: {re}")

                                except Exception as e:
                                    print(f"Caught a different exception: {e}")

                                finally:
                                    if not np.all(np.isnan(fitted_bias)):
                                        Non_lin_Bias[im, jm] = fitted_bias

                    else:
                        # PS_h_m_arr[im,jm,:] = np.zeros((Nk))
                        PS_h_h_arr[im, jm, :] = np.zeros((Nk))

               # if len(indices_im) > 0:
                  #  Nbr_Halos[im] = len(indices_im)# / Lbox ** 3)
                  #  Nbr_Pixels[im] = len(indices_im)  #
                    #PS_h_h_arr[im, im, :] -= Shot_Noise[im]
                   # print('mass bin', im, 'has shot noise',Shot_Noise[im] )


                for im in range(len(M_bin)):
                    for jm in range(im, len(M_bin)):
                        Bias[jm, im] = Bias[im, jm]
                        Non_lin_Bias[jm, im] = Non_lin_Bias[im, jm]


    Dict = {}
    Bias = Bias.clip(min=0)
    Dict['Mh'] = M_bin
    Dict['z'] = round(z, 2)
    Dict['k'] = kk
    Dict['PS_h_m'] = PS_h_m_arr
    Dict['PS_m_m'] = PS_m_m[0]
    Dict['PS_h_h'] = PS_h_h_arr
    Dict['Nbr_Halos'] = Nbr_Halos
    Dict['Nbr_Pixels'] = Nbr_Pixels
    Dict['Bias'] = Bias
    Dict['Non_lin_Bias'] = Non_lin_Bias

    save_f(file=dir+'./Halo_bias/halo_bias_with_cross_B' + str(Lbox) + '_' + str(nGrid) + 'grid_z' + z_str + '.pkl',obj=Dict)




def bias_fit(large_scale_bias,k,a,b):
    # function to fit the bias(k)
    return large_scale_bias+ a*(k)**b


def fit_bias(large_scale_bias,kk_,b_of_k):
    # run this function to find the best fit bias
    # kk_,b_of_k should not contain nan
    # large_scale_bias : mean value of b_of_k at large scales
    def fct_fit(k,a,b):
        return bias_fit(large_scale_bias,k,a,b)
    params, covariance = curve_fit(fct_fit, kk_, b_of_k)

    return params, covariance