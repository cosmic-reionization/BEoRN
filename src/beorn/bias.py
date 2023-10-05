
import math
import numpy as np
from scipy.interpolate import splrep,splev,interp1d
from .constants import *
from .cosmo import D, rhoc_of_z

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


def rhoNFW_fct(rbin,param):
    """
    NFW density profile. We do not use it in the code.
    """
    Mvir = param.source.M_halo
    cvir = param.source.C_halo
    rvir = (3.0*Mvir/(4.0 * np.pi * 200*rhoc_of_z(param)))**(1.0/3.0)
    rho0 = 200*rhoc_of_z(param)*cvir**3.0/(3.0*np.log(1.0+cvir)-3.0*cvir/(1.0+cvir))
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


def compute_bias(param, tab_M=None):
    import os
    from mpi4py import MPI
    import time
    comm = MPI.COMM_WORLD
    if not os.path.isdir('./Halo_bias'):
        os.mkdir('./Halo_bias')

    start_time = time.time()
    print('Comptunig halo bias.')

    if param.sim.cores > 1:
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
            measure_halo_bias(param, z, nGrid, tab_M=tab_M, kbins=kbins)

    comm.Barrier()

    if rank == 0:
        from collections import defaultdict
        dd = defaultdict(list)

        for ii, z in enumerate(z_arr):
            z_str = z_string_format(z)
            file = './Halo_bias/halo_bias_B'+str(Lbox) + '_' + str(nGrid) + 'grid_z' + z_str + '.pkl'
            if exists(file):
                bias__ = load_f(file)
                for key, value in bias__.items():
                    dd[key].append(value)
                os.remove(file)

        for key, value in dd.items():  # change lists to numpy arrays
            dd[key] = np.array(value)

        dd['k'] = bias__['k']

        save_f(file='./Halo_bias/halo_bias_B'+str(Lbox) + '_' + str(nGrid) +'.pkl', obj=dd)


        end_time = time.time()
        print('Finished computing halo bias. It took in total: ', end_time - start_time)
        print('  ')


def measure_halo_bias(param, z, nGrid, tab_M=None, kbins=None):
    Lbox = param.sim.Lbox
    z_str = z_string_format(z)
    Vcell = (Lbox / nGrid) ** 3
    halo_catalog = load_halo(param, z_str)
    H_Masses, H_X, H_Y, H_Z, z_catalog = halo_catalog['M'], halo_catalog['X'], halo_catalog['Y'], halo_catalog['Z'], \
                                         halo_catalog['z']
    if round(z_catalog / z, 2) != 1:
        print('ERROR. Redshifts do not match between halo catalog name and data in catalog.')
        exit()

    delta_rho = load_delta_b(param, z_str)
    print('at z = ', z, 'shape of delta_rho is ', delta_rho.shape)
    min_M = np.min(H_Masses)
    max_M = np.max(H_Masses)
    print('Min and Max : {:.2e}, {:.2e}'.format(min_M, max_M))

    if tab_M is None:
        M_bin = np.logspace(np.log10(min_M), np.log10(max_M), int(2 * np.log10(max_M / min_M)), base=10)
    else:
        M_bin = tab_M

    if kbins is None:
        Nk = 20
        kbin = Nk

    else:
        kbin = kbins
        Nk = len(kbins) - 1

    Nm = len(M_bin) - 1

    PS_h_m_arr = np.zeros((Nm, Nk))
    PS_m_m_arr = np.zeros((Nm, Nk))
    PS_h_h_arr = np.zeros((Nm, Nk))

    Pos_Halos = np.vstack((H_X, H_Y, H_Z)).T  # Halo positions.
    Pos_Halos_Grid = np.array([Pos_Halos / Lbox * nGrid]).astype(int)[0]
    Pos_Halos_Grid[np.where(Pos_Halos_Grid == nGrid)] = nGrid - 1

    Indexing = np.argmin(np.abs(np.log10(H_Masses[:, None] / M_bin)), axis=1)

    for im in range(len(M_bin) - 1):
        indices = np.where(Indexing == im)[0]
        print('mass bin', im, 'over', Nm)
        base_nGrid_position = Pos_Halos_Grid[indices][:, 0] + nGrid * Pos_Halos_Grid[indices][:, 1] + nGrid ** 2 * \
                              Pos_Halos_Grid[indices][:, 2]
        unique_base_nGrid_poz, nbr_of_halos = np.unique(base_nGrid_position, return_counts=True)

        ZZ_indice = unique_base_nGrid_poz // (nGrid ** 2)
        YY_indice = (unique_base_nGrid_poz - ZZ_indice * nGrid ** 2) // nGrid
        XX_indice = (unique_base_nGrid_poz - ZZ_indice * nGrid ** 2 - YY_indice * nGrid)

        Grid_halo_field = np.zeros((nGrid, nGrid, nGrid))
        Grid_halo_field[XX_indice, YY_indice, ZZ_indice] += 1 / Vcell * nbr_of_halos
        delta_h = Grid_halo_field / np.mean(Grid_halo_field) - 1
        PS_h_m = t2c.power_spectrum.cross_power_spectrum_1d(delta_h, delta_rho, box_dims=Lbox, kbins=kbin)
        PS_m_m = t2c.power_spectrum.power_spectrum_1d(delta_rho, box_dims=Lbox, kbins=kbin)
        PS_h_h = t2c.power_spectrum.power_spectrum_1d(delta_h, box_dims=Lbox, kbins=kbin)

        PS_h_m_arr[im, :] = PS_h_m[0]
        PS_m_m_arr[im, :] = PS_m_m[0]
        PS_h_h_arr[im, :] = PS_h_h[0]

    Dict = {}

    Dict['Mh'] = M_bin
    Dict['z'] = round(z, 2)
    Dict['k'] = PS_m_m[1]
    Dict['PS_h_m'] = np.concatenate((PS_h_m_arr, np.zeros((1, Nk))))
    Dict['PS_m_m'] = np.concatenate((PS_m_m_arr, np.zeros((1, Nk))))
    Dict['PS_h_h'] = np.concatenate((PS_h_h_arr, np.zeros((1, Nk))))

    save_f(file='./Halo_bias/halo_bias_B'+str(Lbox) + '_' + str( nGrid) + 'grid_z' + z_str + '.pkl', obj=Dict)



