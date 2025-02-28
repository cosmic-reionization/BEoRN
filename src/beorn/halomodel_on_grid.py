"""
We take as input the HM pprofiles, mass bining etc... and paint them on a grid
"""

import numpy as np
import os
import time
from scipy.interpolate import interp1d

from .constants import cm_per_Mpc, M_sun, m_H, sec_per_year
from .cosmo import T_adiab_fluctu, dTb_fct
from .profiles_on_grid import profile_to_3Dkernel, spreading_excess_fast, put_profiles_group, stacked_lyal_kernel, stacked_T_kernel, cumulated_number_halos, log_binning, bin_edges_log
from .couplings import x_coll, S_alpha
from .cosmo import dTb_factor
from .functions import *
from .run import *

def paint_profile_single_snap_HM_input(z_str, param,HM_PS, temp=True, lyal=True, ion=True, dTb=True, read_temp=False, read_ion=False,
                              read_lyal=False, RSD=False, xcoll=True, S_al=True, cross_corr=False, third_order=False,fourth_order=False,
                              cic=False, variance=False,Rsmoothing=0,truncate=False):
    """
    Paint the Tk, xHII and Lyman alpha profiles on a grid for a single halo catalog named filename.

    Parameters
    ----------
    HM_PS : dictionnary, output of halomodel.py
    param : dictionnary containing all the input parameters
    z_str : the redshift of the snapshot.
                    filename : the name of the halo catalog, contained in param.sim.halo_catalogs.
    temp, lyal, ion, dTb : which map to paint.
    S_alpha : if equals to False, then we write xal = rhoal*Salpha(mean quantity). If true, we take the full inhomogeneous Salpha.


    Returns
    -------
    Does not return anything. Paints and stores the grids int the directory grid_outputs.
    """

    start_time = time.time()
    model_name = param.sim.model_name
    M_Bin = HM_PS['M0']

    LBox = param.sim.Lbox  # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells

    halo_catalog = load_halo(param, z_str)
    H_Masses, H_X, H_Y, H_Z, z = halo_catalog['M'], halo_catalog['X'], halo_catalog['Y'], halo_catalog['Z'], \
                                 halo_catalog['z']

    ### To later add up the adiabatic Tk fluctuations at the grid level.
    delta_b = load_delta_b(param, z_str)  # rho/rhomean-1

    Om, Ob, h0 = param.cosmo.Om, param.cosmo.Ob, param.cosmo.h
    factor = dTb_factor(param)
    coef = rhoc0 * h0 ** 2 * Ob * (1 + z) ** 3 * M_sun / cm_per_Mpc ** 3 / m_H

    # find matching redshift between solver output and simulation snapshot.
    ind_z = np.argmin(np.abs(HM_PS['z'] - z))
    zgrid = HM_PS['z'][ind_z]
    print('z snapshot is ',z,'while z in HM is',zgrid)

    # if H_Masses is digitized to bin_edges_log[i] it means it should take the value of M_Bin[i-1] (bin 0 is what's on the left...)
    # M_Bin                0.   1.    2.    3.  ...
    # bin_edges_log     0.  | 1. |  2. |  3. |  4. ....
    Indexing = log_binning(H_Masses, bin_edges_log(M_Bin))
    Indexing = Indexing - 1
    ### same result as if you do np.argmin(np.abs(np.log10(H_Masses[:,None]/grid_model.Mh_history[ind_z, :]),axis=1), but faster


    xal_mean = 0
    if any(Indexing < 0):
        print('Need lower Mmin ! ')
        exit()

        # np.argmin(np.abs(np.log10(H_Masses[:, None] / grid_model.Mh_history[ind_z, :])), axis=1)
    print('There are', H_Masses.size, 'halos at z=', z, )
    print('Looping over halo mass bins and painting profiles on 3D grid .... ')
    if H_Masses.size == 0:
        print('There is no sources')
        Grid_xHII = np.array([0])
        Grid_Temp = T_adiab_fluctu(z, param, delta_b)

        Grid_xal = np.array([0])
        Grid_xcoll = x_coll(z=z, Tk=Grid_Temp, xHI=(1 - Grid_xHII), rho_b=(delta_b + 1) * coef)
        Grid_dTb = factor * np.sqrt(1 + z) * (1 - Tcmb0 * (1 + z) / Grid_Temp) * (1 - Grid_xHII) * (
                delta_b + 1) * Grid_xcoll / (1 + Grid_xcoll)
        Grid_dTb_RSD = np.array([0])
        xcoll_mean = np.mean(Grid_xcoll)
        del Grid_xcoll

    else:
        if np.max(H_Masses) > np.max(M_Bin):
            print('Max Mh_bin is :', np.max(M_Bin), 'while the largest halo in catalog is',
                  np.max(H_Masses))
            print('WARNING!!! You should use a larger value for param.sim.Mh_bin_max')
        if np.min(H_Masses) < np.min(M_Bin) and np.min(H_Masses)>param.source.M_min:
            print('WARNING!!! You should use a smaller value for param.sim.Mh_bin_min')

        Ionized_vol = 0 #xHII_approx(param, halo_catalog)[1]
       # print('Quick calculation from the profiles predicts xHII = ', round(Ionized_vol, 4))
        if Ionized_vol > 1:
            Grid_xHII = np.array([1])
            Grid_Temp = np.array([1])
            Grid_dTb = np.array([0])
            Grid_xal = np.array([0])
            print('universe is fully inoinzed. Return [1] for the xHII, T and [0] for dTb.')
        else:
            if not read_temp or not read_lyal or not read_ion:

                Pos_Halos_Grid = pixel_position(H_X, H_Y, H_Z,LBox,nGrid)



                Grid_xHII = np.zeros((nGrid, nGrid, nGrid))
                Grid_Temp = np.zeros((nGrid, nGrid, nGrid))
                Grid_xal = np.zeros((nGrid, nGrid, nGrid))
                Grid_xHII_i = np.zeros((nGrid, nGrid, nGrid))

                V_reio = 0
                for i in range(len(M_Bin)):
                    indices = np.where(Indexing == i)[0]  ## indices in H_Masses of halos that have an initial mass at z=z_start between M_Bin[i-1] and M_Bin[i]

                    if len(indices) > 0 and M_Bin[i] > param.source.M_min:
                       # radial_grid = grid_model.r_grid_cell / (1 + zgrid)  # pMpc/h
                       # x_HII_profile = np.zeros((len(radial_grid)))

                        rho_alpha_ = HM_PS['rho_al'][ind_z,:,i]*(1+z)**2* (h0 / cm_per_Mpc) ** 2 / sec_per_year
                        Temp_profile = HM_PS['rho_heat'][ind_z, :, i]

                        r_lyal = HM_PS['r_lyal']
                        r_temp = HM_PS['r_heat']
                        x_alpha_prof = 1.81e11 * rho_alpha_ / (1 + zgrid)  # We add up S_alpha(zgrid, T_extrap, 1 - xHII_extrap) later, a the map level.

                        ### This is the position of halos in base "nGrid". We use this to speed up the code.
                        ### We count with np.unique the number of halos in each cell. Then we do not have to loop over halo positions in --> profiles_on_grid/put_profiles_group
                        # base_nGrid_position = Pos_Halos_Grid[indices][:, 0] + nGrid * Pos_Halos_Grid[indices][:,1] + nGrid ** 2 * Pos_Halos_Grid[ indices][:,2]
                        # unique_base_nGrid_poz, nbr_of_halos = np.unique(base_nGrid_position, return_counts=True)
                        unique_base_nGrid_poz, nbr_of_halos = cumulated_number_halos(param, H_X[indices], H_Y[indices],H_Z[indices], cic=cic)

                        ZZ_indice = unique_base_nGrid_poz // (nGrid ** 2)
                        YY_indice = (unique_base_nGrid_poz - ZZ_indice * nGrid ** 2) // nGrid
                        XX_indice = (unique_base_nGrid_poz - ZZ_indice * nGrid ** 2 - YY_indice * nGrid)

                        ## Every halos in mass bin i are assumed to have mass M_bin[i].
                        if ion:
                            r_reio,x_HII_profile = HM_PS['r_reio'],HM_PS['rho_reio'][ind_z,:,i]
                            profile_xHII = interp1d(r_reio, x_HII_profile, bounds_error=False,
                                                    fill_value=(1, 0))
                            kernel_xHII = profile_to_3Dkernel(profile_xHII, nGrid, LBox)

                          #  V_bubble = 4*np.pi*np.trapz(x_HII_profile*r_reio**2,r_reio)
                           # print('V_bubble is :',V_bubble)
                           # V_reio += np.sum(nbr_of_halos)*V_bubble

                            if not np.any(kernel_xHII > 0):
                                ### if the bubble volume is smaller than the grid size,we paint central cell with ion fraction value
                                Grid_xHII_i[XX_indice, YY_indice, ZZ_indice] += np.trapz(
                                    x_HII_profile * 4 * np.pi * r_reio ** 2, r_reio) / (LBox / nGrid) ** 3 * nbr_of_halos

                            else:
                                renorm = np.trapz(x_HII_profile * 4 * np.pi * r_reio ** 2, r_reio) / (
                                        LBox) ** 3 / np.mean(kernel_xHII)
                             #   print('renorm for reio is ',renorm)
                                Grid_xHII_i += put_profiles_group(np.array((XX_indice, YY_indice, ZZ_indice)),nbr_of_halos,
                                       kernel_xHII * 1e-7 / np.sum(kernel_xHII)) * np.sum(kernel_xHII) / 1e-7 * renorm
                            del kernel_xHII


                        if lyal:
                            ### We use this stacked_kernel functions to impose periodic boundary conditions when the lyal or T profiles extend outside the box size. Very important for Lyman-a.
                            kernel_xal = stacked_lyal_kernel(r_lyal, x_alpha_prof, LBox, nGrid,
                                                             nGrid_min=param.sim.nGrid_min_lyal)
                            renorm = np.trapz(x_alpha_prof * 4 * np.pi * r_lyal ** 2, r_lyal) / (
                                    LBox) ** 3 / np.mean(kernel_xal)

                            M_a = np.trapz(x_alpha_prof * 4 * np.pi * r_lyal ** 2, r_lyal)
                            #print('Renorm for lyal is :',renorm)

                            if np.any(kernel_xal > 0):
                                Grid_xal += put_profiles_group(np.array((XX_indice, YY_indice, ZZ_indice)),
                                                               nbr_of_halos,
                                                               kernel_xal * 1e-7 / np.sum(
                                                                   kernel_xal)) * renorm * np.sum(kernel_xal) / 1e-7  # we do this trick to avoid error from the fft when np.sum(kernel) is too close to zero.

                                xal_mean += np.mean(kernel_xal) * renorm * len(indices)
                            del kernel_xal

                        if temp:
                            kernel_T = stacked_T_kernel(r_temp, Temp_profile, LBox, nGrid,
                                                        nGrid_min=param.sim.nGrid_min_heat)
                            renorm = np.trapz(Temp_profile * 4 * np.pi * r_temp ** 2, r_temp) / (
                                    LBox ) ** 3 / np.mean(kernel_T)
                            #print('Renorm for Tk is :', renorm)
                            if np.any(kernel_T > 0):
                                # Grid_Temp += put_profiles_group(Pos_Halos_Grid[indices],  kernel_T * 1e-7 / np.sum(kernel_T)) * np.sum(kernel_T) / 1e-7 * renorm
                                Grid_Temp += put_profiles_group(np.array((XX_indice, YY_indice, ZZ_indice)),
                                                                nbr_of_halos,
                                                                kernel_T * 1e-7 / np.sum(kernel_T)) * np.sum(
                                    kernel_T) / 1e-7 * renorm
                            del kernel_T
                            #print('MEAN GRID TEMP IS :',np.mean(Grid_Temp))

                        end_time = time.time()
                        print(len(indices), 'halos in mass bin ', i,
                              '. It took ' + print_time(end_time - start_time) + ' to paint the profiles.')

                print('.... Done painting profiles. ')
               # print('V reio total is ',V_reio)

                # Grid_Storage = np.copy(Grid_xHII_i)
                t_start_spreading = time.time()
                if np.sum(Grid_xHII_i) < nGrid ** 3 and ion:
                    Grid_xHII = spreading_excess_fast(param, Grid_xHII_i)
                else:
                    Grid_xHII = np.array([1])

                print('MEAN Grid_xHII is ',np.mean(Grid_xHII))

                print('.... Done. It took:', print_time(time.time() - t_start_spreading),
                      'to redistribute excess photons from the overlapping regions.')

                if np.all(Grid_xHII == 0):
                    Grid_xHII = np.array([0])
                if np.all(Grid_xHII == 1):
                    print('universe is fully inoinzed. Return [1] for Grid_xHII.')
                    Grid_xHII = np.array([1])


                Grid_Temp += T_adiab_fluctu(z, param, delta_b)
                print('MEAN GRID TEMP IS AFTER ADIAB FLUCTU ADDED IS :', np.mean(Grid_Temp))

            if read_temp:
                Grid_Temp = load_grid(param, z=z, type='Tk')
            if read_ion:
                Grid_xHII = load_grid(param, z=z, type='bubbles')
            if read_lyal:
                Grid_xal = load_grid(param, z=z, type='lyal')
            else:
                if S_al:
                    print('--- Including Salpha fluctuations in dTb ---')
                    Grid_xal = Grid_xal * S_alpha(z, Grid_Temp,
                                                  1 - Grid_xHII) / 4 / np.pi  # We divide by 4pi to go to sr**-1 units
                else:
                    print('--- NOT Salpha fluctuations in dTb ---')
                    Grid_xal = Grid_xal * S_alpha(z, np.mean(Grid_Temp), 1 - np.mean(Grid_xHII)) / 4 / np.pi


            if Rsmoothing > 0:
                Grid_xal  = smooth_field(Grid_xal, Rsmoothing, LBox, nGrid)
                Grid_Temp = smooth_field(Grid_Temp, Rsmoothing, LBox, nGrid)
                #Grid_xHII = smooth_field(Grid_xHII, Rsmoothing, LBox, nGrid)
                #delta_b   = smooth_field(delta_b, Rsmoothing, LBox, nGrid)

            if dTb:

                if xcoll:
                    print('--- Including xcoll fluctuations in dTb ---')
                    Grid_xcoll = x_coll(z=z, Tk=Grid_Temp, xHI=(1 - Grid_xHII), rho_b=(delta_b + 1) * coef)
                    xcoll_mean = np.mean(Grid_xcoll)
                    Grid_xtot = Grid_xcoll + Grid_xal
                    del Grid_xcoll
                else:
                    print('--- NOT including xcoll fluctuations in dTb ---')

                    xcoll_mean = x_coll(z=z, Tk=np.mean(Grid_Temp), xHI=(1 - np.mean(Grid_xHII)), rho_b= coef)
                    Grid_xtot = Grid_xal + xcoll_mean

                Grid_dTb = dTb_fct(z=z, Tk=Grid_Temp, xtot=Grid_xtot, delta_b=delta_b, x_HII=Grid_xHII, param=param)


            else :
                Grid_dTb = np.array([0])
                xcoll_mean = 0

    PS_dTb, k_bins = auto_PS(Grid_dTb / np.mean(Grid_dTb) - 1, box_dims=LBox,
                             kbins=def_k_bins(param))
    print('XAL MEAN APPROX GIVES :',xal_mean * S_alpha(z, np.mean(Grid_Temp), 1 - np.mean(Grid_xHII)) / 4 / np.pi)
    print('XAL MEAN APPROX WITHOUT SALPHA GIVES :',xal_mean / 4 / np.pi)

    if not RSD:
        dTb_RSD_mean = 0
        PS_dTb_RSD = 0
    else:
        print('Computing RSD for snapshot...')
        Grid_dTb_RSD = dTb_RSD(param, z, delta_b, Grid_dTb)
        delta_Grid_dTb_RSD = Grid_dTb_RSD / np.mean(Grid_dTb_RSD) - 1
        PS_dTb_RSD = \
            auto_PS(delta_Grid_dTb_RSD, box_dims=LBox, kbins=def_k_bins(param))[0]
        dTb_RSD_mean = np.mean(Grid_dTb_RSD)

        ##

    GS_PS_dict = {'z': z, 'dTb': np.mean(Grid_dTb), 'Tk': np.mean(Grid_Temp), 'x_HII': np.mean(Grid_xHII),
                  'PS_dTb': PS_dTb, 'k': k_bins,
                  'PS_dTb_RSD': PS_dTb_RSD, 'dTb_RSD': dTb_RSD_mean, 'x_al': np.mean(Grid_xal),
                  'x_coll': xcoll_mean}
    if cross_corr:
        GS_PS_dict = compute_cross_correlations(param, GS_PS_dict, Grid_Temp, Grid_xHII, Grid_xal,Grid_dTb, delta_b,
                                                third_order=third_order,fourth_order=fourth_order,truncate=truncate)
    save_f(file='./physics/GS_PS_' + str(param.sim.Ncell) + '_' + param.sim.model_name + '_z' + z_str, obj=GS_PS_dict)

    if variance:
        import copy
        param_copy = copy.deepcopy(
            param)  # we do this since in compute_var we change the kbins to go to smaller scales.
        compute_var_single_z(param_copy, z, Grid_xal, Grid_xHII, Grid_Temp, k_bins)

    if param.sim.store_grids:
        if temp:
            save_grid(param, z=z, grid=Grid_Temp, type='Tk')
        if ion:
            save_grid(param, z=z, grid=Grid_xHII, type='bubbles')
        if lyal:
            save_grid(param, z=z, grid=Grid_xal, type='lyal')
        if dTb:
            if not RSD:
                save_grid(param, z=z, grid=Grid_dTb, type='dTb')
                # save_f(file='./grid_output/dTb_Grid' + str(nGrid) + model_name + '_snap' + z_str, obj=Grid_dTb)
            else:
                save_grid(param, z=z, grid=Grid_dTb_RSD, type='dTb')










def paint_boxes_HM(param, PS_HM, temp=True, lyal=True, ion=True, dTb=True, read_temp=False, read_ion=False, read_lyal=False,
                check_exists=True, RSD=True, xcoll=True, S_al=True, cross_corr=False, third_order=False, fourth_order=False,cic=False,
                variance=False,Rsmoothing=0,truncate=False):
    """
    Parameters
    ----------
    RSD
    check_exists : object
    param : dictionnary containing all the input parameters

    Returns
    -------
    Does not return anything. Loop over all snapshots in param.sim.halo_catalogs and calls paint_profile_single_snap.
    """

    start_time = time.time()
    LBox = param.sim.Lbox  # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells
    catalog_dir = param.sim.halo_catalogs
    model_name = param.sim.model_name

    if catalog_dir is None:
        print('You should specify param.sim.halo_catalogs. Should be a file containing the rockstar halo catalogs.')

    print('Painting profiles on a grid with', nGrid, 'pixels per dim. Box size is', LBox, 'cMpc/h.')

    if param.sim.cores > 1:
        import mpi4py.MPI
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()
    else:
        rank = 0
        size = 1

    z_arr = def_redshifts(param)
    for ii, z in enumerate(z_arr):
        z_str = z_string_format(z)
        z = np.round(z, 2)
        if rank == ii % size:
            print('Core nbr', rank, 'is taking care of z = ', z)
            if check_exists:
                if os.path.exists('./grid_output/xHII_' + str(nGrid) + '_' + model_name + '_z' + z_str):
                    print('xHII map for z = ', z, 'already painted. Skipping.')
                else:
                    print('----- Painting 3D map for z =', z, '-------')
                    paint_profile_single_snap_HM_input(z_str, param,PS_HM, temp=temp, lyal=lyal, ion=ion, dTb=dTb, read_temp=read_temp,
                                              read_ion=read_ion, read_lyal=read_lyal, RSD=RSD, xcoll=xcoll, S_al=S_al,
                                              cross_corr=cross_corr, third_order=third_order,fourth_order=fourth_order, cic=cic,
                                              variance=variance,Rsmoothing=Rsmoothing,truncate=truncate)
                    print('----- Snapshot at z = ', z, ' is done -------')
                    print(' ')
            else:
                print('----- Painting 3D map for z =', z, '-------')
                paint_profile_single_snap_HM_input(z_str, param,PS_HM, temp=temp, lyal=lyal, ion=ion, dTb=dTb, read_temp=read_temp,
                                          read_ion=read_ion, read_lyal=read_lyal, RSD=RSD, xcoll=xcoll, S_al=S_al,
                                          cross_corr=cross_corr, third_order=third_order, fourth_order=fourth_order,cic=cic, variance=variance,Rsmoothing=Rsmoothing,truncate=truncate)
                print('----- Snapshot at z = ', z, ' is done -------')
                print(' ')

    print('Finished painting the maps. They are stored in ./grid_output. It took in total: ' + print_time(
        time.time() - start_time) +
          ' to paint the grids.')
    print('  ')

