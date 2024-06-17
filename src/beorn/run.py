"""
In this script we define functions that can be called to :
1. run the RT solver and compute the evolution of the T, x_HI profiles, and store them
2. paint the profiles on a grid.
"""
import beorn as rad
from scipy.interpolate import splrep, splev, interp1d
import numpy as np
import time
from .constants import cm_per_Mpc, M_sun, m_H
from .cosmo import D, hubble, T_adiab_fluctu, dTb_fct, T_cmb
import os
from .profiles_on_grid import profile_to_3Dkernel, Spreading_Excess_Fast, put_profiles_group, stacked_lyal_kernel, \
    stacked_T_kernel, cumulated_number_halos, average_profile, log_binning, bin_edges_log
from .couplings import x_coll, S_alpha, x_coll_coef
from .global_qty import xHII_approx, compute_glob_qty
from os.path import exists
import tools21cm as t2c
import scipy
from .cosmo import dTb_factor, Tspin_fct
from .functions import *


def run_code(param, compute_profile=True, temp=True, lyal=True, ion=True, dTb=True, read_temp=False, read_ion=False,
             read_lyal=False,
             check_exists=True, RSD=True, xcoll=True, S_al=True, cross_corr=False, third_order=False, cic=False,
             variance=False, compute_corr_fct_=False,Rsmoothing=0, GS = False
             ):
    """
    Function to run the code. Several options are available.

    Parameters
    ----------
    param : BEoRN dictionnary containing all the input parameters

    Returns
    ------- Does not return anything. However, it solves the RT equation for a range of halo masses,
    following their evolution from cosmic dawn to the end of reionization. It stores the profile in a directory "./profiles"
    """


    comm, rank, size = initialise_mpi4py(param)

    if rank == 0:
        print(' ------------ BEORN STARTS ------------ ')
        if variance:
            if not os.path.isdir('./variances'):
                os.mkdir('./variances')
    Barrier(comm)

    if compute_profile:
        if rank == 0:
            compute_profiles(param)
        Barrier(comm)

    if rank == 0:
        print(' ------------ PAINTING BOXES ------------ ')
    Barrier(comm)

    paint_boxes(param, temp=temp, lyal=lyal, ion=ion, dTb=dTb, read_temp=read_temp, check_exists=check_exists,
                read_ion=read_ion, read_lyal=read_lyal, RSD=RSD, xcoll=xcoll, S_al=S_al,
                cross_corr=cross_corr, third_order=third_order, cic=cic, variance=variance,Rsmoothing=Rsmoothing)

    Barrier(comm)

    if rank == 0:
        print(' ------------ MERGING PS FILES  ------------ ')
        gather_files(param, path='./physics/GS_PS_', z_arr = def_redshifts(param), Ncell=param.sim.Ncell,remove=True)
        #gather_GS_PS_files(param, remove=True)

    if variance:
        if rank == 1 % size:
            print(' ------------ COMPUTING VARIANCES OF SMOOTHED FIELDS LYA, T, XHII  ------------ ')
            gather_variances(param)

    if GS and rank == 2 % size:
        print(' ------------ COMPUTING GLOBAL QUANTITIES FROM PROFILES AND HALO CATALOGS  ------------ ')
        GS = compute_glob_qty(param)
        save_f(file='./physics/GS_approx' + '_' + param.sim.model_name + '.pkl', obj=GS)

    if compute_corr_fct_:  ## compute the correlation function at r=0 (variance of unsmoothed fields). We use this for the correction to GS.
        if rank == 3 % size:
            print(' ------------ COMPUTING CORRELATION FUNCTIONS AT R=0  ------------ ')
            compute_corr_fct(param)

    Barrier(comm)
    if rank == 0:
        print(' ------------ DONE ------------ ')

    return None


def compute_profiles(param):
    """
    This function computes the Temperature, Lyman-alpha, and ionisation fraction profiles that will then be painted on 3D maps.
    It calls profiles from compute_profiles.py

    Parameters
    ----------
    param : BEoRN dictionnary containing all the input parameters

    Returns ------- Does not return anything. However, it solves the RT equation for a range of halo masses,
    following their evolution from cosmic dawn to the end of reionization. It stores the profile in a directory "./profiles"
    """
    start_time = time.time()

    print('Computing Temperature (Tk), Lyman-α and ionisation fraction (xHII) profiles...')
    if not os.path.isdir('./profiles'):
        os.mkdir('./profiles')

    if not os.path.isdir('./grid_output'):
        os.mkdir('./grid_output')

    if not os.path.isdir('./physics'):
        os.mkdir('./physics')

    model_name = param.sim.model_name
    pkl_name = './profiles/' + model_name + '.pkl'
    grid_model = rad.profiles(param)
    grid_model.solve(param)
    pickle.dump(file=open(pkl_name, 'wb'), obj=grid_model)
    print('...  Profiles stored in dir ./profiles.')
    print(' ')
    end_time = time.time()
    print('It took ' + print_time(end_time - start_time) + ' to compute the profiles.')


def paint_profile_single_snap(z_str, param, temp=True, lyal=True, ion=True, dTb=True, read_temp=False, read_ion=False,
                              read_lyal=False, RSD=False, xcoll=True, S_al=True, cross_corr=False, third_order=False,fourth_order=False,
                              cic=False, variance=False,Rsmoothing=0, truncate=False):
    """
    Paint the Tk, xHII and Lyman alpha profiles on a grid for a single halo catalog named filename.

    Parameters
    ----------
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
    M_Bin = np.logspace(np.log10(param.sim.Mh_bin_min), np.log10(param.sim.Mh_bin_max), param.sim.binn, base=10)

    LBox = param.sim.Lbox  # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells

    try: halo_catalog = load_halo(param, z_str)
    except: halo_catalog = param.sim.halo_catalogs[float(z_str)] #param.sim.halo_catalogs[z_str]
    H_Masses, H_X, H_Y, H_Z, z = halo_catalog['M'], halo_catalog['X'], halo_catalog['Y'], halo_catalog['Z'], \
                                 halo_catalog['z']

    ### To later add up the adiabatic Tk fluctuations at the grid level.
    if temp or dTb:
        try: delta_b = load_delta_b(param, z_str)  # rho/rhomean-1
        except: delta_b = param.sim.dens_fields[float(z_str)] #param.sim.dens_fields[z_str]
    else:
        delta_b = np.array([0])

    Om, Ob, h0 = param.cosmo.Om, param.cosmo.Ob, param.cosmo.h
    factor = dTb_factor(param)
    coef = rhoc0 * h0 ** 2 * Ob * (1 + z) ** 3 * M_sun / cm_per_Mpc ** 3 / m_H

    # find matching redshift between solver output and simulation snapshot.
    grid_model = load_f(file='./profiles/' + model_name + '.pkl')
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]

    # if H_Masses is digitized to bin_edges_log[i] it means it should take the value of M_Bin[i-1] (bin 0 is what's on the left...)
    # M_Bin                0.   1.    2.    3.  ...
    # bin_edges_log     0.  | 1. |  2. |  3. |  4. ....
    Indexing = log_binning(H_Masses, bin_edges_log(grid_model.Mh_history[ind_z, :]))
    Indexing = Indexing - 1
    ### same result as if you do np.argmin(np.abs(np.log10(H_Masses[:,None]/grid_model.Mh_history[ind_z, :]),axis=1), but faster

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
        Grid_dTb_no_reio = factor * np.sqrt(1 + z) * (1 - Tcmb0 * (1 + z) / Grid_Temp) * (delta_b + 1) *\
                           Grid_xcoll / (1 + Grid_xcoll)
        Grid_dTb_RSD = np.array([0])
        Grid_dTb_T_sat = factor * np.sqrt(1 + z) * (1 - Grid_xHII) * (delta_b + 1) * Grid_xcoll / (1 + Grid_xcoll)
        xcoll_mean = np.mean(Grid_xcoll)
        T_spin = np.mean(Tspin_fct(Tcmb0 * (1 + z), Grid_Temp, Grid_xcoll))
        del Grid_xcoll

    else:
        if np.max(H_Masses) > np.max(grid_model.Mh_history[ind_z, :]):
            print('Max Mh_bin is :', np.max(grid_model.Mh_history[ind_z, :]), 'while the largest halo in catalog is',
                  np.max(H_Masses))
            print('WARNING!!! You should use a larger value for param.sim.Mh_bin_max')
        if np.min(H_Masses) < np.min(grid_model.Mh_history[ind_z, :]):
            print('WARNING!!! You should use a smaller value for param.sim.Mh_bin_min')

        Ionized_vol = xHII_approx(param, halo_catalog)[1]
        print('Quick calculation from the profiles predicts xHII = ', round(Ionized_vol, 4))
        if Ionized_vol > 1:
            Grid_xHII = np.array([1])
            Grid_Temp = np.array([1])
            Grid_dTb = np.array([0])
            Grid_dTb_no_reio = np.array([0])
            Grid_dTb_T_sat = np.array([0])
            Grid_xal = np.array([0])
            print('universe is fully inoinzed. Return [1] for the xHII, T and [0] for dTb.')
        else:
            if not read_temp or not read_lyal or not read_ion:

                Pos_Halos_Grid = pixel_position(H_X, H_Y, H_Z,LBox,nGrid)  # we don't want Pos_Halos_Grid==nGrid. This only happens if Pos_Bubbles=LBox

                Grid_xHII_i = np.zeros((nGrid, nGrid, nGrid))
                Grid_Temp = np.zeros((nGrid, nGrid, nGrid))
                Grid_xal = np.zeros((nGrid, nGrid, nGrid))


                for i in range(len(M_Bin)):
                    indices = np.where(Indexing == i)[0]  ## indices in H_Masses of halos that have an initial mass at z=z_start between M_Bin[i-1] and M_Bin[i]
                    Mh_ = grid_model.Mh_history[ind_z, i]

                    if len(indices) > 0 and Mh_ > param.source.M_min:
                        radial_grid = grid_model.r_grid_cell / (1 + zgrid)  # pMpc/h
                        x_HII_profile = np.zeros((len(radial_grid)))

                        R_bubble, rho_alpha_, Temp_profile = average_profile(param, grid_model, H_Masses[indices],ind_z, i)
                        x_HII_profile[np.where(radial_grid < R_bubble / (1 + zgrid))] = 1  # grid_model.R_bubble[ind_z, i]
                        # Temp_profile = grid_model.rho_heat[ind_z, :, i]


                        r_lyal = grid_model.r_lyal  # np.logspace(-5, 2, 1000, base=10)     ##    physical distance for lyal profile. Never goes further away than 100 pMpc/h (checked)
                        # rho_alpha_ = grid_model.rho_alpha[ind_z, :, i]  # rho_alpha(r_lyal, Mh_, zgrid, param)[0]
                        x_alpha_prof = 1.81e11 * (rho_alpha_) / (1 + zgrid)  # We add up S_alpha(zgrid, T_extrap, 1 - xHII_extrap) later, a the map level.

                        ### This is the position of halos in base "nGrid". We use this to speed up the code.
                        ### We count with np.unique the number of halos in each cell. Then we do not have to loop over halo positions in --> profiles_on_grid/put_profiles_group
                        # base_nGrid_position = Pos_Halos_Grid[indices][:, 0] + nGrid * Pos_Halos_Grid[indices][:,1] + nGrid ** 2 * Pos_Halos_Grid[ indices][:,2]
                        # unique_base_nGrid_poz, nbr_of_halos = np.unique(base_nGrid_position, return_counts=True)

                        unique_base_nGrid_poz, nbr_of_halos = cumulated_number_halos(param, H_X[indices], H_Y[indices], H_Z[indices], cic=cic)

                        ZZ_indice = unique_base_nGrid_poz // (nGrid ** 2)
                        YY_indice = (unique_base_nGrid_poz - ZZ_indice * nGrid ** 2) // nGrid
                        XX_indice = (unique_base_nGrid_poz - ZZ_indice * nGrid ** 2 - YY_indice * nGrid)

                        ## Every halos in mass bin i are assumed to have mass M_bin[i].
                        if ion:
                            profile_xHII = interp1d(radial_grid * (1 + z), x_HII_profile, bounds_error=False,
                                                    fill_value=(1, 0))
                            kernel_xHII = profile_to_3Dkernel(profile_xHII, nGrid, LBox)
                            if not np.any(kernel_xHII > 0):
                                ### if the bubble volume is smaller than the grid size,we paint central cell with ion fraction value
                                # kernel_xHII[int(nGrid / 2), int(nGrid / 2), int(nGrid / 2)] = np.trapz(x_HII_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / nGrid / (1 + z)) ** 3
                                Grid_xHII_i[XX_indice, YY_indice, ZZ_indice] += np.trapz(
                                    x_HII_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / nGrid / (
                                        1 + z)) ** 3 * nbr_of_halos

                            else:
                                renorm = np.trapz(x_HII_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (
                                        LBox / (1 + z)) ** 3 / np.mean(kernel_xHII)
                                # extra_ion = put_profiles_group(Pos_Halos_Grid[indices], kernel_xHII * 1e-7 / np.sum(kernel_xHII)) * np.sum(kernel_xHII) / 1e-7 * renorm
                                Grid_xHII_i += put_profiles_group(np.array((XX_indice, YY_indice, ZZ_indice)),
                                                                  nbr_of_halos,
                                                                  kernel_xHII * 1e-7 / np.sum(kernel_xHII)) * np.sum(
                                    kernel_xHII) / 1e-7 * renorm
                                # bubble_volume = np.trapz(4 * np.pi * radial_grid ** 2 * x_HII_profile, radial_grid)
                                # print('bubble volume is ', len(indices) * bubble_volume,'pMpc, grid volume is', np.sum(extra_ion)* (LBox /nGrid/ (1 + z)) ** 3 )
                                # Grid_xHII_i += extra_ion

                            # fill in empty pixels with the min xHII
                            Grid_xHII_i[Grid_xHII_i<param.source.min_xHII] = param.source.min_xHII

                            del kernel_xHII
                        if lyal:
                            ### We use this stacked_kernel functions to impose periodic boundary conditions when the lyal or T profiles extend outside the box size. Very important for Lyman-a.
                            if isinstance(truncate, float):
                                # truncate below a certain radius
                                x_alpha_prof[r_lyal * (1 + z)< truncate] = x_alpha_prof[r_lyal * (1 + z)< truncate][-1]

                            kernel_xal = stacked_lyal_kernel(r_lyal * (1 + z), x_alpha_prof, LBox, nGrid,
                                                             nGrid_min=param.sim.nGrid_min_lyal)
                            renorm = np.trapz(x_alpha_prof * 4 * np.pi * r_lyal ** 2, r_lyal) / (
                                    LBox / (1 + z)) ** 3 / np.mean(kernel_xal)
                            if np.any(kernel_xal > 0):
                                # Grid_xal += put_profiles_group(Pos_Halos_Grid[indices], kernel_xal * 1e-7 / np.sum(kernel_xal)) * renorm * np.sum( kernel_xal) / 1e-7  # we do this trick to avoid error from the fft when np.sum(kernel) is too close to zero.
                                Grid_xal += put_profiles_group(np.array((XX_indice, YY_indice, ZZ_indice)),
                                                               nbr_of_halos,
                                                               kernel_xal * 1e-7 / np.sum(
                                                                   kernel_xal)) * renorm * np.sum(
                                    kernel_xal) / 1e-7  # we do this trick to avoid error from the fft when np.sum(kernel) is too close to zero.
                            del kernel_xal

                        if temp:
                            if isinstance(truncate, float):
                                # truncate below a certain radius
                                Temp_profile[radial_grid * (1 + z)< truncate] = Temp_profile[radial_grid * (1 + z)< truncate][-1]
                            kernel_T = stacked_T_kernel(radial_grid * (1 + z), Temp_profile, LBox, nGrid,
                                                        nGrid_min=param.sim.nGrid_min_heat)
                            renorm = np.trapz(Temp_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (
                                    LBox / (1 + z)) ** 3 / np.mean(kernel_T)
                            if np.any(kernel_T > 0):
                                # Grid_Temp += put_profiles_group(Pos_Halos_Grid[indices],  kernel_T * 1e-7 / np.sum(kernel_T)) * np.sum(kernel_T) / 1e-7 * renorm
                                Grid_Temp += put_profiles_group(np.array((XX_indice, YY_indice, ZZ_indice)),
                                                                nbr_of_halos,
                                                                kernel_T * 1e-7 / np.sum(kernel_T)) * np.sum(
                                    kernel_T) / 1e-7 * renorm
                            del kernel_T

                        end_time = time.time()
                        print(len(indices), 'halos in mass bin ', i,
                              '. It took ' + print_time(end_time - start_time) + ' to paint the profiles.')
                print('.... Done painting profiles. ')

                print('Dealing with the overlap of ionised bubbles.... ')
                # Grid_Storage = np.copy(Grid_xHII_i)

                t_start_spreading = time.time()
                if np.sum(Grid_xHII_i) < nGrid ** 3 and ion:
                    Grid_xHII = Spreading_Excess_Fast(param, Grid_xHII_i)
                else:
                    Grid_xHII = np.array([1])

                print('.... Done. It took:', print_time(time.time() - t_start_spreading),
                      'to redistribute excess photons from the overlapping regions.')

                if np.all(Grid_xHII == 0):
                    Grid_xHII = np.array([0])
                if np.all(Grid_xHII == 1):
                    print('universe is fully inoinzed. Return [1] for Grid_xHII.')
                    Grid_xHII = np.array([1])

                Grid_Temp += T_adiab_fluctu(z, param, delta_b)

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


            if xcoll:
                print('--- Including xcoll fluctuations in dTb ---')
                Grid_xcoll = x_coll(z=z, Tk=Grid_Temp, xHI=(1 - Grid_xHII), rho_b=(delta_b + 1) * coef)
                xcoll_mean = np.mean(Grid_xcoll)
                Grid_xtot = Grid_xcoll + Grid_xal
                del Grid_xcoll
            else:
                print('--- NOT including xcoll fluctuations in dTb ---')
                xcoll_mean = x_coll(z=z, Tk=np.mean(Grid_Temp), xHI=(1 - np.mean(Grid_xHII)), rho_b=coef)
                Grid_xtot = Grid_xal + xcoll_mean


            if dTb:
                Grid_dTb = dTb_fct(z=z, Tk=Grid_Temp, xtot=Grid_xtot, delta_b=delta_b, x_HII=Grid_xHII, param=param)
                Grid_dTb_no_reio = dTb_fct(z=z, Tk=Grid_Temp, xtot=Grid_xtot, delta_b=delta_b, x_HII=np.array([0]),param=param)

                Grid_dTb_T_sat = dTb_fct(z=z, Tk=1e50, xtot = 1e50, delta_b=delta_b, x_HII=Grid_xHII, param=param)

            else :
                Grid_dTb = np.array([0])
                Grid_dTb_no_reio = np.array([0])
                Grid_dTb_T_sat = np.array([0])

        T_spin = np.mean(Tspin_fct(Tcmb0 * (1 + z), Grid_Temp, Grid_xtot))

    PS_dTb, k_bins = auto_PS(delta_fct(Grid_dTb), box_dims=LBox,
                             kbins=def_k_bins(param))
    PS_dTb_no_reio, k_bins = auto_PS(delta_fct(Grid_dTb_no_reio), box_dims=LBox,
                                     kbins=def_k_bins(param))
    PS_dTb_T_sat, k_bins = auto_PS(delta_fct(Grid_dTb_T_sat), box_dims=LBox,
                                     kbins=def_k_bins(param))


    if not RSD:
        dTb_RSD_mean = 0
        PS_dTb_RSD = 0
        Grid_dTb_RSD = Grid_dTb
    else:
        print('Computing RSD for snapshot...')
        Grid_dTb_RSD = dTb_RSD(param, z, delta_b, Grid_dTb)
        delta_Grid_dTb_RSD = Grid_dTb_RSD / np.mean(Grid_dTb_RSD) - 1
        PS_dTb_RSD = \
            auto_PS(delta_Grid_dTb_RSD, box_dims=LBox, kbins=def_k_bins(param))[0]
        dTb_RSD_mean = np.mean(Grid_dTb_RSD)



        ##

    GS_PS_dict = {'z': z, 'dTb': np.mean(Grid_dTb), 'Tk': np.mean(Grid_Temp), 'x_HII': np.mean(Grid_xHII),
                  'PS_dTb': PS_dTb, 'k': k_bins, 'Tspin':T_spin,
                  'PS_dTb_RSD': PS_dTb_RSD, 'dTb_RSD': dTb_RSD_mean, 'x_al': np.mean(Grid_xal),
                  'x_coll': xcoll_mean,'PS_dTb_no_reio':PS_dTb_no_reio,'dTb_no_reio': np.mean(Grid_dTb_no_reio),
                  'PS_dTb_T_sat':PS_dTb_T_sat,'dTb_T_sat': np.mean(Grid_dTb_T_sat)}
    if cross_corr:
        GS_PS_dict = compute_cross_correlations(param, GS_PS_dict, Grid_Temp, Grid_xHII, Grid_xal,Grid_dTb, delta_b,
                                                third_order=third_order,fourth_order=fourth_order,truncate=truncate)



    save_f(file='./physics/GS_PS_' + str(param.sim.Ncell) + '_' + param.sim.model_name + '_' + z_str+ '.pkl', obj=GS_PS_dict)

    if variance:
        import copy
        param_copy = copy.deepcopy(
            param)  # we do this since in compute_var we change the kbins to go to smaller scales.
        compute_var_single_z(param_copy, z, Grid_xal, Grid_xHII, Grid_Temp, k_bins)

    if param.sim.store_grids is not False:
        if 'Tk' in param.sim.store_grids:
            save_grid(param, z=z, grid=Grid_Temp, type='Tk')
        if 'bubbles' in param.sim.store_grids:
            save_grid(param, z=z, grid=Grid_xHII, type='bubbles')
        if 'lyal' in param.sim.store_grids:
            save_grid(param, z=z, grid=Grid_xal, type='lyal')
        if 'dTb' in param.sim.store_grids:
            save_grid(param, z=z, grid=Grid_dTb_RSD, type='dTb')


def gather_GS_PS_files(param, remove=False):
    """
    Reads in ./physics/GS_PS_... files, gather them into a single file and suppress them.

    Parameters
    ----------
    param : dictionnary containing all the input parameters

    Returns
    -------
    Nothing.
    """
    gather_files(param, path='./physics/GS_PS_', z_arr=def_redshifts(param), Ncell=param.sim.Ncell, remove=remove)


def paint_boxes(param, temp=True, lyal=True, ion=True, dTb=True, read_temp=False, read_ion=False, read_lyal=False,
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
        print('You should specify param.sim.halo_catalogs. Should be a file containing the halo catalogs.')

    print('Painting profiles on a grid with', nGrid, 'pixels per dim. Box size is', LBox, 'cMpc/h.')

    comm, rank, size = initialise_mpi4py(param)

    z_arr = def_redshifts(param)
    for ii, z in enumerate(z_arr):
        z_str = z_string_format(z)
        z = np.round(z, 2)
        if rank == ii % size:
            print('Core nbr', rank, 'is taking care of z = ', z)
            if check_exists and exists('./grid_output/dTb_' + str(nGrid) + '_' + model_name + '_z' + z_str):
                    print('dTb map for z = ', z, 'already painted. Skipping.')
            else:
                print('----- Painting 3D map for z =', z, '-------')
                paint_profile_single_snap(z_str, param, temp=temp, lyal=lyal, ion=ion, dTb=dTb, read_temp=read_temp,
                                          read_ion=read_ion, read_lyal=read_lyal, RSD=RSD, xcoll=xcoll, S_al=S_al,
                                          cross_corr=cross_corr, third_order=third_order, fourth_order=fourth_order,cic=cic, variance=variance,Rsmoothing=Rsmoothing,truncate=truncate)
                print('----- Snapshot at z = ', z, ' is done -------')
                print(' ')

    print('Finished painting the maps. They are stored in ./grid_output. It took in total: ' + print_time(
        time.time() - start_time) +
          ' to paint the grids.')
    print('  ')


def grid_dTb(param, ion='bubbles', RSD=False):
    """
    Creates a grid of xcoll and dTb. Needs to read in Tk grid, xHII grid and density field on grid.
    """
    nGrid, LBox = param.sim.Ncell, param.sim.Lbox
    Om, Ob, h0 = param.cosmo.Om, param.cosmo.Ob, param.cosmo.h
    factor = dTb_factor(param)  # factor used in dTb calculation

    comm, rank, size = initialise_mpi4py(param)

    z_arr = def_redshifts(param)
    for ii, z in enumerate(z_arr):
        z_str = z_string_format(z)
        zz_ = z
        if rank == ii % size:
            Grid_Temp = load_grid(param, z, type='Tk')
            Grid_xHII = load_grid(param, z, type=ion)
            Grid_xal = load_grid(param, z, type='lyal')

            dens_field = param.sim.dens_field
            if dens_field is not None:
                delta_b = load_delta_b(param, z_string_format(z))
            else:
                delta_b = 0  # rho/rhomean -1

            T_cmb_z = Tcmb0 * (1 + zz_)
            Grid_xHI = 1 - Grid_xHII  ### neutral fraction

            # Grid_Sal = S_alpha(zz_, Grid_Temp, 1 - Grid_xHII)
            Grid_xal = Grid_xal  # * Grid_Sal
            coef = x_coll_coef(z,param)
            Grid_xcoll = x_coll(z=zz_, Tk=Grid_Temp, xHI=Grid_xHI, rho_b=(delta_b + 1) * coef)
            Grid_Tspin = ((1 / T_cmb_z + (Grid_xcoll + Grid_xal) / Grid_Temp) / (1 + Grid_xcoll + Grid_xal)) ** -1

            # Grid_Tspin = ((1 / T_cmb_z + (Grid_xcoll+Grid_xal) / Grid_Temp) / (1 + Grid_xcoll+Grid_xal)) ** -1

            Grid_dTb = factor * np.sqrt(1 + zz_) * (1 - T_cmb_z / Grid_Tspin) * Grid_xHI * (delta_b + 1)
            PS_dTb, k_bins = auto_PS(Grid_dTb / np.mean(Grid_dTb) - 1, box_dims=LBox,
                                     kbins=def_k_bins(param))

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

            GS_PS_dict = {'z': zz_, 'dTb': np.mean(Grid_dTb), 'Tk': np.mean(Grid_Temp), 'x_HII': np.mean(Grid_xHII),
                          'PS_dTb': PS_dTb, 'k': k_bins,
                          'PS_dTb_RSD': PS_dTb_RSD, 'dTb_RSD': dTb_RSD_mean}

            save_f(file='./physics/GS_PS_' + z_str, obj=GS_PS_dict)

            # Grid_dTb = factor * np.sqrt(1 + zz_) * (1 - T_cmb_z / Grid_Tspin) * Grid_xHI * (delta_b+1)  #* Grid_xtot / (1 + Grid_xtot)
            # pickle.dump(file=open('./grid_output/Tspin_Grid' + str(nGrid)+ model_name + '_snap' + filename[4:-5], 'wb'),obj=Grid_Tspin)
            # save_f(file='./grid_output/dTb_Grid' + str(nGrid) + model_name + '_snap' + z_str, obj=Grid_dTb)
            save_grid(param, z=z, grid=Grid_dTb, type='dTb')


def compute_GS(param, string='', RSD=False, ion='bubbles'):
    """
    Reads in the grids and compute the global quantities averaged.
    If RSD is True, will add RSD calculation
    If lyal_from_sfrd is True, we will compute xalpha from the sfrd (see eq 19. and 23. from HM paper 2011.12308) and then correct dTb to match this xalpha.
    If ion = exc_set, will read in the xHII maps produce from the excursion set formalism.
    """
    model_name = param.sim.model_name
    nGrid = param.sim.Ncell
    Tadiab = []
    z_ = []
    Tk = []
    dTb = []
    x_HII = []
    x_al = []
    # beta_a = []
    # beta_T = []
    # beta_r = []
    dTb_RSD_arr = []
    z_arr = def_redshifts(param)
    for ii, z in enumerate(z_arr):
        zz_ = z
        Grid_xHII = load_grid(param, z, type=ion)
        Grid_Temp = load_grid(param, z, type='Tk')
        # Grid_xtot_ov       = pickle.load(file=open('./grid_output/xtot_ov_Grid' + str(nGrid)  + model_name + '_snap' + z_str, 'rb'))
        Grid_dTb = load_grid(param, z, type='dTb')
        Grid_xal = load_grid(param, z, type='lyal')

        xal_ = np.mean(Grid_xal)

        z_.append(zz_)
        Tk.append(np.mean(Grid_Temp))
        # Tk_neutral.append(np.mean(Grid_Temp[np.where(Grid_xHII < param.sim.thresh_xHII)]))

        # T_spin.append(np.mean(Grid_Tspin[np.where(Grid_xHII < param.sim.thresh_xHII)]))
        x_HII.append(np.mean(Grid_xHII))
        x_al.append(xal_)
        # x_coll.append(xcol_)
        dTb.append(np.mean(Grid_dTb))
        # beta_a.append(xal_ / (xcol_ + xal_) / (1 + xcol_ + xal_))
        # beta_T.append(Tcmb /(Tk[ii]-Tcmb))
        # beta_r.append(-x_HII[ii] / (1 - x_HII[ii]))

        Tadiab.append(Tcmb0 * (1 + zz_) ** 2 / (1 + param.cosmo.z_decoupl))

        if RSD:
            delta_b = load_delta_b(param, z_string_format(zz_))
            dTb_RSD_arr.append(np.mean(dTb_RSD(param, zz_, delta_b, Grid_dTb)))
        else:
            dTb_RSD_arr.append(0)

    z_, Tk, x_HII, x_al, Tadiab, dTb, dTb_RSD_arr = np.array(z_), np.array(Tk), np.array(x_HII), np.array(
        x_al), np.array(Tadiab), np.array(dTb), np.array(dTb_RSD_arr)

    matrice = np.array([z_, Tk, x_HII, x_al, Tadiab, dTb, dTb_RSD_arr])
    z_, Tk, x_HII, x_al, Tadiab, dTb, dTb_RSD_arr = matrice[:, matrice[0].argsort()]  ## sort according to z_

    Tgam = (1 + z_) * Tcmb0
    T_spin = ((1 / Tgam + x_al / Tk) / (1 + x_al)) ** -1

    #### Here we compute Jalpha using HM formula. It is more precise since it accounts for halos at high redshift that mergerd and are not present at low redshift.
    # dTb_GS = factor * np.sqrt(1 + z_) * (1 - Tcmb0 * (1 + z_) / Tk) * (1-x_HII) * (x_coll + x_al) / (1 + x_coll + x_al)### dTb formula similar to coda HM code
    # dTb_GS_Tkneutral = factor * np.sqrt(1 + z_) * (1 - Tcmb0 * (1 + z_) / Tk_neutral) * (1-x_HII) * (x_coll + x_al) / (1 + x_coll + x_al)
    # beta_a = (x_al / (x_coll + x_al) / (1 + x_coll + x_al))
    # xtot = x_al + x_coll
    # dTb_GS_Tkneutral = dTb_GS_Tkneutral * xtot / (1 + xtot) / ((x_coll + x_al) / (1 + x_coll + x_al))

    Dict = {'Tk': Tk, 'x_HII': x_HII, 'x_al': x_al, 'dTb': dTb, 'dTb_RSD': dTb_RSD_arr, 'Tadiab': Tadiab, 'z': z_,
            'T_spin': T_spin}

    save_f(file='./physics/GS_' + string + str(nGrid) + model_name + '.pkl', obj=Dict)




def compute_cross_correlations(param, GS_PS_dict, Grid_Temp, Grid_xHII, Grid_xal, Grid_dTb, delta_rho, third_order=False, fourth_order=False, truncate = False):
    nGrid = param.sim.Ncell
    Lbox = param.sim.Lbox  # Mpc/h

    print('Computing Power Spectra with all cross correlations.')
    if third_order:
        print('...including third order perturbations in delta_reio.')

    kbins = def_k_bins(param)

    if Grid_Temp.size == 1:  ## to avoid error when measuring power spectrum
        Grid_Temp = np.full((nGrid, nGrid, nGrid), 1)
    if Grid_xHII.size == 1:
        Grid_xHII = np.full((nGrid, nGrid, nGrid), 0)  ## to avoid div by zero
    if Grid_xal.size == 1:
        Grid_xal = np.full((nGrid, nGrid, nGrid), 0)


    if truncate is True:
        #xal = np.mean(Grid_xal)
        #Grid_xal[np.where(Grid_xal > 1 + 2 * xal)] = 1 + 2 * xal
        delta_x_al = delta_fct(Grid_xal)
        mean_x_al = np.mean(Grid_xal)
        R_alpha = delta_x_al / (1 + delta_x_al * mean_x_al / (1 + mean_x_al))
        indics = np.where(delta_x_al * mean_x_al / (1 + mean_x_al) < 1e-1)
        R_alpha[indics] = delta_x_al[indics]
        Grid_xal = mean_x_al * (R_alpha + 1)

        #Grid_Temp -= T_adiab_fluctu(GS_PS_dict['z'], param, delta_rho)
        Tk = np.mean(Grid_Temp)
        delta_Tk = delta_fct(Grid_Temp)
        R_Tk = delta_Tk/(1+delta_Tk)
        indics = np.where(np.abs(delta_Tk) < 1e-1)
        R_Tk[indics] = delta_Tk[indics]
        Grid_Temp = Tk*(R_Tk+1)
        #Grid_Temp += T_adiab_fluctu(GS_PS_dict['z'], param, delta_rho)
        #Grid_Temp[np.where(Grid_Temp > 2*Tk)] = 2 * Tk ## if delta_T>1, then 1/T ~ 0, so 1-delta_T~0 for the expansion to work.


    delta_XHII = delta_fct(Grid_xHII)
    delta_x_al = delta_fct(Grid_xal)
    delta_T = delta_fct(Grid_Temp)

    dens_field = param.sim.dens_field
    if dens_field is not None:
        PS_rho = auto_PS(delta_rho, box_dims=Lbox, kbins=kbins)[0]
        PS_rho_xHII = cross_PS(delta_XHII, delta_rho, box_dims=Lbox, kbins=kbins)[0]
        PS_rho_xal = cross_PS(delta_x_al, delta_rho, box_dims=Lbox, kbins=kbins)[0]
        PS_rho_T = cross_PS(delta_T, delta_rho, box_dims=Lbox, kbins=kbins)[0]
    else:
        PS_rho, PS_rho_xHII, PS_rho_xal, PS_rho_T = 0, 0, 0, 0  # rho/rhomean-1
        print('no density field provided.')

    PS_xHII = auto_PS(delta_XHII, box_dims=Lbox, kbins=kbins)[0]
    PS_T = auto_PS(delta_T, box_dims=Lbox, kbins=kbins)[0]
    PS_xal = auto_PS(delta_x_al, box_dims=Lbox, kbins=kbins)[0]

    PS_T_lyal = cross_PS(delta_T, delta_x_al, box_dims=Lbox, kbins=kbins)[0]
    PS_T_xHII = cross_PS(delta_T, delta_XHII, box_dims=Lbox, kbins=kbins)[0]
    PS_lyal_xHII = cross_PS(delta_x_al, delta_XHII, box_dims=Lbox, kbins=kbins)[0]

    PS_rho_dTb = cross_PS(delta_fct(Grid_dTb), delta_rho, box_dims=Lbox,
                          kbins=kbins)[0]


    dict_cross_corr = {'PS_xHII': PS_xHII, 'PS_T': PS_T, 'PS_xal': PS_xal, 'PS_rho': PS_rho, 'PS_T_lyal': PS_T_lyal,
                       'PS_T_xHII': PS_T_xHII, 'PS_lyal_xHII': PS_lyal_xHII, 'PS_rho_xHII': PS_rho_xHII,
                       'PS_rho_xal': PS_rho_xal, 'PS_rho_T': PS_rho_T,'PS_rho_dTb': PS_rho_dTb}

    if third_order:
        PS_ra_a = cross_PS(delta_XHII * delta_x_al, delta_x_al, box_dims=Lbox, kbins=kbins)[0]
        PS_r_aa = cross_PS(delta_XHII, delta_x_al ** 2, box_dims=Lbox, kbins=kbins)[0]

        PS_rb_b = cross_PS(delta_XHII * delta_rho, delta_rho, box_dims=Lbox, kbins=kbins)[0]

        PS_rT_T = cross_PS(delta_XHII * delta_T, delta_T, box_dims=Lbox, kbins=kbins)[0]
        PS_r_TT = cross_PS(delta_XHII, delta_T ** 2, box_dims=Lbox, kbins=kbins)[0]

        PS_ab_r = cross_PS(delta_rho * delta_x_al, delta_XHII, box_dims=Lbox, kbins=kbins)[0]
        PS_rb_a = cross_PS(delta_XHII * delta_rho, delta_x_al, box_dims=Lbox, kbins=kbins)[0]
        PS_ra_b = cross_PS(delta_XHII * delta_x_al, delta_rho, box_dims=Lbox, kbins=kbins)[0]

        PS_rT_b = cross_PS(delta_XHII * delta_T, delta_rho, box_dims=Lbox, kbins=kbins)[0]
        PS_rb_T = cross_PS(delta_XHII * delta_rho, delta_T, box_dims=Lbox, kbins=kbins)[0]
        PS_Tb_r = cross_PS(delta_T * delta_rho, delta_XHII, box_dims=Lbox, kbins=kbins)[0]

        PS_aT_r = cross_PS(delta_x_al * delta_T, delta_XHII, box_dims=Lbox, kbins=kbins)[0]
        PS_ar_T = cross_PS(delta_XHII * delta_x_al, delta_T, box_dims=Lbox, kbins=kbins)[0]
        PS_Tr_a = cross_PS(delta_XHII * delta_T, delta_x_al, box_dims=Lbox, kbins=kbins)[0]

        PS_Tr_r = cross_PS(delta_T * delta_XHII, delta_XHII, box_dims=Lbox, kbins=kbins)[0]
        PS_ar_r = cross_PS(delta_x_al * delta_XHII, delta_XHII, box_dims=Lbox, kbins=kbins)[0]
        PS_br_r = cross_PS(delta_rho * delta_XHII, delta_XHII, box_dims=Lbox, kbins=kbins)[0]

        PS_rT_rT = auto_PS(delta_XHII * delta_T, box_dims=Lbox, kbins=kbins)[0]
        PS_ra_ra = auto_PS(delta_XHII * delta_x_al, box_dims=Lbox, kbins=kbins)[0]
        PS_rb_rb = auto_PS(delta_XHII * delta_rho, box_dims=Lbox, kbins=kbins)[0]

        PS_raa_r = cross_PS(delta_XHII * delta_x_al ** 2, delta_XHII, box_dims=Lbox, kbins=kbins)[0]
        PS_rTT_r = cross_PS(delta_XHII * delta_T ** 2, delta_XHII, box_dims=Lbox, kbins=kbins)[0]

        PS_rba_r = cross_PS(delta_XHII * delta_rho * delta_x_al, delta_XHII, box_dims=Lbox, kbins=kbins)[0]
        PS_rTa_r = cross_PS(delta_XHII * delta_T * delta_x_al, delta_XHII, box_dims=Lbox, kbins=kbins)[0]
        PS_rTb_r = cross_PS(delta_XHII * delta_T * delta_rho, delta_XHII, box_dims=Lbox, kbins=kbins)[0]

        PS_rb_ra = cross_PS(delta_XHII * delta_rho, delta_XHII * delta_x_al, box_dims=Lbox, kbins=kbins)[0]
        PS_rT_ra = cross_PS(delta_XHII * delta_T, delta_XHII * delta_x_al, box_dims=Lbox, kbins=kbins)[0]
        PS_rT_rb = cross_PS(delta_XHII * delta_T, delta_XHII * delta_rho, box_dims=Lbox, kbins=kbins)[0]

        Dict_3rd_order = {'PS_ra_a': PS_ra_a, 'PS_r_aa': PS_r_aa, 'PS_rb_b': PS_rb_b, 'PS_rT_T': PS_rT_T,
                          'PS_r_TT': PS_r_TT, 'PS_ab_r': PS_ab_r, 'PS_rb_a': PS_rb_a, 'PS_ra_b': PS_ra_b,
                          'PS_rT_b': PS_rT_b, 'PS_rb_T': PS_rb_T, 'PS_Tb_r': PS_Tb_r, 'PS_aT_r': PS_aT_r,
                          'PS_ar_T': PS_ar_T, 'PS_Tr_a': PS_Tr_a, 'PS_Tr_r': PS_Tr_r, 'PS_ar_r': PS_ar_r,
                          'PS_br_r': PS_br_r, 'PS_rT_rT': PS_rT_rT, 'PS_ra_ra': PS_ra_ra, 'PS_rb_rb': PS_rb_rb,
                          'PS_raa_r': PS_raa_r, 'PS_rTT_r': PS_rTT_r, 'PS_rba_r': PS_rba_r, 'PS_rTa_r': PS_rTa_r,
                          'PS_rTb_r': PS_rTb_r, 'PS_rb_ra': PS_rb_ra, 'PS_rT_ra': PS_rT_ra, 'PS_rT_rb': PS_rT_rb}
        dict_cross_corr = Merge(Dict_3rd_order, dict_cross_corr)


    if fourth_order :
        ## 3rd order in aT (aaT,aaa, TTT,TTa...)
        PS_aa_a = cross_PS(delta_x_al ** 2, delta_x_al, box_dims=Lbox, kbins=kbins)[0]
        PS_aa_aa = cross_PS(delta_x_al ** 2, delta_x_al**2, box_dims=Lbox, kbins=kbins)[0]
        PS_aT_a = cross_PS(delta_x_al * delta_T, delta_x_al, box_dims=Lbox, kbins=kbins)[0]
        PS_aT_T = cross_PS(delta_x_al * delta_T, delta_T, box_dims=Lbox, kbins=kbins)[0]
        PS_aa_T = cross_PS(delta_x_al ** 2, delta_T, box_dims=Lbox, kbins=kbins)[0]
        PS_TT_a = cross_PS(delta_T ** 2, delta_x_al, box_dims=Lbox, kbins=kbins)[0]
        PS_TT_T = cross_PS(delta_T ** 2, delta_T, box_dims=Lbox, kbins=kbins)[0]
        PS_TT_TT = cross_PS(delta_T ** 2, delta_T**2, box_dims=Lbox, kbins=kbins)[0]

        ## 3rd order in aTb (aab,aTb, TTb,TbT...)
        PS_ab_a = cross_PS(delta_x_al * delta_rho, delta_x_al, box_dims=Lbox, kbins=kbins)[0]
        PS_aa_b = cross_PS(delta_x_al ** 2, delta_rho, box_dims=Lbox, kbins=kbins)[0]
        PS_ab_b = cross_PS(delta_x_al * delta_rho, delta_rho, box_dims=Lbox, kbins=kbins)[0]

        PS_ab_T = cross_PS(delta_x_al * delta_rho, delta_T, box_dims=Lbox, kbins=kbins)[0]
        PS_aT_b = cross_PS(delta_x_al * delta_T, delta_rho, box_dims=Lbox, kbins=kbins)[0]
        PS_bT_a = cross_PS(delta_rho * delta_T, delta_x_al, box_dims=Lbox, kbins=kbins)[0]

        PS_bT_b = cross_PS(delta_T * delta_rho, delta_rho, box_dims=Lbox, kbins=kbins)[0]
        PS_Tb_T = cross_PS(delta_T * delta_rho, delta_T, box_dims=Lbox, kbins=kbins)[0]
        PS_TT_b = cross_PS(delta_T ** 2, delta_rho, box_dims=Lbox, kbins=kbins)[0]





        ##### 4th order terms

        PS_aa_ba = cross_PS(delta_x_al ** 2, delta_rho * delta_x_al, box_dims=Lbox, kbins=kbins)[0]
        PS_aab_a = cross_PS(delta_x_al ** 2 * delta_rho, delta_x_al, box_dims=Lbox, kbins=kbins)[0]
        PS_ab_ab = cross_PS(delta_x_al * delta_rho, delta_x_al * delta_rho, box_dims=Lbox, kbins=kbins)[0]
        PS_aab_b = cross_PS(delta_x_al ** 2 * delta_rho, delta_rho, box_dims=Lbox, kbins=kbins)[0]

        PS_aaT_a = cross_PS(delta_x_al ** 2 * delta_T, delta_x_al, box_dims=Lbox, kbins=kbins)[0]
        PS_aa_Ta = cross_PS(delta_x_al ** 2, delta_T * delta_x_al, box_dims=Lbox, kbins=kbins)[0]

        PS_baT_a = cross_PS(delta_rho * delta_x_al * delta_T, delta_x_al, box_dims=Lbox, kbins=kbins)[0]
        PS_ba_aT = cross_PS(delta_rho * delta_x_al, delta_x_al * delta_T, box_dims=Lbox, kbins=kbins)[0]

        PS_aaT_b = cross_PS(delta_x_al ** 2 * delta_T, delta_rho, box_dims=Lbox, kbins=kbins)[0]
        PS_aab_T = cross_PS(delta_x_al ** 2 * delta_rho, delta_T, box_dims=Lbox, kbins=kbins)[0]
        PS_aa_Tb = cross_PS(delta_x_al ** 2, delta_T * delta_rho, box_dims=Lbox, kbins=kbins)[0]

        PS_ba_Tb = cross_PS(delta_rho * delta_x_al, delta_T * delta_rho, box_dims=Lbox, kbins=kbins)[0]
        PS_abT_b = cross_PS(delta_rho * delta_x_al*delta_T, delta_rho, box_dims=Lbox, kbins=kbins)[0]
        PS_aTT_a = cross_PS(delta_x_al * delta_T ** 2, delta_x_al, box_dims=Lbox, kbins=kbins)[0]

        PS_aa_TT = cross_PS(delta_x_al ** 2, delta_T ** 2, box_dims=Lbox, kbins=kbins)[0]
        PS_aT_aT = cross_PS(delta_x_al * delta_T, delta_x_al * delta_T, box_dims=Lbox, kbins=kbins)[0]
        PS_aaT_T = cross_PS(delta_x_al ** 2 * delta_T, delta_T, box_dims=Lbox, kbins=kbins)[0]

        PS_aTT_b = cross_PS(delta_x_al * delta_T ** 2, delta_rho, box_dims=Lbox, kbins=kbins)[0]
        PS_bTT_a = cross_PS(delta_rho * delta_T ** 2, delta_x_al, box_dims=Lbox, kbins=kbins)[0]
        PS_TT_ab = cross_PS(delta_T ** 2, delta_x_al * delta_rho, box_dims=Lbox, kbins=kbins)[0]

        PS_baT_T = cross_PS(delta_rho * delta_x_al * delta_T, delta_T, box_dims=Lbox, kbins=kbins)[0]
        PS_bT_aT = cross_PS(delta_rho * delta_T, delta_x_al * delta_T, box_dims=Lbox, kbins=kbins)[0]
        PS_bTT_b = cross_PS(delta_rho * delta_T ** 2, delta_rho, box_dims=Lbox, kbins=kbins)[0]

        PS_Tb_Tb = cross_PS(delta_rho * delta_T, delta_rho * delta_T, box_dims=Lbox, kbins=kbins)[0]

        PS_aTT_T = cross_PS(delta_x_al * delta_T ** 2, delta_T, box_dims=Lbox, kbins=kbins)[0]
        PS_TT_aT = cross_PS(delta_T ** 2, delta_x_al * delta_T, box_dims=Lbox, kbins=kbins)[0]

        PS_bTT_T = cross_PS(delta_rho * delta_T ** 2, delta_T, box_dims=Lbox, kbins=kbins)[0]
        PS_TT_bT = cross_PS(delta_T ** 2, delta_rho * delta_T, box_dims=Lbox, kbins=kbins)[0]


        Dict_4th_order = {'PS_aa_a': PS_aa_a, 'PS_aT_a': PS_aT_a, 'PS_aT_T': PS_aT_T, 'PS_aa_T': PS_aa_T,
                          'PS_TT_a': PS_TT_a, 'PS_TT_T': PS_TT_T, 'PS_aa_aa':PS_aa_aa,'PS_TT_TT':PS_TT_TT,
                          'PS_ab_a':PS_ab_a, 'PS_aa_b': PS_aa_b, 'PS_ab_b': PS_ab_b, 'PS_ab_T': PS_ab_T,
                          'PS_aT_b': PS_aT_b, 'PS_bT_a': PS_bT_a, 'PS_bT_b': PS_bT_b, 'PS_Tb_T': PS_Tb_T,
                          'PS_TT_b': PS_TT_b , 'PS_aa_ba': PS_aa_ba, 'PS_aab_a': PS_aab_a, 'PS_ab_ab': PS_ab_ab, 'PS_aab_b': PS_aab_b
        , 'PS_aaT_a': PS_aaT_a, 'PS_aa_Ta': PS_aa_Ta
        , 'PS_baT_a': PS_baT_a, 'PS_ba_aT': PS_ba_aT
        , 'PS_aaT_b': PS_aaT_b, 'PS_aab_T': PS_aab_T, 'PS_aa_Tb': PS_aa_Tb
        , 'PS_ba_Tb': PS_ba_Tb, 'PS_abT_b': PS_abT_b, 'PS_aTT_a': PS_aTT_a
        , 'PS_aa_TT': PS_aa_TT, 'PS_aT_aT': PS_aT_aT, 'PS_aaT_T': PS_aaT_T

        , 'PS_aTT_b': PS_aTT_b, 'PS_bTT_a': PS_bTT_a, 'PS_TT_ab': PS_TT_ab
        , 'PS_baT_T': PS_baT_T, 'PS_bT_aT': PS_bT_aT, 'PS_bTT_b': PS_bTT_b
        , 'PS_Tb_Tb': PS_Tb_Tb
        , 'PS_aTT_T': PS_aTT_T, 'PS_TT_aT': PS_TT_aT
        , 'PS_bTT_T': PS_bTT_T, 'PS_TT_bT': PS_TT_bT
                          }

        dict_cross_corr = Merge(Dict_4th_order, dict_cross_corr)

    return Merge(GS_PS_dict, dict_cross_corr)


def Merge(dict_1, dict_2):
    return {**dict_1, **dict_2}


def compute_PS(param, Tspin=False, RSD=False, ion='bubbles', cross_corr=False):
    """
    Parameters
    ----------
    param : dictionnary containing all the input parameters
    Tspin : if True, will compute the spin temperature Power Spectrum as well as cross correlation with matter field and xHII field.
    cross_corr : Choose to compute the cross correlations. If False, it speeds up.
    Returns
    -------
    Computes the power spectra of the desired quantities

    """
    import tools21cm as t2c
    start_time = time.time()
    model_name = param.sim.model_name
    nGrid = param.sim.Ncell
    Lbox = param.sim.Lbox  # Mpc/h
    if cross_corr == True:
        print('Computing PS with all cross correlations.')

    kbins = def_k_bins(param)

    z_arr = def_redshifts(param)
    nbr_snap = len(z_arr)

    dTb_arr = np.zeros((nbr_snap))
    dTb_RSD_arr = np.zeros((nbr_snap))

    PS_xHII = np.zeros((nbr_snap, len(kbins) - 1))
    PS_T = np.zeros((nbr_snap, len(kbins) - 1))
    PS_xal = np.zeros((nbr_snap, len(kbins) - 1))
    PS_rho = np.zeros((nbr_snap, len(kbins) - 1))
    PS_dTb = np.zeros((nbr_snap, len(kbins) - 1))
    if RSD:
        PS_dTb_RSD = np.zeros((nbr_snap, len(kbins) - 1))

    if cross_corr:
        PS_T_lyal = np.zeros((nbr_snap, len(kbins) - 1))
        PS_T_xHII = np.zeros((nbr_snap, len(kbins) - 1))
        PS_rho_xHII = np.zeros((nbr_snap, len(kbins) - 1))
        PS_rho_xal = np.zeros((nbr_snap, len(kbins) - 1))
        PS_rho_T = np.zeros((nbr_snap, len(kbins) - 1))
        PS_lyal_xHII = np.zeros((nbr_snap, len(kbins) - 1))

    if Tspin:
        PS_Ts = np.zeros((nbr_snap, len(kbins) - 1))
        PS_rho_Ts = np.zeros((nbr_snap, len(kbins) - 1))
        PS_Ts_xHII = np.zeros((nbr_snap, len(kbins) - 1))
        PS_T_Ts = np.zeros((nbr_snap, len(kbins) - 1))

    for ii, z in enumerate(z_arr):
        zz_ = z
        Grid_Temp = load_grid(param, z,
                              type='Tk')  # pickle.load(file=open('./grid_output/T_Grid'    + str(nGrid) + model_name + '_snap' + z_str, 'rb'))
        Grid_xHII = load_grid(param, z, type=ion)
        Grid_dTb = load_grid(param, z, type='dTb')
        Grid_xal = load_grid(param, z, type='lyal')

        if Tspin:
            T_cmb_z = Tcmb0 * (1 + zz_)
            # Grid_xcoll = pickle.load(file=open('./grid_output/xcoll_Grid' + str(nGrid) + model_name + '_snap' + filename[4:-5], 'rb'))
            Grid_Tspin = ((1 / T_cmb_z + Grid_xal / Grid_Temp) / (1 + Grid_xal)) ** -1

        if Grid_Temp.size == 1:  ## to avoid error when measuring power spectrum
            Grid_Temp = np.full((nGrid, nGrid, nGrid), 1)
        if Grid_xHII.size == 1:
            Grid_xHII = np.full((nGrid, nGrid, nGrid), 0)  ## to avoid div by zero
        if Grid_dTb.size == 1:
            Grid_dTb = np.full((nGrid, nGrid, nGrid), 1)
        if Grid_xal.size == 1:
            Grid_xal = np.full((nGrid, nGrid, nGrid), 1)

        delta_dTb = Grid_dTb / np.mean(Grid_dTb) - 1
        delta_XHII = Grid_xHII / np.mean(Grid_xHII) - 1

        if cross_corr:
            delta_T = Grid_Temp / np.mean(Grid_Temp) - 1
            delta_x_al = Grid_xal / np.mean(Grid_xal) - 1

        ii = np.where(z_arr == zz_)

        if Tspin:
            delta_Tspin = Grid_Tspin / np.mean(Grid_Tspin) - 1

        dens_field = param.sim.dens_field
        if dens_field is not None:
            delta_rho = load_delta_b(param, z_string_format(z))
            PS_rho[ii] = auto_PS(delta_rho, box_dims=Lbox, kbins=kbins)[0]
            if cross_corr:
                PS_rho_xHII[ii] = \
                    cross_PS(delta_XHII, delta_rho, box_dims=Lbox, kbins=kbins)[0]
                PS_rho_xal[ii] = \
                    cross_PS(delta_x_al, delta_rho, box_dims=Lbox, kbins=kbins)[0]
                PS_rho_T[ii] = \
                    cross_PS(delta_T, delta_rho, box_dims=Lbox, kbins=kbins)[0]
            if Tspin:
                PS_rho_Ts[ii] = \
                    cross_PS(delta_Tspin, delta_rho, box_dims=Lbox, kbins=kbins)[0]
        else:
            delta_rho = 0, 0  # rho/rhomean-1
            print('no density field provided.')

        if RSD:
            Grid_dTb_RSD = dTb_RSD(param, zz_, delta_rho, Grid_dTb)
            delta_Grid_dTb_RSD = Grid_dTb_RSD / np.mean(Grid_dTb_RSD) - 1
            PS_dTb_RSD[ii] = auto_PS(delta_Grid_dTb_RSD, box_dims=Lbox, kbins=kbins)[0]
            dTb_RSD_arr[ii] = np.mean(Grid_dTb_RSD)

        z_arr[ii] = zz_
        if cross_corr:
            PS_xHII[ii] = auto_PS(delta_XHII, box_dims=Lbox, kbins=kbins)[0]
            PS_T[ii] = auto_PS(delta_T, box_dims=Lbox, kbins=kbins)[0]
            PS_xal[ii] = auto_PS(delta_x_al, box_dims=Lbox, kbins=kbins)[0]

        PS_dTb[ii], k_bins = auto_PS(delta_dTb, box_dims=Lbox, kbins=kbins)
        dTb_arr[ii] = np.mean(Grid_dTb)

        if cross_corr:
            PS_T_lyal[ii] = cross_PS(delta_T, delta_x_al, box_dims=Lbox, kbins=kbins)[0]
            PS_T_xHII[ii] = cross_PS(delta_T, delta_XHII, box_dims=Lbox, kbins=kbins)[0]
            PS_lyal_xHII[ii] = cross_PS(delta_x_al, delta_XHII, box_dims=Lbox, kbins=kbins)[0]

        if Tspin:
            PS_Ts[ii] = auto_PS(delta_Tspin, box_dims=Lbox, kbins=kbins)[0]
            PS_Ts_xHII[ii] = \
                cross_PS(delta_Tspin, delta_XHII, box_dims=Lbox, kbins=kbins)[0]
            PS_T_Ts[ii] = cross_PS(delta_Tspin, delta_T, box_dims=Lbox, kbins=kbins)[
                0]

    Dict = {'z': z_arr, 'k': k_bins, 'PS_xHII': PS_xHII, 'PS_T': PS_T, 'PS_xal': PS_xal, 'PS_dTb': PS_dTb,
            'PS_rho': PS_rho,
            'dTb': dTb_arr}

    if RSD:
        Dict['PS_dTb_RSD'] = PS_dTb_RSD, Dict['dTb_RSD'] = dTb_RSD_arr
    if cross_corr:
        Dict['PS_T_lyal'], Dict['PS_T_xHII'], Dict['PS_rho_xHII'], Dict['PS_rho_xal'], Dict['PS_rho_T'], Dict[
            'PS_lyal_xHII'] = PS_T_lyal, PS_T_xHII, PS_rho_xHII, PS_rho_xal, PS_rho_T, PS_lyal_xHII
    if Tspin:
        Dict['PS_Ts'], Dict['PS_rho_Ts'], Dict['PS_xHII_Ts'], Dict['PS_T_Ts'] = PS_Ts, PS_rho_Ts, PS_Ts_xHII, PS_T_Ts
    end_time = time.time()

    print('Computing the power spectra took : ', start_time - end_time)
    pickle.dump(file=open('./physics/PS_' + str(nGrid) + model_name + '.pkl', 'wb'), obj=Dict)

def RSD_field(param, density_field, zz):
    """
    eq 4 from 411, 955–972 (Mesinger 2011, 21cmFAST..):  dvr/dr(k) = -kr**2/k**2 * dD/da(z)/D(z) * a * delta_nl(k) * da/dt
    And da/dt = H * a
    Take density field, go in Fourier space, transform it, go back to real space to get dvr/dr.
    Divide dTb to the output of this function to add RSD corrections.

    Parameters
    ----------
    density_field : delta_b, output of load_delta_b

    Returns
    ---------
    Returns a meshgrid containing values of -->(dv/dr/H+1) <--. Dimensionless.
    """

    import scipy
    Ncell = param.sim.Ncell
    Lbox = param.sim.Lbox
    delta_k = scipy.fft.fftn(density_field)

    scale_factor = np.linspace(1 / 40, 1 / 7, 100)
    growth_factor = np.zeros(len(scale_factor))
    for i in range(len(scale_factor)):
        growth_factor[i] = D(scale_factor[i], param)
    dD_da = np.gradient(growth_factor, scale_factor)

    kx_meshgrid = (np.fft.fftfreq(Ncell) * Ncell * 2 * np.pi / Lbox)[:, None, None]
    ky_meshgrid = (np.fft.fftfreq(Ncell) * Ncell * 2 * np.pi / Lbox)[None, :, None]
    kz_meshgrid = (np.fft.fftfreq(Ncell) * Ncell * 2 * np.pi / Lbox)[None, None, :]

    k_sq = kx_meshgrid ** 2 + ky_meshgrid ** 2 + kz_meshgrid ** 2  #

    aa = 1 / (zz + 1)
    dv_dr_k_over_H = -kx_meshgrid ** 2 / k_sq * np.interp(aa, scale_factor, dD_da) * delta_k / D(aa, param) * aa * aa
    dv_dr_k_over_H[np.where(np.isnan(dv_dr_k_over_H))] = np.interp(aa, scale_factor, dD_da) * delta_k[
        np.where(np.isnan(dv_dr_k_over_H))] / D(aa, param) * aa * aa  ## to deal with nan value for k=0

    dv_dr_over_H = np.real(scipy.fft.ifftn(dv_dr_k_over_H))  #### THIS IS dv_dr/H

    return dv_dr_over_H + 1


def saturated_Tspin(param, ion='bubbles'):
    """
    Computes the power spectrum and GS under the assumption that Tspin>>Tgamma (saturated).
    """
    print('Computing GS and PS under the assumption Tspin >> Tgamma')
    start_time = time.time()
    import tools21cm as t2c
    catalog_dir = param.sim.halo_catalogs
    model_name = param.sim.model_name
    nGrid = param.sim.Ncell
    Om, Ob, h0 = param.cosmo.Om, param.cosmo.Ob, param.cosmo.h
    factor = dTb_factor(param)  # factor used in dTb calculation

    Lbox = param.sim.Lbox  # Mpc/h
    if isinstance(param.sim.kbin, int):
        kbins = np.logspace(np.log10(param.sim.kmin), np.log10(param.sim.kmax), param.sim.kbin, base=10)  # h/Mpc
    elif isinstance(param.sim.kbin, str):
        kbins = np.loadtxt(param.sim.kbin)
    else:
        print(
            'param.sim.kbin should be either a path to a text files containing kbins edges values or it should be an int.')

    nbr_snap = 0
    for filename in os.listdir(catalog_dir):  # count the number of snapshots
        nbr_snap += 1

    PS_xHII = np.zeros((nbr_snap, len(kbins) - 1))
    PS_rho = np.zeros((nbr_snap, len(kbins) - 1))
    PS_dTb = np.zeros((nbr_snap, len(kbins) - 1))

    zz, xHII, dTb = [], [], []
    print('Looping over redshifts....')
    for ii, filename in enumerate(os.listdir(catalog_dir)):
        zz_ = load_f(catalog_dir + filename)['z']
        dens_field = param.sim.dens_field
        if dens_field is not None:
            delta_b = load_delta_b(param, z_string_format(z))
        else:
            delta_b = 0
        zz.append(zz_)

        Grid_xHII = load_grid(param, z, type=ion)

        Grid_dTb = factor * np.sqrt(1 + zz_) * (1 - Grid_xHII) * (delta_b + 1)

        if Grid_xHII.size == 1 and zz_ > 20:  ## arbitrary
            Grid_xHII = np.full((nGrid, nGrid, nGrid), 0)  ## to avoid div by zero
        if Grid_xHII.size == 1 and zz_ < 20:  ## arbitrary
            Grid_xHII = np.full((nGrid, nGrid, nGrid), 1)  ## to avoid div by zero
        if Grid_dTb.size == 1:
            Grid_dTb = np.full((nGrid, nGrid, nGrid), 1)

        delta_XHII = Grid_xHII / np.mean(Grid_xHII) - 1
        delta_dTb = Grid_dTb / np.mean(Grid_dTb) - 1
        xHII.append(np.mean(Grid_xHII))
        dTb.append(np.mean(Grid_dTb))
        PS_rho[ii] = auto_PS(delta_b, box_dims=Lbox, kbins=kbins)[0]
        PS_xHII[ii], k_bins = auto_PS(delta_XHII, box_dims=Lbox, kbins=kbins)
        PS_dTb[ii] = auto_PS(delta_dTb, box_dims=Lbox, kbins=kbins)[0]

    z_arr, xHII, dTb = np.array(zz), np.array(xHII), np.array(dTb)
    Dict = {'z': z_arr, 'k': k_bins, 'dTb': dTb, 'xHII': xHII, 'PS_dTb': PS_dTb, 'PS_xHII': PS_xHII, 'PS_rho': PS_rho}
    end_time = time.time()

    print('Computing the power spectra under the assumption Tspin >> Tgamma took : ', start_time - end_time)
    pickle.dump(file=open('./physics/GS_PS_Tspin_saturated_' + str(nGrid) + '_' + model_name + '.pkl', 'wb'), obj=Dict)

    return Grid_dTb


def compute_velocity(param, zz, density_field):
    """
    Take density field, go in Fourier space, transform it, go back to real space to get v.

    Parameters
    ----------
    param : dictionnary
    density_field : delta_b, output of load_delta_b
    zz : float, redshift.

    Returns
    ---------
    Returns 3 meshgrids containing values of v_x, v_y, v_z, in physical km/s.
    """

    Ncell, Lbox = param.sim.Ncell, param.sim.Lbox
    aa = 1 / (zz + 1)
    scale_factor = np.linspace(1 / (40), 1 / 7, 100)
    growth_factor = np.zeros(len(scale_factor))
    for i in range(len(scale_factor)):
        growth_factor[i] = D(scale_factor[i], param)
    dD_da = np.gradient(growth_factor, scale_factor)

    delta_k = np.fft.fftn(
        density_field.astype('float64'))  ### -> delta_k/N**3  gives Fourier(delta)(2*pi*k/L) for k = [0...N-1]^3

    kx_meshgrid = (np.fft.fftfreq(Ncell) * Ncell * 2 * np.pi / Lbox)[:, None, None]
    ky_meshgrid = (np.fft.fftfreq(Ncell) * Ncell * 2 * np.pi / Lbox)[None, :, None]
    kz_meshgrid = (np.fft.fftfreq(Ncell) * Ncell * 2 * np.pi / Lbox)[None, None, :]

    k_sq = kx_meshgrid ** 2 + ky_meshgrid ** 2 + kz_meshgrid ** 2  #

    v_x_k = kx_meshgrid / k_sq * (0 + 1j) * np.interp(aa, scale_factor, dD_da) * hubble(zz,
                                                                                        param) * aa ** 2 / param.cosmo.h * delta_k / D(
        aa, param)
    v_y_k = ky_meshgrid / k_sq * (0 + 1j) * np.interp(aa, scale_factor, dD_da) * hubble(zz,
                                                                                        param) * aa ** 2 / param.cosmo.h * delta_k / D(
        aa, param)
    v_z_k = kz_meshgrid / k_sq * (0 + 1j) * np.interp(aa, scale_factor, dD_da) * hubble(zz,
                                                                                        param) * aa ** 2 / param.cosmo.h * delta_k / D(
        aa, param)

    v_x_k[np.where(np.isnan(v_x_k))] = np.interp(aa, scale_factor, dD_da) * hubble(zz,
                                                                                   param) * aa ** 2 / param.cosmo.h * \
                                       delta_k[np.where(np.isnan(v_x_k))] / D(aa, param) * (0 + 1j)
    v_y_k[np.where(np.isnan(v_y_k))] = np.interp(aa, scale_factor, dD_da) * hubble(zz,
                                                                                   param) * aa ** 2 / param.cosmo.h * \
                                       delta_k[np.where(np.isnan(v_y_k))] / D(aa, param) * (0 + 1j)
    v_z_k[np.where(np.isnan(v_z_k))] = np.interp(aa, scale_factor, dD_da) * hubble(zz,
                                                                                   param) * aa ** 2 / param.cosmo.h * \
                                       delta_k[np.where(np.isnan(v_z_k))] / D(aa, param) * (0 + 1j)

    v_x = np.real(scipy.fft.ifftn(v_x_k))  #### km/s
    v_y = np.real(scipy.fft.ifftn(v_y_k))  #### km/s
    v_z = np.real(scipy.fft.ifftn(v_z_k))  #### km/s

    return v_x, v_y, v_z


def dTb_RSD(param, zz, delta_b, grid_dTb):
    """
    Use tools21cm to apply Redshift Space Distortion to the dTb field.

    Parameters
    ----------
    param : dictionnary
    density_field : delta_b, output of load_delta_b (meshgrid (Ncell, Ncell,Ncell))
    zz : float, redshift.
    grid_dTb: meshgrid (Ncell, Ncell,Ncell), dTb field.

    Returns
    ---------
    meshgrid (Ncell, Ncell,Ncell) : dTb field with RSD.
    """
    ##### Computing Velocities from Zeldo'vich
    v_x, v_z, v_z = compute_velocity(param, zz, delta_b)
    kms = np.array((v_x, v_z, v_z))  ### kms should be in km/s, and of shape (3,nGridx,nGridy,nGridyz)
    t2c.conv.LB = param.sim.Lbox
    dT_rsd = t2c.get_distorted_dt(grid_dTb, kms, zz, los_axis=0, velocity_axis=0, num_particles=20)
    return dT_rsd


def compute_variance(param,k_bins,temp=True,lyal=True,rho_b = True, ion = True):
    if not os.path.isdir('./variances'):
        os.mkdir('./variances')

    start_time = time.time()
    print('Compute variance of the individual fields.')

    comm, rank, size = initialise_mpi4py(param)

    z_arr = def_redshifts(param)
    for ii, z in enumerate(z_arr):
        z = np.round(z, 2)
        if rank == ii % size:
            z_arr = def_redshifts(param)
            nGrid = param.sim.Ncell
            z_str = z_string_format(z)
            file = './variances/var_z' + str(nGrid) + '_' + param.sim.model_name + '_' + z_str + '.pkl'
            if not exists(file):
                print('Core nbr', rank, 'is taking care of z = ', z)
                print('----- Computing variance for z =', z, '-------')
                if temp :
                    Grid_Temp = load_grid(param, z=z, type='Tk')
                else : Grid_Temp = None

                if ion:
                    Grid_xHII = load_grid(param, z=z, type='bubbles')
                else : Grid_xHII = None

                if lyal :
                    Grid_xal  = load_grid(param, z=z, type='lyal')
                else : Grid_xal = None

                if rho_b:
                    delta_b   = load_delta_b(param, z_string_format(z))
                else : delta_b = None


                compute_var_single_z(param, z, Grid_xal, Grid_xHII, Grid_Temp,delta_b,k_bins)
                print('----- Variance at z = ', z, ' is computed -------')
            else :
                continue

    Barrier(comm)

    if rank == 0:
        gather_files(param, path='./variances/var_z', z_arr=z_arr, Ncell=param.sim.Ncell)

        end_time = time.time()
        print('Finished computing variances. It took in total: ',
              end_time - start_time)
        print('  ')

def compute_var_single_z(param, z, Grid_xal, Grid_xHII, Grid_Temp,delta_b,k_bins):
    # k_bins : extra kbins to measure the variance at same k values as PS
    z_str = z_string_format(z)
    print('Computing variance for xal, xHII and Temp at z = ' + z_str + '....')
    tstart = time.time()
    nGrid = param.sim.Ncell

    Grid_Temp, Grid_xHII, Grid_xal = format_grid_for_PS_measurement(Grid_Temp, Grid_xHII, Grid_xal, nGrid)

    variance_lyal, R_scale, k_values, skewness_lyal, kurtosis_lyal = compute_var_field(param, delta_fct(Grid_xal),k_bins)
    variance_xHII, R_scale, k_values, skewness_xHII, kurtosis_xHII = compute_var_field(param, delta_fct(Grid_xHII),k_bins)
    variance_Temp, R_scale, k_values, skewness_Temp, kurtosis_Temp = compute_var_field(param, delta_fct(Grid_Temp),k_bins)
    variance_rho,  R_scale, k_values, skewness_rho,  kurtosis_rho  = compute_var_field(param, delta_b,k_bins)
    variance_U,  R_scale, k_values, skewness_rho,  kurtosis_rho    = compute_var_field(param, delta_fct(Grid_xal/(1+Grid_xal)),k_bins)
    variance_V,  R_scale, k_values, skewness_rho,  kurtosis_rho    = compute_var_field(param, delta_fct(1-T_cmb(z)/Grid_Temp),k_bins)

    print('nbr of scales is', len(k_values))

    save_f(file='./variances/var_z' + str(nGrid) + '_' + param.sim.model_name + '_' + z_str + '.pkl',
           obj={'z': z, 'var_lyal': np.array(variance_lyal), 'var_xHII': np.array(variance_xHII)
               , 'var_Temp': np.array(variance_Temp),'skewness_lyal': np.array(skewness_lyal), 'skewness_xHII': np.array(skewness_xHII)
               , 'skewness_Temp': np.array(skewness_Temp), 'kurtosis_lyal': np.array(kurtosis_lyal), 'kurtosis_xHII': np.array(kurtosis_xHII)
               , 'kurtosis_Temp': np.array(kurtosis_Temp), 'k': k_values, 'R': R_scale,
                'variance_rho':variance_rho, 'skewness_rho':skewness_rho, 'kurtosis_rho':kurtosis_rho,'variance_U':variance_U,'variance_V':variance_V })

    print('.... Done computing variance. It took :', print_time(time.time() - tstart))


def compute_var_field(param, field,k_bins):
    from .excursion_set import profile_kern
    from astropy.convolution import convolve_fft

    Lbox = param.sim.Lbox  # Mpc/h
    #kmin = 2 * np.pi / Lbox
    #kmax = 2 * np.pi / Lbox * param.sim.Ncell
    #kbin = int(2 * np.log10(kmax / kmin))
    #param.sim.kmin = kmin
    #param.sim.kmax = kmax
    #param.sim.kbin = kbin

    k_values = def_k_bins(param)
    k_values = np.sort(np.unique(np.concatenate((k_values,k_bins))))

    R_scale = np.pi / k_values
    nGrid = param.sim.Ncell  # number of grid cells

    pixel_size = Lbox / nGrid
    x = np.linspace(-Lbox / 2, Lbox / 2, nGrid)  # y, z will be the same.
    rx, ry, rz = np.meshgrid(x, x, x, sparse=True)
    rgrid = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

    variance = []
    skewness = []
    kurtosis = []
    for Rsmoothing in R_scale:
        if Rsmoothing > pixel_size:
            print('R is ', round(Rsmoothing, 2))
            kern = profile_kern(rgrid, Rsmoothing)
            smoothed_field = convolve_fft(field, kern, boundary='wrap', normalize_kernel=True, allow_huge=True)  #
            var_ = np.var(smoothed_field)
            variance.append(var_)
            skewness.append(np.mean((smoothed_field / np.sqrt(var_)) ** 3))
            kurtosis.append(np.mean((smoothed_field / np.sqrt(var_)) ** 4))
        else:
            variance.append(0)
            skewness.append(0)
            kurtosis.append(0)

    print('return : variance, R_scale, k_values, skewness, kurtosis')
    return variance, R_scale, k_values, skewness, kurtosis





def compute_cross_var(param, field1, field2):
    from .excursion_set import profile_kern
    from astropy.convolution import convolve_fft

    k_values = def_k_bins(param)
    R_scale = np.pi / k_values
    Lbox = param.sim.Lbox  # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells

    pixel_size = Lbox / nGrid
    x = np.linspace(-Lbox / 2, Lbox / 2, nGrid)  # y, z will be the same.
    rx, ry, rz = np.meshgrid(x, x, x, sparse=True)
    rgrid = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

    variance = []
    for Rsmoothing in R_scale:
        if Rsmoothing > pixel_size:
            print('R is ', round(Rsmoothing, 2))
            kern = profile_kern(rgrid, Rsmoothing)
            smoothed_field1 = convolve_fft(field1, kern, boundary='wrap', normalize_kernel=True, allow_huge=True)  #
            smoothed_field2 = convolve_fft(field2, kern, boundary='wrap', normalize_kernel=True, allow_huge=True)  #
            variance.append(np.mean(smoothed_field1 * smoothed_field2))
        else:
            variance.append(0)
    print('return : variance, R_scale, k_values')
    return variance, R_scale, k_values


def compute_corr_fct(param):
    if not os.path.isdir('./variances'):
        os.mkdir('./variances')

    start_time = time.time()
    print('Compute correlation functions of the fields, for r=0.')
    z_arr = def_redshifts(param)

    comm, rank, size = initialise_mpi4py(param)

    for ii, z in enumerate(z_arr):
        if rank == ii % size:
            print('Core nbr', rank, 'is taking care of z = ', z)
            z_str = z_string_format(z)
            delta_xHII = delta_fct(load_grid(param, z=z, type='bubbles'))
            delta_Tk = delta_fct(load_grid(param, z=z, type='Tk'))
            delta_lyal = delta_fct(load_grid(param, z=z, type='lyal'))
            delta_b = load_delta_b(param, z_string_format(z))
            Xi_TT = np.mean(delta_Tk ** 2)
            Xi_aa = np.mean(delta_lyal ** 2)
            Xi_Tb = np.mean(delta_b * delta_Tk)
            Xi_rT = np.mean(delta_xHII * delta_Tk)
            Xi_ar = np.mean(delta_xHII * delta_lyal)
            Xi_aT = np.mean(delta_Tk * delta_lyal)
            Xi_rb = np.mean(delta_xHII * delta_b)
            Xi_ab = np.mean(delta_b * delta_lyal)

            Xi_rba = np.mean(delta_xHII * delta_b * delta_lyal)
            Xi_rbT = np.mean(delta_xHII * delta_b * delta_Tk)
            Xi_raT = np.mean(delta_xHII * delta_lyal * delta_Tk)
            Xi_raa = np.mean(delta_xHII * delta_lyal ** 2)
            Xi_rTT = np.mean(delta_xHII * delta_Tk ** 2)

            Xi_aTb = np.mean(delta_lyal * delta_Tk * delta_b)
            Xi_aTrb = np.mean(delta_lyal * delta_Tk * delta_xHII * delta_b)
            Xi_aab = np.mean(delta_lyal ** 2 * delta_b)
            Xi_aarb = np.mean(delta_lyal ** 2 * delta_xHII * delta_b)
            Xi_TTb = np.mean(delta_Tk ** 2 * delta_b)
            Xi_TTrb = np.mean(delta_Tk ** 2 * delta_xHII * delta_b)

            Dict = {'z': z, 'Xi_TT': Xi_TT, 'Xi_aa': Xi_aa, 'Xi_Tb': Xi_Tb, 'Xi_rT': Xi_rT, 'Xi_ar': Xi_ar,
                'Xi_aT': Xi_aT, 'Xi_rb': Xi_rb, 'Xi_ab': Xi_ab,
                'Xi_rba': Xi_rba, 'Xi_rbT': Xi_rbT, 'Xi_raT': Xi_raT, 'Xi_raa': Xi_raa, 'Xi_rTT': Xi_rTT,
                'Xi_aTb': Xi_aTb, 'Xi_aTrb': Xi_aTrb, 'Xi_aab': Xi_aab, 'Xi_aarb': Xi_aarb, 'Xi_TTb': Xi_TTb,
                'Xi_TTrb': Xi_TTrb}

            save_f(file='./variances/Xi_corr_fct_' + str(param.sim.Ncell) + '_' + param.sim.model_name + '_' + z_str + '.pkl', obj=Dict)

    Barrier(comm)

    if rank == 0:
        gather_files(param, path='./variances/Xi_corr_fct_', z_arr=z_arr, Ncell=param.sim.Ncell)

        end_time = time.time()
        print('Finished computing Xi at r=0. It took in total: ', print_time(end_time - start_time))
        print('  ')






def investigate_xal(param,HO = False):

    if not os.path.isdir('./physics'):
        os.mkdir('./physics')

    start_time = time.time()
    print('Computing PS of xal/(1+xal).')

    comm, rank, size = initialise_mpi4py(param)

    kbins = def_k_bins(param)
    z_arr = def_redshifts(param)
    Ncell = param.sim.Ncell
    nGrid = Ncell
    Lbox = param.sim.Lbox

    for ii, z in enumerate(z_arr):
        z = np.round(z, 2)
        z_str = z_string_format(z)
        if rank == ii % size:
            print('Core nbr', rank, 'is taking care of z = ', z)
            print('----- Investigating xal for z =', z, '-------')
            Grid_Temp = load_grid(param, z=z, type='Tk')
            Grid_xHII = load_grid(param, z=z, type='bubbles')
            Grid_xal = load_grid(param, z=z, type='lyal')
            delta_b = load_delta_b(param, z_str)

            Grid_Temp, Grid_xHII, Grid_xal = format_grid_for_PS_measurement(Grid_Temp, Grid_xHII, Grid_xal, nGrid)

            PS_xal,kk               = auto_PS(delta_fct(Grid_xal), box_dims=Lbox, kbins=kbins)
            PS_xal_term_x_temps_term= auto_PS(delta_fct(Grid_xal/(Grid_xal+1)*(1-T_cmb(z)/Grid_Temp)), box_dims=Lbox, kbins=kbins)[0]
            PS_xal_term_x_matter    = auto_PS(delta_fct(Grid_xal/(Grid_xal+1)*(1+delta_b)), box_dims=Lbox, kbins=kbins)[0]
            PS_xal_matter           = auto_PS(delta_fct(Grid_xal/(Grid_xal+1)*(1+delta_b)), box_dims=Lbox, kbins=kbins)[0]
            PS_temp                 = auto_PS(delta_fct((1-T_cmb(z)/Grid_Temp)), box_dims=Lbox, kbins=kbins)[0]
            PS_temp_x_matter        = auto_PS(delta_fct((1-T_cmb(z)/Grid_Temp)*(1+delta_b)), box_dims=Lbox, kbins=kbins)[0]
            PS_xal_over_1_plus_xal  = auto_PS(delta_fct(Grid_xal/(Grid_xal+1)), box_dims=Lbox, kbins=kbins)[0]
            PS_xal_term_x_reio_term = auto_PS(delta_fct(Grid_xal/(Grid_xal+1)*(1-Grid_xHII)), box_dims=Lbox, kbins=kbins)[0]
            PS_xal_Temp_matter_term = auto_PS(delta_fct(Grid_xal/(Grid_xal+1)*(1-T_cmb(z)/Grid_Temp)*(1+delta_b)), box_dims=Lbox, kbins=kbins)[0]
            PS_xal_reio_Temp_matter_term = auto_PS(delta_fct(Grid_xal/(Grid_xal+1)*(1-T_cmb(z)/Grid_Temp)*(1+delta_b)*(1-Grid_xHII)), box_dims=Lbox, kbins=kbins)[0]


            if HO:
                PS_aa_a  = cross_PS(delta_fct(Grid_xal)**2,delta_fct(Grid_xal), box_dims=Lbox, kbins=kbins)[0]
                PS_aaa_a = cross_PS(delta_fct(Grid_xal)**3,delta_fct(Grid_xal), box_dims=Lbox, kbins=kbins)[0]
                PS_aa_aa = cross_PS(delta_fct(Grid_xal)**2,delta_fct(Grid_xal)**2, box_dims=Lbox, kbins=kbins)[0]

                PS_TT_T  = cross_PS(delta_fct(Grid_Temp) ** 2, delta_fct(Grid_Temp), box_dims=Lbox, kbins=kbins)[0]
                PS_TTT_T = cross_PS(delta_fct(Grid_Temp) ** 3, delta_fct(Grid_Temp), box_dims=Lbox, kbins=kbins)[0]
                PS_TT_TT = cross_PS(delta_fct(Grid_Temp) ** 2, delta_fct(Grid_Temp) ** 2, box_dims=Lbox, kbins=kbins)[0]

                PS_aT_a = cross_PS(delta_fct(Grid_xal) * delta_fct(Grid_Temp), delta_fct(Grid_xal), box_dims=Lbox, kbins=kbins)[0]
                PS_aT_T = cross_PS(delta_fct(Grid_xal) * delta_fct(Grid_Temp), delta_fct(Grid_Temp), box_dims=Lbox, kbins=kbins)[0]

                PS_aa_T= cross_PS(delta_fct(Grid_xal)*delta_fct(Grid_xal),delta_fct(Grid_Temp), box_dims=Lbox, kbins=kbins)[0]
                PS_TT_a= cross_PS(delta_fct(Grid_Temp)*delta_fct(Grid_Temp),delta_fct(Grid_xal), box_dims=Lbox, kbins=kbins)[0]




            mean_xal  = np.mean(Grid_xal)
            mean_xHII = np.mean(Grid_xHII)
            mean_Temp = np.mean(Grid_Temp)
            mean_Temp_term = np.mean((1-T_cmb(z)/Grid_Temp))
            mean_xal_reio = np.mean(Grid_xal/(Grid_xal+1)*(1-Grid_xHII))
            mean_xal_temp = np.mean(Grid_xal/(Grid_xal+1)*(1-T_cmb(z)/Grid_Temp))
            mean_xal_matter = np.mean(Grid_xal/(Grid_xal+1)*(1+delta_b))
            mean_xal_temp_matter = np.mean(Grid_xal/(Grid_xal+1)*(1-T_cmb(z)/Grid_Temp)*(1+delta_b))
            mean_xal_reio_temp_matter = np.mean(Grid_xal/(Grid_xal+1)*(1-T_cmb(z)/Grid_Temp)*(1+delta_b)*(1-Grid_xHII))
            mean_1_ov_1_plus_xal = np.mean(Grid_xal/(Grid_xal+1))
            var_1_ov_1_plus_xal = np.var(Grid_xal / (Grid_xal + 1))
            cross_var_temp_xal = np.mean(delta_fct(Grid_xal)*delta_fct(Grid_Temp))
            var_xal = np.mean(delta_fct(Grid_xal)**2)
            var_temp = np.mean(delta_fct(Grid_Temp)**2)

            print('----- Investigating xal at z = ', z, ' is computed -------')

            if HO:
                PS_error_1 = mean_xal/(1+mean_xal) * auto_PS(delta_fct(Grid_xal)**2/(1+Grid_xal), box_dims=Lbox, kbins=kbins)[0]
                PS_error_2 = mean_xal**2/(1+mean_xal)**2*auto_PS(delta_fct(Grid_xal)**3/(1+Grid_xal), box_dims=Lbox, kbins=kbins)[0]
                PS_error_3 = mean_xal**3/(1+mean_xal)**3*auto_PS(delta_fct(Grid_xal)**4/(1+Grid_xal), box_dims=Lbox, kbins=kbins)[0]


                delta_U = delta_fct(Grid_xal)/(1+mean_xal)-delta_fct(Grid_xal)**2/(1+Grid_xal)*mean_xal/(1+mean_xal)
                PS_real_1 = auto_PS(delta_U, box_dims=Lbox, kbins=kbins)[0]

            Dict = {'z': z, 'k': kk, 'PS_xal': PS_xal, 'PS_xal_over_1_plus_xal': PS_xal_over_1_plus_xal,'PS_xal_term_x_temps_term':PS_xal_term_x_temps_term,
                    'xal': mean_xal, '1_ov_1_pl_xal': mean_1_ov_1_plus_xal,'var_1_ov_1_plus_xal':var_1_ov_1_plus_xal,'PS_xal_term_x_reio_term'      : PS_xal_term_x_reio_term,
                'PS_xal_Temp_matter_term': PS_xal_Temp_matter_term      ,'PS_xal_reio_Temp_matter_term'  : PS_xal_reio_Temp_matter_term ,'mean_xal'   : mean_xal,
                'mean_xHII'  : mean_xHII ,'mean_Temp'  : mean_Temp ,'mean_xal_reio' : mean_xal_reio,'mean_xal_temp_matter'  : mean_xal_temp_matter ,'mean_xal_reio_temp_matter'  : mean_xal_reio_temp_matter,'mean_xal_temp':mean_xal_temp,
                   'mean_xal_matter':mean_xal_matter ,'PS_xal_matter':PS_xal_matter,'PS_temp':PS_temp,'mean_Temp_term':mean_Temp_term,'cross_var_temp_xal':cross_var_temp_xal,'var_xal':var_xal,'var_temp':var_temp,'PS_xal_term_x_matter':PS_xal_term_x_matter,
                   'PS_temp_x_matter':PS_temp_x_matter }

            if HO :
                Dict_HO = {'PS_aT_a': PS_aT_a, 'PS_aT_T': PS_aT_T, 'PS_aa_a': PS_aa_a, 'PS_aa_T': PS_aa_T, 'PS_TT_a': PS_TT_a, 'PS_aa_aa':PS_aa_aa,'PS_aaa_a':PS_aaa_a,'PS_TT_T':PS_TT_T,'PS_TT_TT':PS_TT_TT,'PS_TTT_T':PS_TTT_T,
                    'PS_TT_T': PS_TT_T,'PS_error_1':PS_error_1,'PS_error_2':PS_error_2,'PS_error_3':PS_error_3,'PS_real_1':PS_real_1}
                Dict = Merge(Dict_HO, Dict)
            save_f(file='./physics/xal_data_' + str(Ncell) + '_' + param.sim.model_name + '_' + z_str + '.pkl',
                   obj=Dict)

    Barrier(comm)
    if rank == 0:
        gather_files(param, path = './physics/xal_data_', z_arr = z_arr, Ncell = Ncell)

        end_time = time.time()
        print('Finished investigating xal. It took in total: ', end_time - start_time)
        print('  ')



def investigate_expansion(param):

    if not os.path.isdir('./physics'):
        os.mkdir('./physics')

    start_time = time.time()
    print('Computing PS of xal/(1+xal).')

    comm, rank, size = initialise_mpi4py(param)

    kbins = def_k_bins(param)
    z_arr = def_redshifts(param)
    Ncell, Lbox = param.sim.Ncell, param.sim.Lbox
    nGrid = Ncell

    for ii, z in enumerate(z_arr):
        z = np.round(z, 2)
        z_str = z_string_format(z)
        if rank == ii % size:
            print('Core nbr', rank, 'is taking care of z = ', z)
            print('----- Investigating expansion for z =', z, '-------')
            Grid_Temp = load_grid(param, z=z, type='Tk')
            Grid_xHII = load_grid(param, z=z, type='bubbles')
            Grid_xal  = load_grid(param, z=z, type='lyal')
            delta_b   = load_delta_b(param, z_str)
            Grid_dTb  = load_grid(param, z=z, type='dTb')


            Grid_Temp, Grid_xHII, Grid_xal = format_grid_for_PS_measurement(Grid_Temp, Grid_xHII, Grid_xal, nGrid)
            Grid_xcoll = x_coll(z=z, Tk=Grid_Temp, xHI=(1 - Grid_xHII), rho_b=(delta_b + 1) * x_coll_coef(z,param))
            Grid_xtot  = Grid_xal + Grid_xcoll

            Grid_dTb_no_reio = dTb_fct(z=z, Tk=Grid_Temp, xtot=Grid_xtot, delta_b=delta_b, x_HII=np.array([0]), param=param)

            U = Grid_xtot/(1+Grid_xtot)
            V = 1-T_cmb(z)/Grid_Temp

            PS_UU,kk               = auto_PS(delta_fct(U), box_dims=Lbox, kbins=kbins)
            PS_VV,kk               = auto_PS(delta_fct(V), box_dims=Lbox, kbins=kbins)
            PS_bb,kk               = auto_PS(delta_b, box_dims=Lbox, kbins=kbins)

            delta_U, delta_V, delta_xHII =delta_fct(U), delta_fct(V), delta_fct(Grid_xHII)

            #### dTb with Taylor expansion
            Tk, x_tot, x_al, x_HII = np.mean(Grid_Temp), np.mean(Grid_xtot), np.mean(Grid_xal), np.mean(Grid_xHII)

            beta_r, beta_T, beta_a = -x_HII / (1 - x_HII), T_cmb(z) / (Tk - T_cmb(z)), x_al / x_tot / (1 + x_tot)
            dTb_fake = dTb_fct(z, Tk, x_tot, 0, x_HII, param)
            grid_dTb_Taylor = dTb_fake * (1 + delta_b) * (1+ beta_r * delta_xHII) * (1 + beta_a * delta_fct(Grid_xal)) \
                                        * (1 + beta_T * delta_fct(Grid_Temp))
            PS_dTb_Taylor = auto_PS(delta_fct(grid_dTb_Taylor), box_dims=Lbox, kbins=kbins)[0]


            grid_dTb_Taylor_no_rieo = dTb_fake * (1 + delta_b)  * (1 + beta_a * delta_fct(Grid_xal)) \
                              * (1 + beta_T * delta_fct(Grid_Temp))
            PS_dTb_Taylor_no_reio = auto_PS(delta_fct(grid_dTb_Taylor_no_rieo), box_dims=Lbox, kbins=kbins)[0]


            #### without reio field
            PS_UV = cross_PS(delta_U, delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_Ub = cross_PS(delta_U, delta_b, box_dims=Lbox, kbins=kbins)[0]
            PS_bV = cross_PS(delta_b, delta_V, box_dims=Lbox, kbins=kbins)[0]

            PS_U_UV = cross_PS(delta_U, delta_U * delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_U_Ub = cross_PS(delta_U, delta_U * delta_b, box_dims=Lbox, kbins=kbins)[0]
            PS_U_bV = cross_PS(delta_U, delta_b * delta_V, box_dims=Lbox, kbins=kbins)[0]

            PS_V_UV = cross_PS(delta_V, delta_U * delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_V_Ub = cross_PS(delta_V, delta_U * delta_b, box_dims=Lbox, kbins=kbins)[0]
            PS_V_bV = cross_PS(delta_V, delta_b * delta_V, box_dims=Lbox, kbins=kbins)[0]

            PS_b_UV = cross_PS(delta_b, delta_U * delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_b_Ub = cross_PS(delta_b, delta_U * delta_b, box_dims=Lbox, kbins=kbins)[0]
            PS_b_bV = cross_PS(delta_b, delta_b * delta_V, box_dims=Lbox, kbins=kbins)[0]

            PS_UV_UV = auto_PS(delta_U * delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_Ub_Ub = auto_PS(delta_U * delta_b, box_dims=Lbox, kbins=kbins)[0]
            PS_bV_bV = auto_PS(delta_b * delta_V, box_dims=Lbox, kbins=kbins)[0]

            PS_UV_Ub = cross_PS(delta_U * delta_V, delta_U * delta_b, box_dims=Lbox, kbins=kbins)[0]
            PS_Ub_bV = cross_PS(delta_U * delta_b, delta_b * delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_UV_bV = cross_PS(delta_U * delta_V, delta_b * delta_V, box_dims=Lbox, kbins=kbins)[0]

            PS_b_bUV = cross_PS(delta_b, delta_b * delta_U * delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_U_bUV = cross_PS(delta_U, delta_b * delta_U * delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_V_bUV = cross_PS(delta_V, delta_b * delta_U * delta_V, box_dims=Lbox, kbins=kbins)[0]

            PS_bU_bUV = cross_PS(delta_b * delta_U, delta_b * delta_U * delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_bV_bUV = cross_PS(delta_b * delta_V, delta_b * delta_U * delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_UV_bUV = cross_PS(delta_U * delta_V, delta_b * delta_U * delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_bUV_bUV = auto_PS(delta_b * delta_U * delta_V, box_dims=Lbox, kbins=kbins)[0]



            #### with reionisation
            PS_rr = auto_PS(delta_xHII, box_dims=Lbox, kbins=kbins)[0]
            PS_rU = cross_PS(delta_xHII,delta_U, box_dims=Lbox, kbins=kbins)[0]
            PS_rV = cross_PS(delta_xHII,delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_rb = cross_PS(delta_xHII,delta_b, box_dims=Lbox, kbins=kbins)[0]


            PS_rU_U = cross_PS(delta_xHII * delta_U, delta_U, box_dims=Lbox, kbins=kbins)[0]
            PS_r_UU = cross_PS(delta_xHII, delta_U ** 2, box_dims=Lbox, kbins=kbins)[0]

            PS_rb_b = cross_PS(delta_xHII * delta_b, delta_b, box_dims=Lbox, kbins=kbins)[0]

            PS_rV_V = cross_PS(delta_xHII * delta_V, delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_r_VV = cross_PS(delta_xHII, delta_V ** 2, box_dims=Lbox, kbins=kbins)[0]

            PS_Ub_r = cross_PS(delta_b * delta_U, delta_xHII, box_dims=Lbox, kbins=kbins)[0]
            PS_rb_U = cross_PS(delta_xHII * delta_b, delta_U, box_dims=Lbox, kbins=kbins)[0]
            PS_rU_b = cross_PS(delta_xHII * delta_U, delta_b, box_dims=Lbox, kbins=kbins)[0]

            PS_rV_b = cross_PS(delta_xHII * delta_V, delta_b, box_dims=Lbox, kbins=kbins)[0]
            PS_rb_V = cross_PS(delta_xHII * delta_b, delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_Vb_r = cross_PS(delta_V * delta_b, delta_xHII, box_dims=Lbox, kbins=kbins)[0]

            PS_UV_r = cross_PS(delta_U * delta_V, delta_xHII, box_dims=Lbox, kbins=kbins)[0]
            PS_Ur_V = cross_PS(delta_xHII * delta_U, delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_Vr_U = cross_PS(delta_xHII * delta_V, delta_U, box_dims=Lbox, kbins=kbins)[0]

            PS_Vr_r = cross_PS(delta_V * delta_xHII, delta_xHII, box_dims=Lbox, kbins=kbins)[0]
            PS_Ur_r = cross_PS(delta_U * delta_xHII, delta_xHII, box_dims=Lbox, kbins=kbins)[0]
            PS_br_r = cross_PS(delta_b * delta_xHII, delta_xHII, box_dims=Lbox, kbins=kbins)[0]

            PS_rV_rV = auto_PS(delta_xHII * delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_rU_rU = auto_PS(delta_xHII * delta_U, box_dims=Lbox, kbins=kbins)[0]
            PS_rb_rb = auto_PS(delta_xHII * delta_b, box_dims=Lbox, kbins=kbins)[0]

            PS_rbU_r = cross_PS(delta_xHII * delta_b * delta_U, delta_xHII, box_dims=Lbox, kbins=kbins)[0]
            PS_rVU_r = cross_PS(delta_xHII * delta_V * delta_U, delta_xHII, box_dims=Lbox, kbins=kbins)[0]
            PS_rVb_r = cross_PS(delta_xHII * delta_V * delta_b, delta_xHII, box_dims=Lbox, kbins=kbins)[0]

            PS_rb_rU = cross_PS(delta_xHII * delta_b, delta_xHII * delta_U, box_dims=Lbox, kbins=kbins)[0]
            PS_rV_rU = cross_PS(delta_xHII * delta_V, delta_xHII * delta_U, box_dims=Lbox, kbins=kbins)[0]
            PS_rV_rb = cross_PS(delta_xHII * delta_V, delta_xHII * delta_b, box_dims=Lbox, kbins=kbins)[0]


            #### terms that matter during reio/heating (+ reionisation)
            #### these matter the most : data_UV['PS_bV_bV']+ 2*(data_UV['PS_b_bV']+data_UV['PS_V_bV'])

            PS_rbV_bV  = cross_PS(delta_xHII * delta_b *  delta_V, delta_b *  delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_rbV_rbV = auto_PS(delta_xHII * delta_b *  delta_V, box_dims=Lbox, kbins=kbins)[0]

            PS_rb_bV  = cross_PS(delta_xHII * delta_b , delta_b * delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_b_rbV  = cross_PS(delta_b , delta_xHII * delta_b * delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_rb_rbV  = cross_PS(delta_xHII * delta_b , delta_xHII *  delta_b * delta_V, box_dims=Lbox, kbins=kbins)[0]

            PS_V_rbV   = cross_PS(delta_V , delta_xHII * delta_b * delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_rV_bV   = cross_PS(delta_xHII * delta_V,  delta_b * delta_V, box_dims=Lbox, kbins=kbins)[0]
            PS_rV_rbV  = cross_PS(delta_xHII * delta_V , delta_xHII * delta_b * delta_V, box_dims=Lbox, kbins=kbins)[0]


            #### extra terms that for reio/lyal (+ reionisation)
            ####same as avobe with V but replaced all V by U

            PS_rbU_bU  = cross_PS(delta_xHII * delta_b *  delta_U, delta_b *  delta_U, box_dims=Lbox, kbins=kbins)[0]
            PS_rbU_rbU = auto_PS(delta_xHII * delta_b *  delta_U, box_dims=Lbox, kbins=kbins)[0]

            PS_rb_bU  = cross_PS(delta_xHII * delta_b , delta_b * delta_U, box_dims=Lbox, kbins=kbins)[0]
            PS_b_rbU  = cross_PS(delta_b , delta_xHII * delta_b * delta_U, box_dims=Lbox, kbins=kbins)[0]
            PS_rb_rbU  = cross_PS(delta_xHII * delta_b , delta_xHII *  delta_b * delta_U, box_dims=Lbox, kbins=kbins)[0]

            PS_U_rbU   = cross_PS(delta_U , delta_xHII * delta_b * delta_U, box_dims=Lbox, kbins=kbins)[0]
            PS_rU_bU   = cross_PS(delta_xHII * delta_U,  delta_b * delta_U, box_dims=Lbox, kbins=kbins)[0]
            PS_rU_rbU  = cross_PS(delta_xHII * delta_U , delta_xHII * delta_b * delta_U, box_dims=Lbox, kbins=kbins)[0]


            ## dTb
            PS_dTb = auto_PS(delta_fct(Grid_dTb), box_dims=Lbox, kbins=kbins)[0]
            PS_dTb_no_reio = auto_PS(delta_fct(Grid_dTb_no_reio), box_dims=Lbox, kbins=kbins)[0]


            ## corr fct
            Xi_UV = np.mean(delta_V * delta_U)
            Xi_Ub = np.mean(delta_b * delta_U)
            Xi_Vb = np.mean(delta_b * delta_V)
            Xi_rV = np.mean(delta_xHII * delta_V)
            Xi_rU = np.mean(delta_xHII * delta_U)
            Xi_rb = np.mean(delta_xHII * delta_b)

            Dict = {'z': z, 'k': kk,'dTb':np.mean(Grid_dTb),'dTb_no_reio':np.mean(Grid_dTb_no_reio),
                    'x_al':x_al,'Tk':Tk,'x_HII':x_HII,
                    'U': np.mean(U),'V': np.mean(V), 'x_coll':np.mean(Grid_xcoll),
                    'Xi_UV': Xi_UV, 'Xi_Ub': Xi_Ub, 'Xi_Vb': Xi_Vb,
                    'Xi_rV': Xi_rV, 'Xi_rU': Xi_rU, 'Xi_rb': Xi_rb,
                    'PS_UU': PS_UU ,'PS_VV': PS_VV ,'PS_bb': PS_bb , 'PS_UV': PS_UV,
                    'PS_Ub': PS_Ub,'PS_bV':PS_bV,'PS_dTb':PS_dTb,'PS_dTb_no_reio':PS_dTb_no_reio,
                    'PS_dTb_Taylor':PS_dTb_Taylor,'dTb_fake':dTb_fake, 'dTb_Taylor':np.mean(grid_dTb_Taylor),
                    'PS_dTb_Taylor_no_reio':PS_dTb_Taylor_no_reio,'dTb_Taylor_no_reio':np.mean(grid_dTb_Taylor_no_rieo),

                    'PS_U_UV': PS_U_UV, 'PS_U_Ub': PS_U_Ub, 'PS_U_bV': PS_U_bV,
                    'PS_V_UV': PS_V_UV, 'PS_V_Ub': PS_V_Ub, 'PS_V_bV': PS_V_bV,
                    'PS_b_UV': PS_b_UV, 'PS_b_Ub': PS_b_Ub, 'PS_b_bV': PS_b_bV,
                    'PS_UV_UV':PS_UV_UV,'PS_Ub_Ub':PS_Ub_Ub,'PS_bV_bV':PS_bV_bV,
                    'PS_UV_Ub':PS_UV_Ub, 'PS_Ub_bV':PS_Ub_bV,'PS_UV_bV':PS_UV_bV,
                    'PS_b_bUV':PS_b_bUV, 'PS_U_bUV':PS_U_bUV, 'PS_V_bUV':PS_V_bUV,
                    'PS_bU_bUV':PS_bU_bUV,'PS_bV_bUV':PS_bV_bUV,'PS_UV_bUV':PS_UV_bUV,
                    'PS_bUV_bUV':PS_bUV_bUV ,

                        'PS_rr':PS_rr,'PS_rU':PS_rU,'PS_rV':PS_rV,'PS_rb':PS_rb,
                              'PS_rU_U': PS_rU_U, 'PS_rb_b': PS_rb_b, 'PS_rV_V': PS_rV_V,
                              'PS_Ub_r': PS_Ub_r, 'PS_rb_U': PS_rb_U, 'PS_rU_b': PS_rU_b,
                              'PS_rV_b': PS_rV_b, 'PS_rb_V': PS_rb_V, 'PS_Vb_r': PS_Vb_r, 'PS_UV_r': PS_UV_r,
                              'PS_Ur_V': PS_Ur_V, 'PS_Vr_U': PS_Vr_U, 'PS_Vr_r': PS_Vr_r, 'PS_Ur_r': PS_Ur_r,
                              'PS_br_r': PS_br_r, 'PS_rV_rV': PS_rV_rV, 'PS_rU_rU': PS_rU_rU, 'PS_rb_rb': PS_rb_rb,
                              'PS_rbU_r': PS_rbU_r, 'PS_rVU_r': PS_rVU_r,

                              'PS_rVb_r': PS_rVb_r, 'PS_rb_rU': PS_rb_rU, 'PS_rV_rU': PS_rV_rU, 'PS_rV_rb': PS_rV_rb,

                            'PS_rbV_bV' :PS_rbV_bV   ,'PS_rbV_rbV':PS_rbV_rbV ,'PS_rb_bV'  :PS_rb_bV ,'PS_b_rbV'  :PS_b_rbV,
                            'PS_rb_rbV' :PS_rb_rbV   ,'PS_V_rbV'  :PS_V_rbV    ,'PS_rV_bV'  :PS_rV_bV ,'PS_rV_rbV' :PS_rV_rbV  ,

                    'PS_rbU_bU': PS_rbU_bU, 'PS_rbU_rbU': PS_rbU_rbU, 'PS_rb_bU': PS_rb_bU, 'PS_b_rbU': PS_b_rbU,
                    'PS_rb_rbU': PS_rb_rbU, 'PS_U_rbU': PS_U_rbU, 'PS_rU_bU': PS_rU_bU, 'PS_rU_rbU': PS_rU_rbU,
                    }

            save_f(file='./physics/data_expansion_U_V_' + str(Ncell) + '_' + param.sim.model_name + '_' + z_str + '.pkl',
                   obj=Dict)
            print('----- Done for z =', z, '-------')

    Barrier(comm)

    if rank == 0:
        gather_files(param, path = './physics/data_expansion_U_V_', z_arr = z_arr, Ncell = Ncell)

        end_time = time.time()
        print('Finished calculating U, V, delta_b auto and cross power spectra. It took in total: ', end_time - start_time)
        print('  ')




def investigate_Tylor_no_reio(param):

    if not os.path.isdir('./physics'):
        os.mkdir('./physics')

    start_time = time.time()
    print('Computing investigate_Tylor_no_reio.')

    comm, rank, size = initialise_mpi4py(param)

    kbins = def_k_bins(param)
    z_arr = def_redshifts(param)
    Ncell, Lbox = param.sim.Ncell, param.sim.Lbox
    nGrid = Ncell

    for ii, z in enumerate(z_arr):
        z = np.round(z, 2)
        z_str = z_string_format(z)
        if rank == ii % size:
            print('Core nbr', rank, 'is taking care of z = ', z)
            print('----- Investigating expansion for z =', z, '-------')
            Grid_Temp = load_grid(param, z=z, type='Tk')
            Grid_xHII = load_grid(param, z=z, type='bubbles')
            Grid_xal  = load_grid(param, z=z, type='lyal')
            delta_b   = load_delta_b(param, z_str)
            Grid_dTb  = load_grid(param, z=z, type='dTb')


            Grid_Temp, Grid_xHII, Grid_xal = format_grid_for_PS_measurement(Grid_Temp, Grid_xHII, Grid_xal, nGrid)
            Grid_xcoll = x_coll(z=z, Tk=Grid_Temp, xHI=(1 - Grid_xHII), rho_b=(delta_b + 1) * x_coll_coef(z,param))
            Grid_xtot  = Grid_xal + Grid_xcoll
            PS_bb, kk = auto_PS(delta_b, box_dims=Lbox, kbins=kbins)
            Grid_dTb_no_reio = dTb_fct(z=z, Tk=Grid_Temp, xtot=Grid_xtot, delta_b=delta_b, x_HII=np.array([0]), param=param)

            U = Grid_xtot/(1+Grid_xtot)
            V = 1-T_cmb(z)/Grid_Temp

            delta_U, delta_V, delta_xHII =delta_fct(U), delta_fct(V), delta_fct(Grid_xHII)

            #### dTb with Taylor expansion
            Tk, x_tot, x_al, x_HII = np.mean(Grid_Temp), np.mean(Grid_xtot), np.mean(Grid_xal), np.mean(Grid_xHII)

            beta_r, beta_T, beta_a = -x_HII / (1 - x_HII), T_cmb(z) / (Tk - T_cmb(z)), x_al / x_tot / (1 + x_tot)
            dTb_fake = dTb_fct(z, Tk, x_tot, 0, x_HII, param)
            grid_dTb_Taylor = dTb_fake * (1 + delta_b) * (1+ beta_r * delta_xHII) * (1 + beta_a * delta_fct(Grid_xal)) \
                                        * (1 + beta_T * delta_fct(Grid_Temp))

            PS_dTb_Taylor = auto_PS(delta_fct(grid_dTb_Taylor), box_dims=Lbox, kbins=kbins)[0]

            dTb_fake_no_reio = dTb_fct(z, Tk, x_tot, 0, 0, param)
            grid_dTb_Taylor_no_rieo = dTb_fake_no_reio * (1 + delta_b)  * (1 + beta_a * delta_fct(Grid_xal)) \
                              * (1 + beta_T * delta_fct(Grid_Temp))
            PS_dTb_Taylor_no_reio = auto_PS(delta_fct(grid_dTb_Taylor_no_rieo), box_dims=Lbox, kbins=kbins)[0]

            ## dTb
            PS_dTb = auto_PS(delta_fct(Grid_dTb), box_dims=Lbox, kbins=kbins)[0]
            PS_dTb_no_reio = auto_PS(delta_fct(Grid_dTb_no_reio), box_dims=Lbox, kbins=kbins)[0]


            Dict = {'z': z, 'k': kk,'dTb':np.mean(Grid_dTb),'dTb_no_reio':np.mean(Grid_dTb_no_reio),
                    'x_al':x_al,'Tk':Tk,'x_HII':x_HII,
                    'U': np.mean(U),'V': np.mean(V), 'x_coll':np.mean(Grid_xcoll),
                    'PS_dTb':PS_dTb,'PS_dTb_no_reio':PS_dTb_no_reio,
                    'PS_dTb_Taylor':PS_dTb_Taylor,'dTb_fake':dTb_fake, 'dTb_Taylor':np.mean(grid_dTb_Taylor),
                    'PS_dTb_Taylor_no_reio':PS_dTb_Taylor_no_reio,'dTb_fake_no_reio':dTb_fake_no_reio,'dTb_Taylor_no_reio':np.mean(grid_dTb_Taylor_no_rieo)}

            save_f(file='./physics/Taylor_no_reio_' + str(Ncell) + '_' + param.sim.model_name + '_' + z_str + '.pkl',
                   obj=Dict)
            print('----- Done for z =', z, '-------')

    Barrier(comm)

    if rank == 0:
        gather_files(param, path = './physics/Taylor_no_reio_', z_arr = z_arr, Ncell = Ncell)

        end_time = time.time()
        print('Finished calculating TaylorNoReio. It took in total: ', end_time - start_time)
        print('  ')







###COMPUTING THE FRACTION OF POINTS IN LYAL AND Tk THAT MAKE THE TAYLOR EXP WRONG.

def compute_frac_non_lin_points(param, k_values=[0.1]):
    # k_values : value to smooth the field in h/Mpc
    # Dictionnary output contains :
    # frac_nl_lyal :
    #     2d array of shape (zz,k_values+1). Containing fraction of points such that xal/(1+xal)*dxal >1
    #     first k value is the unsmoothed field, the other are smoothed over scale k
    # frac_nl_temp :  same but for delta_T

    start_time = time.time()
    print('Computing fractions of points where Taylor exp is not valid.')

    comm, rank, size = initialise_mpi4py(param)

    z_arr = def_redshifts(param)
    Ncell, Lbox = param.sim.Ncell, param.sim.Lbox
    nGrid = Ncell

    for ii, z in enumerate(z_arr):
        z = np.round(z, 2)
        z_str = z_string_format(z)
        if rank == ii % size:
            print('Core nbr', rank, 'is taking care of z = ', z)
            print('----- Going for z =', z, '-------')
            # read in grids
            Grid_Temp = load_grid(param, z=z, type='Tk')
            Grid_xal = load_grid(param, z=z, type='lyal')
            xal = np.mean(Grid_xal)

            # the array containing the fraction of points for k=0, and then kvalues
            frac_nl_lyal = np.zeros((len(k_values) + 1))
            frac_nl_Tk = np.zeros((len(k_values) + 1))

            ##define the fields that have to be <<1 for the Taylor exp to be valid
            field_lyal = xal / (1 + xal) * delta_fct(Grid_xal)
            field_temp = delta_fct(Grid_Temp)

            frac_nl_lyal[0] = len(np.where(field_lyal > 1)[0]) / nGrid ** 3
            frac_nl_Tk[0] = len(np.where(field_temp > 1)[0]) / nGrid ** 3

            if len(k_values)>0:
                for ik, kk in enumerate(k_values):
                    Rsmoothing = np.pi / kk
                    smoothed_lyal = smooth_field(field_lyal, Rsmoothing, Lbox, nGrid)
                    smoothed_Tk = smooth_field(field_temp, Rsmoothing, Lbox, nGrid)

                    frac_nl_lyal[ik + 1] = len(np.where(smoothed_lyal > 1)[0]) / nGrid ** 3
                    frac_nl_Tk[ik + 1] = len(np.where(smoothed_Tk > 1)[0]) / nGrid ** 3


            k = np.concatenate((np.array([0]), np.array(k_values)))
            Dict = {'z': z, 'frac_nl_lyal': frac_nl_lyal, 'frac_nl_Tk': frac_nl_Tk, 'k':k}

            save_f(file='./physics/frac_nl_points_' + str(Ncell) + '_' + param.sim.model_name + '_' + z_str + '.pkl',
                   obj=Dict)
            print('----- Done for z =', z, '-------')

    #comm.Barrier()
    Barrier(comm)

    if rank == 0:
        gather_files(param, path='./physics/frac_nl_points_', z_arr=z_arr, Ncell=Ncell)

        end_time = time.time()
        print('Finished calculating the fraction of points that are non linear for lyal and Tk. It took in total: ',
              end_time - start_time)
        print('  ')







def investigate_expansion_mean_xcoll(param):

    if not os.path.isdir('./physics'):
        os.mkdir('./physics')

    start_time = time.time()
    print('Computing PS of xal/(1+xal).')

    comm, rank, size = initialise_mpi4py(param)

    kbins = def_k_bins(param)
    z_arr = def_redshifts(param)
    Ncell, Lbox = param.sim.Ncell, param.sim.Lbox
    nGrid = Ncell

    for ii, z in enumerate(z_arr):
        z = np.round(z, 2)
        z_str = z_string_format(z)
        if rank == ii % size:
            print('Core nbr', rank, 'is taking care of z = ', z)
            print('----- Investigating expansion for z =', z, '-------')
            Grid_Temp = load_grid(param, z=z, type='Tk')
            Grid_xHII = load_grid(param, z=z, type='bubbles')
            Grid_xal  = load_grid(param, z=z, type='lyal')
            delta_b   = load_delta_b(param, z_str)
            Grid_dTb  = load_grid(param, z=z, type='dTb')


            Grid_Temp, Grid_xHII, Grid_xal = format_grid_for_PS_measurement(Grid_Temp, Grid_xHII, Grid_xal, nGrid)
            Grid_xcoll = np.mean(x_coll(z=z, Tk=Grid_Temp, xHI=(1 - Grid_xHII), rho_b=(delta_b + 1) * x_coll_coef(z,param)))
            Grid_xtot  = Grid_xal + Grid_xcoll

            U = Grid_xtot/(1+Grid_xtot)
            V = 1-T_cmb(z)/Grid_Temp

            PS_UU,kk               = auto_PS(delta_fct(U), box_dims=Lbox, kbins=kbins)
            PS_VV,kk               = auto_PS(delta_fct(V), box_dims=Lbox, kbins=kbins)

            Dict = {'z': z, 'k': kk,
                    'U': np.mean(U),'V': np.mean(V),
                    'PS_UU': PS_UU ,'PS_VV': PS_VV }

            save_f(file='./physics/data_expansion_U_V_meanxcoll_' + str(Ncell) + '_' + param.sim.model_name + '_' + z_str + '.pkl',
                   obj=Dict)
            print('----- Done for z =', z, '-------')

    Barrier(comm)

    if rank == 0:
        gather_files(param, path = './physics/data_expansion_U_V_meanxcoll_', z_arr = z_arr, Ncell = Ncell)

        end_time = time.time()
        print('Finished calculating U, V, delta_b auto and cross power spectra. It took in total: ', end_time - start_time)
        print('  ')



