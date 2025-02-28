"""
Global quantity computed from the halo catalogs.
"""

import os.path
import numpy as np

from .cosmo import T_adiab, dTb_factor
from .couplings import S_alpha, x_coll
from .constants import *
from .astro import f_star_Halo
from .functions import *


def global_xhii_approx(param):
    zz = []
    xHII = []
    z_arr = def_redshifts(param)
    for ii, z in enumerate(z_arr):
        z_str = z_string_format(z)
        halo_catalog = load_halo(param, z_str)
        zz_, xHII_ = xHII_approx(param, halo_catalog)

        print(zz_, 'done.')
        zz.append(zz_)
        xHII.append(xHII_)
    return np.array((zz, xHII))


def xHII_approx(parameters: Parameters, halo_catalog):
    ## compute mean ion fraction from Rbubble values and halo catalog.  for the simple bubble solver
    LBox = parameters.simulation.Lbox  # Mpc/h
    M_Bin = np.logspace(np.log10(parameters.simulation.halo_mass_bin_min), np.log10(parameters.simulation.halo_mass_bin_min), parameters.simulation.halo_mass_bin_n, base=10)
    model_name = parameters.simulation.model_name
    H_Masses = halo_catalog['M']
    z = halo_catalog['z']

    grid_model = load_f('./profiles/' + model_name + '.pkl')
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]

    H_Masses = np.delete(H_Masses,
                         np.where(H_Masses < parameters.source.halo_mass_min))  ## remove element smaller than minimum SF halo mass.

    Mh_z_bin = grid_model.Mh_history[ind_z, :]  # M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))
    Mh_bin_array = np.concatenate(([Mh_z_bin[0] / 2], np.sqrt(Mh_z_bin[1:] * Mh_z_bin[:-1]),
                                   [2 * Mh_z_bin[-1]]))  ## shape of binning array is (len(M_bin)+1)
    # Mh_bin_array = Mh_bin_array * np.exp(-param.source.alpha_MAR * (z - z_start))
    bins = np.digitize(np.log10(H_Masses), np.log10(Mh_bin_array), right=False)

    # if H_Masses is digitized to Mh_bin_array[i] it means it should take the value of M_Bin[i-1] (bin 0 is what's on the left...)
    # M_Bin                0.   1.    2.    3.  ...
    # Mh_bin_array     0.  | 1. |  2. |  3. |  4. ....
    bins, number_per_bins = np.unique(bins, return_counts=True)
    bins = bins.clip(max=len(M_Bin))
    ### Ionisation
    Bubble_radii = grid_model.R_bubble[ind_z, bins - 1]
    Bubble_covol = 4 / 3 * np.pi * Bubble_radii ** 3
    Ionized_covolume = np.sum(Bubble_covol * number_per_bins)
    Ionized_fraction = Ionized_covolume / LBox ** 3
    x_HII = Ionized_fraction.clip(max=1)

    return zgrid, x_HII


def compute_glob_qty(parameters: Parameters):
    print('Computing global quantities (sfrd, Tk, xHII, dTb, xal, xcoll) from 1D profiles and halo catalogs....')
    LBox = parameters.simulation.Lbox  # Mpc/h
    model_name = parameters.simulation.model_name
    M_Bin = np.logspace(np.log10(parameters.simulation.halo_mass_bin_min), np.log10(parameters.simulation.halo_mass_bin_max), parameters.simulation.halo_mass_bin_n, base=10)
    grid_model = load_f('./profiles/' + model_name + '.pkl')

    Om, Ob, h0 = parameters.cosmology.Om, parameters.cosmology.Ob, parameters.cosmology.h
    factor = dTb_factor(parameters)

    zz = []
    xHII = []
    Tk = []
    sfrd = []
    s_alpha = []
    x_alpha = []
    dTb_arr = []
    xcoll_arr = []

    z_arr = def_redshifts(parameters)
    for ii, z in enumerate(z_arr):
        z_str = z_string_format(z)
        halo_catalog = load_halo(parameters, z_str)
        H_Masses = halo_catalog['M']
        H_Masses = np.delete(H_Masses, np.where(
            H_Masses < parameters.source.halo_mass_min))  ## remove element smaller than minimum SF halo mass.
        z = halo_catalog['z']

        ind_z = np.argmin(np.abs(grid_model.z_history - z))
        radial_grid = grid_model.r_grid_cell / (1 + z)  # pMpc/h

        Mh_z_bin = grid_model.Mh_history[ind_z, :]  # M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))
        Mh_bin_array = np.concatenate(([Mh_z_bin[0] / 2], np.sqrt(Mh_z_bin[1:] * Mh_z_bin[:-1]),
                                       [2 * Mh_z_bin[-1]]))  ## shape of binning array is (len(M_bin)+1)
        # Mh_bin_array = Mh_bin_array * np.exp(-param.source.alpha_MAR * (z - z_start))
        bins = np.digitize(np.log10(H_Masses), np.log10(Mh_bin_array), right=False)

        # if H_Masses is digitized to Mh_bin_array[i] it means it should take the value of M_Bin[i-1] (bin 0 is what's on the left...)
        # M_Bin                0.   1.    2.    3.  ...
        # Mh_bin_array     0.  | 1. |  2. |  3. |  4. ....
        bins, number_per_bins = np.unique(bins, return_counts=True)
        bins = bins.clip(max=len(M_Bin))

        ### Ionisation
        Bubble_radii = grid_model.R_bubble[ind_z, bins - 1]
        Bubble_covol = 4 / 3 * np.pi * Bubble_radii ** 3
        Ionized_covolume = np.sum(Bubble_covol * number_per_bins)
        Ionized_fraction = Ionized_covolume / LBox ** 3
        x_HII = Ionized_fraction.clip(max=1)

        ### Temperature
        Temp_profile = grid_model.rho_heat[ind_z, :, bins - 1]
        temp_volume = np.trapz(4 * np.pi * radial_grid ** 2 * Temp_profile, radial_grid, axis=1) * number_per_bins
        Temp = np.sum(temp_volume) / (LBox / (1 + z)) ** 3  ##physical volume !!

        ### Lyman-alpha
        r_lyal = grid_model.r_lyal  # np.logspace(-5, 2, 1000,base=10)  ##    physical distance for lyal profile. Never goes further away than 100 pMpc/h (checked)
        rho_alpha_ = grid_model.rho_alpha[ind_z, :,
                     bins - 1]  # rho_alpha(r_lyal, Mh_z_bin[bins - 1][:, None], z, param)
        x_alpha_prof = 1.81e11 * rho_alpha_ / (1 + z)
        xal_volume = np.sum(
            np.trapz(4 * np.pi * r_lyal ** 2 * x_alpha_prof, r_lyal, axis=1) * number_per_bins)  ##physical volume !!
        x_al = xal_volume / (LBox / (1 + z)) ** 3

        ### SFRD
        Temp = Temp + T_adiab(z, parameters)
        dMstar_dt = grid_model.dMh_dt[ind_z, :] * f_star_Halo(parameters,
                                                              Mh_z_bin) * parameters.cosmology.Ob / parameters.cosmology.Om  # param.source.alpha_MAR * Mh_z_bin * (z + 1) * Hubble(z, param) * f_star_Halo(param, Mh_z_bin)
        dMstar_dt[np.where(Mh_z_bin < parameters.source.halo_mass_min)] = 0
        SFRD = np.sum(number_per_bins * dMstar_dt[bins - 1])
        SFRD = SFRD / LBox ** 3  #### [(Msol/h) / yr /(cMpc/h)**3]

        coef = rhoc0 * h0 ** 2 * Ob * (1 + z) ** 3 * M_sun / cm_per_Mpc ** 3 / m_H
        Salpha = S_alpha(z, Temp, 1 - x_HII)
        x_al = x_al * Salpha / 4 / np.pi
        xcoll = x_coll(z=z, Tk=Temp, xHI=(1 - x_HII), rho_b=coef)
        xtot = xcoll + x_al
        dTb = factor * np.sqrt(1 + z) * (1 - Tcmb0 * (1 + z) / Temp) * (1 - x_HII) * xtot / (1 + xtot)

        Tk.append(Temp)
        zz.append(z)
        xHII.append(x_HII)
        sfrd.append(SFRD)
        s_alpha.append(Salpha)
        x_alpha.append(x_al)
        dTb_arr.append(dTb)
        xcoll_arr.append(xcoll)


    print( '....done. Returns a dictionnary.')

    zz, Tk, xHII, sfrd, s_alpha, x_alpha, dTb_arr, xcoll_arr = np.array(zz), np.array(Tk), np.array(xHII), np.array(
        sfrd), np.array(s_alpha), np.array(x_alpha), np.array(dTb_arr), np.array(xcoll_arr)
    #matrice = np.array([zz, Tk, xHII, sfrd, s_alpha, x_alpha, dTb_arr, xcoll_arr])
    #zz, Tk, xHII, sfrd, s_alpha, x_alpha, dTb_arr, xcoll_arr = matrice[:, matrice[0].argsort()]  ## sort according to zz

    return {'z': zz, 'Tk': Tk, 'x_HII': xHII, 'sfrd': sfrd, 'S_al': s_alpha, 'x_al': x_alpha, 'dTb': dTb_arr,
            'xcoll': xcoll_arr}


def compute_sfrd(param, zz_, Maccr, dM_dt_accr):
    model_name = param.sim.model_name
    if os.path.exists('./physics/sfrd_' + model_name + '.txt'):
        print('Reading SFRD from ./physics/sfrd_' + model_name + '.txt')
        data = np.loadtxt('./physics/sfrd_' + model_name + '.txt')
        zz, sfrd = data[0], data[1]
    else:
        print('Computing the SFRD [(Msol/h) / yr /(cMpc/h)**3] from halo catalogs and source models parameters....')
        LBox = param.sim.Lbox  # Mpc/h
        M_Bin = np.logspace(np.log10(param.sim.Mh_bin_min), np.log10(param.sim.Mh_bin_max), param.sim.binn, base=10)

        zz = []
        sfrd = []

        z_arr = def_redshifts(param)
        for ii, z in enumerate(z_arr):
            z_str = z_string_format(z)
            halo_catalog = load_halo(param, z_str)
            H_Masses = halo_catalog['M']
            z = halo_catalog['z']
            ind_z = np.argmin(np.abs(zz_ - z))
            Mh_z_bin = Maccr[ind_z, :]  # M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))
            Mh_bin_array = np.concatenate(([Mh_z_bin[0] / 2], np.sqrt(Mh_z_bin[1:] * Mh_z_bin[:-1]),
                                           [2 * Mh_z_bin[-1]]))  ## shape of binning array is (len(M_bin)+1)
            # Mh_bin_array = Mh_bin_array * np.exp(-param.source.alpha_MAR * (z - z_start))
            bins = np.digitize(np.log10(H_Masses), np.log10(Mh_bin_array), right=False)

            # if H_Masses is digitized to Mh_bin_array[i] it means it should take the value of M_Bin[i-1] (bin 0 is what's on the left...)
            # M_Bin                0.   1.    2.    3.  ...
            # Mh_bin_array     0.  | 1. |  2. |  3. |  4. ....
            bins, number_per_bins = np.unique(bins, return_counts=True)
            bins_clipped = bins.clip(max=len(M_Bin))
            if not np.array_equal(bins_clipped, bins):
                print('WARNING : you may want to use a higher param.source.Mh_bin_max.')
            ### if bins[0] is 1, it means H_Masses[0] is between Mh_bin_array[0] and Mh_bin_array[1]. Therefore is takes the profile of M_Bin[0]

            dMstar_dt = dM_dt_accr[ind_z, :] * f_star_Halo(param,
                                                           Mh_z_bin) * param.cosmo.Ob / param.cosmo.Om  # param.source.alpha_MAR * Mh_z_bin * (z + 1) * Hubble(z, param) * f_star_Halo(param, Mh_z_bin)
            dMstar_dt[np.where(Mh_z_bin < param.source.M_min)] = 0

            SFRD = np.sum(number_per_bins * dMstar_dt[bins_clipped - 1])

            SFRD = SFRD / LBox ** 3  ## [(Msol/h) / yr /(cMpc/h)**3]
            sfrd.append(SFRD)
            zz.append(z)
            print('z=', z, 'finished.')

        zz, sfrd = np.array(zz), np.array(sfrd)
        matrice = np.array([zz, sfrd])
        zz, sfrd = matrice[:, matrice[0].argsort()]  ## sort according to zz
        np.savetxt('./physics/sfrd_' + model_name + '.txt', (zz, sfrd))

    print('....Done computing SFRD')
    return zz, sfrd

