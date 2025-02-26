"""
Some old functions that we store here before trashing them, one day.
"""
import numpy as np

from pathlib import Path





################## OLD FUNCTIONS WHERE WE LOOP OVER HALO MASS BINS. USELESS

def global_signal(param, heat=None, redshifting='yes', simple_model=False):
    catalog_dir = param.sim.halo_catalogs
    xHII = []
    G_heat = []
    z = []
    sfrd = []
    Jalpha = []
    Gheat_GS_style = []
    xal = []
    heat_per_baryon = []
    T_gas = []
    T_gas_neutral = []
    for ii, filename in enumerate(os.listdir(catalog_dir)):
        catalog = catalog_dir + filename
        halo_catalog = load_f(catalog)  # Read_Rockstar(catalog)

        if heat is not None:
            heat_per_baryon.append(G_heat_approx(param, halo_catalog))
        else:
            heat_per_baryon.append(0)

        zz_, SFRD = sfrd_approx(param, halo_catalog)
        zz_, x_HII = xHII_approx_old(param, halo_catalog, simple_model)
        Jalpha_, x_alpha_ = mean_Jalpha_approx(param, halo_catalog, simple_model)
        Erange, Jal, Gam_heat = mean_J_xray_nu_approx(param, halo_catalog, simple_model, density_normalization=1,
                                                      redshifting=redshifting)
        Tk, Tkneutral = Tgas_from_profiles(param, halo_catalog, simple_model=simple_model)
        itlH = sigma_HI(Erange) * Jal * (Erange - E_HI)
        # itl_2 = sigma_s * min(x_HII, 1) / m_e_eV * (I2_Ta + T_grid * I2_Tb)

        if np.any(Erange == 0):  # similar to : if E is not 0
            Gheat_GS_style.append(0)
        else:
            Gheat_GS_style.append(np.trapz(itlH, Erange * Hz_per_eV))  # eV/s
            # Gheat_GS_style.append(np.trapz(itlH,Erange*Hz_per_eV)) # eV/s

        T_gas.append(Tk)
        T_gas_neutral.append(Tkneutral)
        xal.append(x_alpha_)
        Jalpha.append(Jalpha_)
        xHII.append(min(x_HII, 1))
        z.append(zz_)
        G_heat.append(Gam_heat)
        sfrd.append(SFRD)

    sfrd, xHII, z_array, G_heat, Jalpha, xal, Gheat_GS_style, heat_per_baryon, T_gas, T_gas_neutral = np.array(
        sfrd), np.array(xHII), np.array(z), np.array(G_heat), np.array(Jalpha), np.array(xal), np.array(
        Gheat_GS_style), np.array(heat_per_baryon), np.array(T_gas), np.array(T_gas_neutral)
    matrice = np.array(
        [z_array, xHII, sfrd, G_heat, Jalpha, xal, Gheat_GS_style, heat_per_baryon, T_gas, T_gas_neutral])
    z, xHII, sfrd, G_heat, Jalpha, xal, Gheat_GS_style, heat_per_baryon, T_gas, T_gas_neutral = matrice[:, matrice[
                                                                                                               0].argsort()]  ## sort according to zarray

    T_gas += T_adiab(z, param)
    T_gas_neutral += T_adiab(z, param)
    x_col = x_coll(z=z, Tk=T_gas, xHI=1 - xHII,
                   rho_b=rhoc0 * param.cosmo.h ** 2 * param.cosmo.Ob * (1 + z) ** 3 * M_sun / cm_per_Mpc ** 3 / m_H)

    Jal_coda_style = J_alpha_n(z, sfrd, param)
    xal_coda_style = np.sum(Jal_coda_style[1::], axis=0) * S_alpha(z, T_gas, 1 - xHII) * 1.81e11 / (1 + z)

    Om, Ob, h0 = param.cosmo.Om, param.cosmo.Ob, param.cosmo.h
    factor = 27 * (1 / 10) ** 0.5 * (Ob * h0 ** 2 / 0.023) * (Om * h0 ** 2 / 0.15) ** (-0.5)
    dTb = factor * np.sqrt(1 + z) * (1 - Tcmb0 * (1 + z) / T_gas) * (x_col + xal_coda_style) / (
                1 + x_col + xal_coda_style) * (1 - xHII)
    return {'z': z, 'x_HII': xHII, 'sfrd': sfrd, 'Gamma_heat': G_heat, 'Jalpha': Jalpha, 'x_al': xal_coda_style,
            'x_coll': x_coll, 'Gheat_GS_style': Gheat_GS_style, 'heat_per_baryon': heat_per_baryon, 'Tk': T_gas,
            'T_gas_neutral': T_gas_neutral, 'dTb': dTb}


def Tgas_from_profiles(param, halo_catalog, simple_model):
    LBox = param.sim.Lbox  # Mpc/h
    M_Bin = np.logspace(np.log10(param.sim.Mh_bin_min), np.log10(param.sim.Mh_bin_max), param.sim.binn, base=10)
    z_start = param.solver.z_max
    model_name = param.sim.model_name
    H_Masses = halo_catalog['M']
    z = halo_catalog['z']
    # quick load to find matching redshift between solver output and simulation snapshot.
    grid_model = pickle.load(
        file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[0]), 'rb'))
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]
    Indexing = np.argmin(
        np.abs(np.log10(H_Masses[:, None] / (M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))))),
        axis=1)  ## values of Mh at z_start, binned via M_Bin.

    Tgas, Tgas_neutral = 0, 0
    for i in range(len(M_Bin)):
        nbr_halos = np.where(Indexing == i)[0].size
        grid_model = pickle.load(
            file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[i]),
                      'rb'))
        M_h = grid_model.Mh_history[ind_z]

        if nbr_halos > 0 and M_h > param.source.M_min:

            r_grid_, Temp_profile, Temp_neutral = grid_model.r_grid_cell, grid_model.T_history[str(round(zgrid, 2))], \
                                                  grid_model.T_neutral_hist[str(round(zgrid, 2))]

            if simple_model:
                r_grid_ = r_grid_ / (1 + z)

            if param.cosmo.Temp_IC == 1:  ## adiab IC
                T_adiab_z_solver = Temp_profile[-1]
                Temp_profile = (Temp_profile - T_adiab_z_solver).clip(min=0)
                Temp_neutral = (Temp_neutral - T_adiab_z_solver).clip(min=0)
            T_vol = np.trapz(4 * np.pi * r_grid_ ** 2 * Temp_profile, r_grid_)
            T_vol_neutral = np.trapz(4 * np.pi * r_grid_ ** 2 * Temp_neutral, r_grid_)
            Tgas += T_vol * nbr_halos
            Tgas_neutral += T_vol_neutral * nbr_halos

    Tgas = Tgas / (LBox / (1 + z)) ** 3  # normalize by total physical volume
    Tgas_neutral = Tgas_neutral / (LBox / (1 + z)) ** 3
    return Tgas, Tgas_neutral


def xHII_approx_old(param, halo_catalog, simple_model=False):
    """
    Approximation of the mean ionization fraction (maybe more correct to say volume filling factor.)
    We compute for each halo the volume of the surounding ionized bubble. Sum all these volumes and normalize to the total simulation volume.
    """
    LBox = param.sim.Lbox  # Mpc/h
    M_Bin = np.logspace(np.log10(param.sim.Mh_bin_min), np.log10(param.sim.Mh_bin_max), param.sim.binn, base=10)
    z_start = param.solver.z_max
    model_name = param.sim.model_name

    H_Masses = halo_catalog['M']
    z = halo_catalog['z']

    # quick load to find matching redshift between solver output and simulation snapshot.
    grid_model = pickle.load(
        file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[0]), 'rb'))
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]
    Indexing = np.argmin(
        np.abs(np.log10(H_Masses[:, None] / (M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))))),
        axis=1)  ## values of Mh at z_start, binned via M_Bin.

    Ionized_vol = 0
    for i in range(len(M_Bin)):
        nbr_halos = np.where(Indexing == i)[0].size
        grid_model = pickle.load(
            file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[i]),
                      'rb'))
        Mh_ = grid_model.Mh_history[ind_z]
        if nbr_halos > 0 and Mh_ > param.source.M_min:
            r_grid_, xHII_profile = ion_profile(grid_model, zgrid, simple_model)
            bubble_volume = np.trapz(4 * np.pi * r_grid_ ** 2 * xHII_profile, r_grid_)
            Ionized_vol += bubble_volume * nbr_halos  ##physical volume !!

        print(nbr_halos, 'halos in mass bin ', i)
    x_HII = Ionized_vol / (LBox / (1 + z)) ** 3  # normalize by total physical volume
    return zgrid, x_HII


def ion_profile(grid_model, zgrid, simple_model):
    """
    Reads in grid_model (output of solver) and output the ionisation profile, with or without the "simple model" option.
    Returns r_grid in physical Mpc/h and xHII profile
    """
    if not simple_model:
        return grid_model.r_grid_cell, grid_model.xHII_history[str(round(zgrid, 2))]
    else:
        radial_grid = grid_model.r_grid_cell / (1 + zgrid)  # from co to phyz
        ind_z = np.argmin(np.abs(grid_model.z_history - zgrid))
        ion_front = grid_model.R_bubble[ind_z]  # cMpc/h
        x_HII_profile = np.zeros((len(radial_grid)))
        x_HII_profile[np.where(radial_grid < ion_front / (1 + zgrid))] = 1  ## sharp ionisation front.
        return radial_grid, x_HII_profile


def sfrd_approx(param, halo_catalog):
    """
    Approximation of the sfrd of a given snapshot. We sum over all halos from  halo_catalog, acoording to the source model in param. We then normlize to sim volume.
    Output is in  [(Msol/h) / yr /(cMpc/h)**3]
    """
    LBox = param.sim.Lbox  # Mpc/h
    M_Bin = np.logspace(np.log10(param.sim.Mh_bin_min), np.log10(param.sim.Mh_bin_max), param.sim.binn, base=10)
    z_start = param.solver.z_max
    model_name = param.sim.model_name

    H_Masses = halo_catalog['M']
    z = halo_catalog['z']
    print('There are', H_Masses.size, 'halos at z=', z, )

    # quick load to find matching redshift between solver output and simulation snapshot.
    grid_model = pickle.load(
        file=open('./profiles/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[0]), 'rb'))
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]
    Indexing = np.argmin(
        np.abs(np.log10(H_Masses[:, None] / (M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))))),
        axis=1)  ## values of Mh at z_start, binned via M_Bin.

    SFRD = 0
    for i in range(len(M_Bin)):
        nbr_halos = np.where(Indexing == i)[0].size
        if nbr_halos > 0:
            # grid_model = pickle.load(file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[i]),'rb'))
            M_halo = M_Bin[i] * np.exp(-param.source.alpha_MAR * (z - z_start))  # grid_model.Mh_history[ind_z]
            dMh_dt = param.source.alpha_MAR * M_halo * (z + 1) * Hubble(z, param)  ## [(Msol/h) / yr]
            SFRD += nbr_halos * dMh_dt * f_star_Halo(param, M_halo) * param.cosmo.Ob / param.cosmo.Om

    SFRD = SFRD / LBox ** 3  ## [(Msol/h) / yr /(cMpc/h)**3]

    return z, float(SFRD)


def mean_Jalpha_approx(param, halo_catalog, simple_model):
    """
    Approximation of the Jalpha in x_alpha calculation. To compare to Halo Model and to our calculation when puting profiles on grid.
    """
    LBox = param.sim.Lbox  # Mpc/h
    M_Bin = np.logspace(np.log10(param.sim.Mh_bin_min), np.log10(param.sim.Mh_bin_max), param.sim.binn, base=10)
    z_start = param.solver.z_max
    model_name = param.sim.model_name

    H_Masses = halo_catalog['M']
    z = halo_catalog['z']

    # quick load to find matching redshift between solver output and simulation snapshot.
    grid_model = pickle.load(
        file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[0]), 'rb'))
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]
    Indexing = np.argmin(
        np.abs(np.log10(H_Masses[:, None] / (M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))))),
        axis=1)  ## values of Mh at z_start, binned via M_Bin.

    r_lyal = np.logspace(-5, 2, 1000, base=10)  ## physical distance for lyal profile

    Jal_mean = 0
    X_al_mean = 0
    for i in range(len(M_Bin)):
        nbr_halos = np.where(Indexing == i)[0].size
        grid_model = pickle.load(
            file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[i]),
                      'rb'))
        Mh_ = grid_model.Mh_history[ind_z]
        if nbr_halos > 0 and Mh_ > param.source.M_min:
            r_grid, x_HII_profile = ion_profile(grid_model, zgrid, simple_model)
            # grid_model.rho_al_history[str(round(zgrid, 2))]
            # x_alpha = grid_model.x_al_history[str(round(zgrid, 2))]
            rho_alpha_ = rho_alpha(r_lyal, grid_model.Mh_history[ind_z], zgrid, param)[0]
            T_extrap = np.interp(r_lyal, r_grid, grid_model.T_history[str(round(zgrid, 2))])

            xHII_extrap = np.interp(r_lyal, r_grid, x_HII_profile)
            x_alpha = 1.81e11 * (rho_alpha_) * S_alpha(zgrid, T_extrap, 1 - xHII_extrap) / (1 + zgrid)
            mean_rho = np.trapz(rho_alpha_ * 4 * np.pi * r_lyal ** 2, r_lyal)
            Jal_mean += nbr_halos * mean_rho
            X_al_mean += nbr_halos * np.trapz(x_alpha * 4 * np.pi * r_lyal ** 2, r_lyal)

    Jal_mean = Jal_mean / (LBox / (1 + z)) ** 3  ## [pcm**-2.Hz-1.s-1]
    X_al_mean = X_al_mean / (LBox / (1 + z)) ** 3
    return Jal_mean, X_al_mean


def G_heat_approx(param, halo_catalog):
    """
    Compute the energy deposited as heat per baryon in [eV.s-1]
    Similar to xHII : We take the heat profile, integrate over volume, sum over halos and normlize to simulation volume. Gives an average Gamma_heat.
    """
    LBox = param.sim.Lbox  # Mpc/h
    M_Bin = np.logspace(np.log10(param.sim.Mh_bin_min), np.log10(param.sim.Mh_bin_max), param.sim.binn, base=10)
    z_start = param.solver.z_max
    model_name = param.sim.model_name

    H_Masses = halo_catalog['M']
    z = halo_catalog['z']

    # quick load to find matching redshift between solver output and simulation snapshot.
    grid_model = pickle.load(
        file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[0]), 'rb'))
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]
    Indexing = np.argmin(
        np.abs(np.log10(H_Masses[:, None] / (M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))))),
        axis=1)  ## values of Mh at z_start, binned via M_Bin.

    heat_per_baryon = 0
    for i in range(len(M_Bin)):
        nbr_halos = np.where(Indexing == i)[0].size
        grid_model = pickle.load(
            file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[i]),
                      'rb'))
        M_h = grid_model.Mh_history[ind_z]
        if nbr_halos > 0 and M_h > param.source.M_min:
            heat_profile = grid_model.heat_history[str(round(zgrid, 2))]
            r_grid_ = grid_model.r_grid_cell
            heat_per_baryon += np.trapz(4 * np.pi * r_grid_ ** 2 * heat_profile, r_grid_) * nbr_halos

    heat_per_baryon = heat_per_baryon / (LBox / (1 + z)) ** 3

    return heat_per_baryon  # [eV.s**-1]


def J_alpha_n(zz, sfrd, param):
    """
    Same as in Halo model. take as an inputa the sfrd
    """

    Om = param.cosmo.Om
    Ob = param.cosmo.Ob
    h0 = param.cosmo.h

    # comoving proton number density
    nb0 = rhoc0 * Ob / (m_p_in_Msun * h0)  # [h/Mpc]^3

    # rec fraction
    names = 'n, f'
    path_to_file = Path(importlib.util.find_spec('beorn').origin).parent / 'files' / 'recfrac.dat'
    rec = np.genfromtxt(path_to_file, usecols=(0, 1), comments='#', dtype=float, names=names)
    rectrunc = 23

    # line frequencies
    nu_n = nu_LL * (1 - 1 / rec['n'][2:] ** 2)
    nu_n = np.insert(nu_n, [0, 0], np.inf)

    # sfrd_tck = splrep(zz, sfrd)

    # binning of z_prime
    dz_prime = 0.01  # param.code.dz_prime_lyal

    # flux intensity (Eq.15 in arXiv:astro-ph/0604040)
    J_al = []
    for i in range(len(zz)):
        zmax_n = np.full(len(rec['n']), zz[i])
        zmax_n[2:] = (1 - (rec['n'][2:] + np.ones(len(rec['n']) - 2)) ** (-2)) / (1 - (rec['n'][2:]) ** (-2)) * (
                    1 + zz[i]) - 1

        J_al_n = []
        for k in range(0, rectrunc):
            zrange = zmax_n[k] - zz[i]
            N_prime = int(zrange / dz_prime)
            if (N_prime < 2):
                N_prime = 2
            z_prime = np.logspace(np.log(zz[i]), np.log(zmax_n[k]), N_prime, base=np.e)

            eps_b = eps_lyal(nu_n[k] * (1 + z_prime) / (1 + zz[i]), param)
            J_al_n_prime = c_km_s * h0 / hubble(z_prime, param) * eps_b * np.interp(z_prime, zz,
                                                                                    sfrd)  # splev(z_prime, sfrd_tck)  # [1/Hz/yr/(Mpc/h)^2]

            J_al_nk = (1 + zz[i]) ** 2 / (4 * np.pi) * rec['f'][k] * np.trapz(J_al_n_prime,
                                                                              z_prime)  # [1/Hz/yr/(Mpc/h)^2]
            J_al_n += [J_al_nk * (h0 / cm_per_Mpc) ** 2 / sec_per_year]  # [1/cm^2/Hz/s]
        J_al += [J_al_n]
    J_al = np.array(J_al)
    J_al = np.ndarray.transpose(J_al)

    return J_al




def mean_J_xray_nu_approx(param, halo_catalog, simple_model, density_normalization=1, redshifting='yes'):
    """
    X-ray flux per frequency nu. Same method as above (Jalpha_approx)
    """
    LBox = param.sim.Lbox  # Mpc/h
    M_Bin = np.logspace(np.log10(param.sim.Mh_bin_min), np.log10(param.sim.Mh_bin_max), param.sim.binn, base=10)
    z_start = param.solver.z_max
    model_name = param.sim.model_name

    H_Masses = halo_catalog['M']
    z = halo_catalog['z']

    # quick load to find matching redshift between solver output and simulation snapshot.
    grid_model = pickle.load(
        file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[0]), 'rb'))
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]
    Indexing = np.argmin(
        np.abs(np.log10(H_Masses[:, None] / (M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))))),
        axis=1)  ## values of Mh at z_start, binned via M_Bin.
    Ob, Om, h0, alpha = param.cosmo.Ob, param.cosmo.Om, param.cosmo.h, param.source.alpha_MAR

    mean_Gamma_heat = 0
    Jxray_mean = 0
    E_range = 0
    for i in range(len(M_Bin)):
        nbr_halos = np.where(Indexing == i)[0].size
        grid_model = pickle.load(
            file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[i]),
                      'rb'))
        M_h = grid_model.Mh_history[ind_z]
        if nbr_halos > 0 and M_h > param.source.M_min:
            r_grid, xHII = ion_profile(grid_model, zgrid, simple_model)
            Mh = grid_model.Mh_history[ind_z]
            nB = (1 + z) ** 3 * bar_density_2h(r_grid, param, z, Mh)
            n_HI = nB * (1 - xHII)

            if param.source.type == 'SED':
                dMh_dt = alpha * Mh * (z + 1) * Hubble(z, param)  ## [(Msol/h) / yr], SFR at zprime, at emission
                Edot = dMh_dt * f_star_Halo(param, Mh) * Ob / Om * param.source.cX * eV_per_erg / h0  # eV/s at emission
            else:
                print('Jxray approx not implemented for source.type other than SED.')

            if redshifting == 'no':
                E_range, Jxray_flux = J_xray_no_redshifting(r_grid, n_HI * density_normalization, Edot,
                                                            param)  # shape of Jxray is (r_grid,E_range)
                mean_J = np.trapz(Jxray_flux * 4 * np.pi * r_grid[:, None] ** 2, r_grid,
                                  axis=0)  # shape is E_range. "spatial" mean
                mean_Gam = 0
            else:
                E_range, Jxray_flux = J_xray_with_redshifting(r_grid, n_HI * density_normalization, Mh, z,
                                                              param)  # [eV, gam/Hz/s/pcm^2]
                factor_ = sigma_HI(E_range) * (E_range - E_HI)
                Gamma_heat_ = np.trapz(Jxray_flux * factor_[:, None], E_range * Hz_per_eV,
                                       axis=0)  # shape is (r_grid), [eV/s]
                mean_J = np.trapz(Jxray_flux * 4 * np.pi * r_grid ** 2, r_grid,
                                  axis=1)  # shape is E_range. "spatial" mean
                mean_Gam = np.trapz(Gamma_heat_ * 4 * np.pi * r_grid ** 2, r_grid)
            Jxray_mean += nbr_halos * mean_J
            mean_Gamma_heat += nbr_halos * mean_Gam

    Jxray_mean = Jxray_mean / (LBox / (1 + z)) ** 3  ## [pcm**-2.Hz-1.s-1]
    mean_Gamma_heat = mean_Gamma_heat / (LBox / (1 + z)) ** 3  ## [pcm**-2.Hz-1.s-1]

    return E_range, Jxray_mean, mean_Gamma_heat
