"""
Bunch of python functions useful to plot, load, read in...
"""
import numpy as np
import scipy
import sys
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from .constants import *


def plot_Beorn(physics, qty='dTb', xlim=None, ylim=None, label='', color='C0', ls='-', lw=1, alpha=1):
    """""""""
    This functions plots Beorn global quantities.

    Parameters
    ----------    
    physics : GS pickle file given by Beorn.
    """""""""

    try:
        plt.plot(physics['z'], physics[qty], label=label, color=color, ls=ls, lw=lw, alpha=alpha)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
    except Exception:
        if qty == 'x_HI' or qty == 'xHI':
            plt.plot(physics['z'], 1 - physics['x_HII'], label=label, color=color, ls=ls, lw=lw, alpha=alpha)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.legend()
        print('available qty in Beorn: ', physics.keys())


def plot_PS_Beorn(z, PS, color, ax, Beta=1, label='', qty='xHII', with_dTb=False, GS=None, ls='-', alpha=0.5, lw=3):
    """""""""
    Plot a Beorn power spectrum as a function of k.
    Beta : float, the beta factor in the perturbative expansion of dTb. Set to 1 by default. 
    """""""""
    import matplotlib
    matplotlib.rc('xtick', labelsize=15)
    matplotlib.rc('ytick', labelsize=15)

    kk, zz = PS['k'], PS['z']
    ind_z = np.argmin(np.abs(zz - z))
    print('--BEORN -- plotting power spectrum of ', qty, 'at redshift ', round(PS['z'][ind_z], 3))
    coef = 1
    if with_dTb:
        coef = GS['dTb'][ind_z] ** 2
        print('at z = ', zz[ind_z], 'dTb Boern IS : ', GS['dTb'][ind_z])

    try:
        ax.loglog(kk * 0.68, coef * kk ** 3 * Beta * np.abs(PS['PS_' + qty][ind_z]) / 2 / np.pi ** 2, lw=lw,
                  alpha=alpha, label=label,
                  color=color, ls=ls)
    except Exception:
        print('RT:', PS.keys())
    plt.legend()
    plt.xlabel('k [1/Mpc]', fontsize=14)
    plt.ylabel(r'$\Delta^{2} = k^{3}P(k)/(2\pi^{2})$', fontsize=14)


def plot_Beorn_PS_of_z(k, GS_Beorn, PS_Beorn, ls='-', lw=1, color='b', RSD=False, label='', qty='dTb', alpha=1, ax=plt):
    """""""""
    Plot a Beorn Power Spectrum as a function of z. 
    """""""""
    ind_k = np.argmin(np.abs(PS_Beorn['k'] - k))
    print('k RT is', PS_Beorn['k'][ind_k], 'Mpc/h')
    kk, PS_dTb_RT = PS_Beorn['k'][ind_k], PS_Beorn['PS_' + qty][:, ind_k]
    dTb_RT = GS_Beorn['dTb']
    ax.semilogy(GS_Beorn['z'], kk ** 3 * dTb_RT ** 2 * PS_dTb_RT / 2 / np.pi ** 2, ls=ls, lw=lw, alpha=alpha,
                label=label, color=color)
    if RSD:
        dTb_RSD, PS_dTb_RT_RSD = GS_Beorn['dTb_RSD'], PS_Beorn['PS_dTb_RSD'][:, ind_k]
        ax.semilogy(GS_Beorn['z'], kk ** 3 * dTb_RSD ** 2 * PS_dTb_RT_RSD / 2 / np.pi ** 2, lw=1, alpha=alpha,
                    label=label, color=color)
        print(PS_dTb_RT_RSD / PS_dTb_RT)


def plot_2d_map(grid, Ncell, Lbox, slice_nbr, qty='label', scale='lin'):
    # Ncell : int, nbr of grid pixels
    # slice_nbr : int, slice to plot
    # Lbox : int, Box size in Mpc/h
    if scale == 'lin':
        norm = None
    elif scale == 'log':
        norm = matplotlib.colors.LogNorm()
    else:
        print('scale should be lin or log.')

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    pos = ax.pcolormesh(np.linspace(0, Lbox, Ncell + 1), np.linspace(0, Lbox, Ncell + 1),
                        grid[:, slice_nbr, :], norm=norm)  # , interpolation='none')#,vmin = -5,vmax = 2.)
    cb = fig.colorbar(pos, ax=ax, label=qty)
    ax.set_xlabel('cMpc/h', fontsize=14)
    ax.set_ylabel('cMpc/h', fontsize=14)
    cb.set_label(label=qty, fontsize=15)
    cb.ax.tick_params(labelsize=14)
    ax.set_rasterized(True)
    ax.tick_params(labelsize=14)


def Tgas_from_rho_heat(HM_PS):
    rmin = 0.1
    rmax = 6000
    Nr = 50
    rr = np.logspace(np.log10(rmin), plot_PS_Beornnp.log10(rmax), Nr)
    rho_heat_HM = HM_PS['rho_heat_prof']  #
    dn_dlnm = HM_PS['dndlnm']
    masses = HM_PS['M0']

    rho_int = np.trapz(rho_heat_HM * rr[:, None] ** 2 * 4 * np.pi, rr, axis=1)
    T_mean = np.trapz(dn_dlnm * rho_int, np.log(masses), axis=1)
    return T_mean


def Gheat_from_rho_x(rr, HM_PS):
    rho_xray_HM = HM_PS['rho_X_prof']  # [erg/s]
    dn_dlnm = HM_PS['dndlnm']
    masses = HM_PS['M0']

    rho_int = np.trapz(rho_xray_HM * rr[:, None] ** 2 * 4 * np.pi, rr, axis=1)
    Gh_mean = np.trapz(dn_dlnm * rho_int, np.log(masses), axis=1)
    return Gh_mean


#### Mhalo(z) from simulation from HM paper
z_array_1 = np.array(
    [10.08083140877598, 9.705542725173212, 9.301385681293302, 8.897228637413393, 8.348729792147806, 7.684757505773671,
     7.107390300230946, 6.5011547344110845, 5.92378752886836, 5.08660508083141])
Mh_z_1 = np.array([94706100.0077222, 126579158.66671978, 163154262.77379718, 222053024.50155312, 324945871.5591836,
                   530163041.15627587, 790019471.240894, 1243048829.5695167, 1991648063.8308542, 3689173954.4382358])

z_array_2 = np.array([12.361431870669746, 11.870669745958429, 11.379907621247115, 10.744803695150113, 9.821016166281755,
                      8.897228637413393, 8.146651270207853, 7.453810623556583, 6.732101616628176, 5.981524249422634,
                      5.057736720554274])
Mh_z_2 = np.array([98203277.90631257, 136100044.66105506, 195586368.30395073, 302214265.51783735, 570040236.7896342,
                   1114920997.0080914, 1852322215.6741183, 3191074972.923546, 5597981979.123267, 10000000000,
                   21414535638.539528])

z_array_3 = np.array([14.988452655889146, 14.642032332563511, 13.833718244803697, 12.70785219399538, 11.986143187066977,
                      11.23556581986143, 10.600461893764434, 9.79214780600462, 8.926096997690532, 8.146651270207853,
                      7.511547344110854, 6.876443418013858, 6.241339491916859, 5.6928406466512715, 5.288683602771364,
                      5.057736720554274])
Mh_z_3 = np.array([172274291.4769941, 218063348.75063255, 368917395.4438236, 775825016.8566794, 1243048829.5695167,
                   2102977594.5461233, 3493872774.7491226, 6129168695.9257145, 11987818459.583773, 21029775945.46132,
                   35577964903.39488, 61291686959.25715, 109488896512.76825, 178635801924.5735, 266193134612.6126,
                   343109759067.9875])


def plot_1D_profiles(param, profile, ind_M, z_liste):
    import warnings
    import matplotlib.pyplot as plt
    from beorn.cosmo import T_adiab

    warnings.filterwarnings('ignore')
    plt.subplots(1, 4, figsize=(17, 5))

    co_radial_grid = profile.r_grid_cell
    r_lyal_phys = profile.r_lyal
    zz = profile.z_history
    Mh_liste = []
    for ii, zi in enumerate(z_liste):
        ind_z = np.argmin(np.abs(zz - zi))

        zzi = profile.z_history[ind_z]
        Mh_i = profile.Mh_history[ind_z, ind_M]
        print('z, Mh = ', zzi, ', {:.2e}'.format(Mh_i / 0.68))
        Mh_liste.append(Mh_i / 0.68)
        T_adiab_z = T_adiab(zzi, param)

        x_HII_profile = np.zeros((len(co_radial_grid)))
        x_HII_profile[np.where(co_radial_grid < profile.R_bubble[ind_z, ind_M])] = 1
        Temp_profile = profile.rho_heat[ind_z, :, ind_M] + T_adiab_z
        lyal_profile = profile.rho_alpha[ind_z, :, ind_M]  # *1.81e11/(1+zzi)

        plt.subplot(141)
        plt.semilogy([zzi], [Mh_i / 0.68], '*', color='C' + str(ii), markersize=13.5)

        plt.subplot(143)
        plt.loglog(co_radial_grid / 0.68, Temp_profile, lw=1.7)

        plt.subplot(142)
        plt.loglog(r_lyal_phys * (1 + zzi) / 0.68, lyal_profile, color='C' + str(ii), lw=1.7)

        plt.subplot(144)
        plt.semilogx(co_radial_grid / 0.68, x_HII_profile, color='C' + str(ii), lw=1.7)

    plt.subplot(141)

    plt.semilogy(z_array_1, Mh_z_1 / 0.68, color='gold', ls='--', lw=3, alpha=0.8)
    plt.semilogy(z_array_2, Mh_z_2 / 0.68, color='gold', ls='--', lw=3, alpha=0.8)
    plt.semilogy(z_array_3, Mh_z_3 / 0.68, color='gold', ls='--', lw=3, alpha=0.8, label='Simu (Behroozi +20)')
    plt.semilogy(zz, profile.Mh_history[:, ind_M] / 0.68, color='gray', alpha=1, lw=2, label='analytical MAR')
    plt.xlim(15, 5)
    plt.ylim(1.5e8, 8e12)
    plt.xlabel('z', fontsize=15)
    plt.ylabel('$M_h$ [M$_\odot$]', fontsize=17)
    plt.tick_params(axis="both", labelsize=13.5)
    plt.legend(fontsize=15, loc='upper left')

    plt.subplot(142)
    plt.xlim(2e-1, 1e3)
    plt.ylim(2e-17, 1e-5)
    plt.loglog([], [], color='C0', label='$z\sim$' + '{},'.format(z_liste[0]) + '$M_{\mathrm{h}}=$' + '{:.2e}'.format(
        Mh_liste[0]) + '$M_{\odot}$')
    plt.xlabel('r [cMpc]', fontsize=15)
    plt.tick_params(axis="both", labelsize=13.5)
    plt.ylabel('ρ$_{α}$ [$\mathrm{pcm}^{-2}\, \mathrm{s}^{-1} \, \mathrm{Hz}^{-1}$]', fontsize=17)
    plt.legend(fontsize=13)

    plt.subplot(143)
    plt.xlim(2e-2, 1e2)
    plt.ylim(0.8, 5e6)
    plt.loglog([], [], color='C1', label='$z\sim$' + '{},'.format(z_liste[1]) + '$M_{\mathrm{h}}=$' + '{:.2e}'.format(
        Mh_liste[1]) + '$M_{\odot}$')
    plt.xlabel('r [cMpc]', fontsize=15)
    plt.ylabel(' ρ$_{h}$ [K]', fontsize=17)
    plt.tick_params(axis="both", labelsize=13.5)
    plt.legend(fontsize=13)

    plt.subplot(144)
    plt.xlim(2e-2, 1e2)
    plt.ylim(0, 1.2)
    plt.semilogx([], [], color='C2', label='$z\sim$' + '{},'.format(z_liste[2]) + '$M_{\mathrm{h}}=$' + '{:.2e}'.format(
        Mh_liste[2]) + '$M_{\odot}$')
    plt.xlabel('r [cMpc]', fontsize=15)
    plt.tick_params(axis="both", labelsize=13.5)
    plt.ylabel(' x$_{\mathrm{HII}}$', fontsize=17)
    plt.legend(fontsize=13)

    plt.tight_layout()


########### FUNCTIONS TO DO COMPARISON PLOTS WITH ------>HALO MODEL<--------

def Plotting_GS(physics, x__approx, GS, PS, save_loc=None):
    """""""""""
    physics : GS history dictionnary data from radtrans
    x__approx : approximation dictionnary (when we compute the global quantities from halo catalogs + profiles, without using a grid.)
    GS, PS : from halo model
    """""""""""
    ################################################ loading
    from matplotlib import pyplot as plt
    RT_dTb_GS = physics['dTb_GS']
    RT_dTb_GS_Tkneutr = physics['dTb_GS_Tkneutral']
    RT_dTb = physics['dTb']  # physics['dTb']
    RT_zz = physics['z']
    x_tot = physics['xal_coda_style'] + physics['x_coll']
    Tcmb = (1 + RT_zz) * Tcmb0

    HM_zz, HM_dTb = GS['z'], GS['dTb']

    fig = plt.figure(figsize=(10, 20))

    ################################################ dTb
    axis1 = fig.add_subplot(421)
    axis1.plot(HM_zz, HM_dTb, label='HM')

    axis1.plot(RT_zz, RT_dTb_GS, label='RT dTb=f(mean)', ls='--')
    axis1.plot(RT_zz, RT_dTb, label='RT dTb=mean(dTb))', ls='--')
    # axis1.plot(RT_zz , RT_dTb_GS_Tkneutr   ,label='RT dTb=f(mean) with Tkneutral)',ls='--')

    axis1.set_xlim(5, 21)
    axis1.set_ylabel('dTb')
    axis1.legend(loc='best')

    ################################################ Temperatures
    axis2 = fig.add_subplot(422)
    axis2.semilogy(HM_zz, GS['Tgas'], 'k', label='Tgas')
    axis2.semilogy(HM_zz, GS['Tcmb'], 'orange', label='Tcmb')
    axis2.semilogy(HM_zz, GS['Tspin'], 'r', label='Tspin')
    axis2.semilogy(RT_zz, physics['Tadiab'], 'cyan', ls='--', label='Tadiab')
    axis2.semilogy(RT_zz, physics['T_spin'], 'r', ls='--')
    axis2.semilogy(RT_zz, physics['Tk'], 'k', ls=':')
    # axis2.semilogy(RT_zz,physics['Tk_neutral_regions'],'k',ls='-.')

    if PS is not None:
        Tgas_rhoheat = Tgas_from_rho_heat(PS)
        axis2.semilogy(HM_zz, Tgas_rhoheat, 'g', label='Tgas from rhoheat')

    axis2.set_xlim(5, 22)
    axis2.set_ylabel('T')
    axis2.set_xlabel('z')
    axis2.legend(loc='best')

    axis2.set_ylim(1, 1e3)
    if save_loc is not None:
        plt.savefig(save_loc)

    ################################################ Gamma_heat and sfrd
    axis3 = fig.add_subplot(423)
    # axis3.semilogy(HM_zz,GS['Gamma_heat']*eV_per_erg,'r',label='Gamma_heat [eV]')
    # axis3.semilogy(x__approx['z'],x__approx['Gamma_heat'],'r',ls='--')

    axis3.semilogy(HM_zz, GS['sfrd_lyal'], 'b', ls='-', label='sfrd [Msol h^2/yr/Mpc]')
    # axis3.semilogy(x__approx['z'], x__approx['sfrd'], 'b', ls='--')
    axis3.set_xlim(5, 24)
    axis3.set_ylim(1e-6, 1e2)
    axis3.set_ylabel('')
    axis3.set_xlabel('z')

    axis3.legend(loc='best')

    ################################################ xal

    axis4 = fig.add_subplot(424)
    axis4.semilogy(GS['z'], GS['x_al'], 'r', label='x_al')
    axis4.semilogy(physics['z'], physics['x_al'], 'r', ls='--')

    # axis4.semilogy(HM_zz,GS['sfrd_lyal'],'b',ls='-',label='sfrd [Msol h^2/yr/Mpc]')
    # axis4.semilogy(x__approx['z'],x__approx['sfrd'],'b',ls='--')
    axis4.set_xlim(5, 20)
    axis4.set_ylim(1e-3, 1e4)
    axis4.set_ylabel('')
    axis4.set_xlabel('z')

    axis4.legend(loc='best')

    ################################################ xHI

    axis5 = fig.add_subplot(425)
    axis5.plot(GS['z'], GS['x_HI'], 'r', label='xHI')
    # axis5.plot(x__approx['z'], 1 - x__approx['xHII'], 'r', ls='--', label='approx')
    axis5.plot(physics['z'], 1 - physics['x_HII'], ls='--', lw=5)
    axis5.set_xlim(5, 20)
    # axis5.set_ylim(1e-3,1e4)
    axis5.set_ylabel('')
    axis5.set_xlabel('z')

    if save_loc is not None:
        plt.savefig(save_loc)

    ################################################ XCOL

    # axis6 = fig.add_subplot(426)
    # axis6.semilogy(GS['z'],GS['x_cl'],'r',label='x_col')
    # axis6.semilogy(x__approx['z'],physics['x_coll'],'r',ls='--')
    # axis6.semilogy(GS['z'],GS['x_cl']+GS['x_al'],'r',label='x_col')
    # axis6.semilogy(x__approx['z'],physics['x_coll']+physics['xal_coda_style'],'r',ls='--')
    # axis6.set_xlim(5,20)
    ##axi65.set_ylim(1e-3,1e4)
    # axis6.set_ylabel('')
    # axis6.set_xlabel('z')

    # axis7 = fig.add_subplot(427)
    # axis7.semilogy(GS['z'],GS['Gamma_heat']*eV_per_erg,'k',label='Gamma_heat')
    # axis7.semilogy(x__approx['z'],x__approx['Gamma_heat'],'k',ls='--')
    # axis7.set_xlim(5,20)
    # axis7.set_ylim(1e-25,1e-15)
    # axis7.set_ylabel('')
    # axis7.set_xlabel('z')
    #
    # if save_loc is not None :
    #    plt.savefig(save_loc)


def Plotting_PS(k_array, z, physics, PS_Dict, GS, PS, save_loc=None):
    """""""""""
    k_array,z : values for the power spectra plot
    physics,PS_Dict : GS history and PS dictionnary data from radtrans
    GS, PS : from halo model
    """""""""""
    ################################################ loading
    from matplotlib import pyplot as plt
    RT_zz = physics['z']  #
    Tcmb = (1 + RT_zz) * Tcmb0
    RT_dTb_GS = physics['dTb_GS']  # *(1-Tcmb/physics['Tk'])/(1-Tcmb/physics['Tk_neutral'])
    x_tot = physics['x_al'] + physics['x_coll']
    RT_dTb = physics[
        'dTb']  # * x_tot/(1 + x_tot) * (physics['x_al']+physics['x_coll']+1) / (physics['x_al']+physics['x_coll'])

    beta_a = physics['beta_a']  # physics['xal_coda_style'] / x_tot / (1 + x_tot)  # physics['beta_a'] #
    beta_a_HM = GS['x_al'] / (GS['x_al'] + GS['x_coll']) / (1 + GS['x_al'] + GS['x_coll'])

    Tbg = Tcmb0 * (1 + 135) * (1 + RT_zz) ** 2 / (1 + 135) ** 2
    fr = 1  # (physics['Tk']-Tbg)/physics['Tk'] ### Halomodel primordial/heating decomp
    beta_T = Tcmb / (physics['Tk'] - Tcmb)  # physics['beta_T'] * fr  #

    beta_T_HM = (1 + GS['z']) * Tcmb0 / (GS['Tgas'] - (1 + GS['z']) * Tcmb0)
    beta_r = physics['beta_r']  # -physics['x_HII']/(1-physics['x_HII'])
    beta_r_HM = -(1 - GS['x_HI']) / (GS['x_HI'])
    RT_iz = np.argmin(np.abs(RT_zz - z))

    HM_zz, HM_dTb = GS['z'], GS['dTb']
    HM_iz = np.argmin(np.abs(PS['z'] - RT_zz[RT_iz]))

    fig = plt.figure(figsize=(10, 20))

    ################################################ dTb
    axis1 = fig.add_subplot(421)
    axis1.plot(HM_zz, HM_dTb, label='HM')
    axis1.plot(RT_zz, RT_dTb_GS, label='rt dTb_GS f(mean)', ls='--')
    axis1.plot(RT_zz, RT_dTb, label='rt mean(dTb) ', ls='--')
    # axis1.plot(RT_zz , RT_dTb,label='radtrans',ls='-.')
    axis1.set_xlim(6, 21)
    axis1.set_ylabel('dTb')
    axis1.legend()

    ################################################ xHII
    axis2 = fig.add_subplot(422)
    axis2.plot(HM_zz, 1 - GS['x_HI'])
    axis2.plot(RT_zz, physics['x_HII'], ls='--')
    axis2.set_xlim(6, 20)
    axis2.set_ylabel('xHII')
    axis2.set_xlabel('z')

    ################################################ PS(k)
    axis3 = fig.add_subplot(423)
    print('z is', z, '. Radtrans z is :', RT_zz[RT_iz])

    PS_RT = PS_Dict
    kk = PS_RT['k']
    PS_xHI = PS_RT['PS_xHII'] * beta_r[:, None] ** 2  ### sizes are size(kk)
    PS_T = PS_RT['PS_T'] * beta_T[:, None] ** 2
    PS_dTb = PS_RT['PS_dTb']
    PS_rho = PS_RT['PS_rho']
    PS_xal = PS_RT['PS_xal'] * beta_a[:, None] ** 2

    PS_bb = PS_rho  # gas (matter)
    PS_aa = PS_xal  #
    PS_TT = PS_T  #
    PS_rr = PS_xHI  # reio

    PS_ab = beta_a[:, None] * PS_RT['PS_rho_xal']
    PS_Tb = beta_T[:, None] * PS_RT['PS_rho_T']
    PS_aT = beta_a[:, None] * beta_T[:, None] * PS_RT['PS_T_lyal']
    PS_rb = beta_r[:, None] * PS_RT['PS_rho_xHII']
    PS_ra = beta_r[:, None] * beta_a[:, None] * PS_RT['PS_lyal_xHII']
    PS_rT = beta_r[:, None] * beta_T[:, None] * PS_RT['PS_T_xHII']

    PS_dTb_HMstyle = PS_bb + PS_aa + PS_TT + PS_rr + 2 * (PS_ab + PS_Tb + PS_aT + PS_rb + PS_ra + PS_rT)

    dTb_RT_z = RT_dTb_GS[RT_iz]
    dTb_HM_z = GS['dTb'][HM_iz]
    #### dTb
    axis3.loglog(PS['k'], PS['k'] ** 3 * dTb_HM_z ** 0 * PS['P_mu0'][HM_iz] / 2 / np.pi ** 2, color='r', label='PS_dTb')
    axis3.loglog(kk, kk ** 3 * abs(dTb_RT_z) ** 0 * PS_dTb_HMstyle[RT_iz] / 2 / np.pi ** 2,
                 color='r')  # ,alpha=0.5,ls='--',lw=4,label='PS perturb')
    axis3.loglog(kk, kk ** 3 * abs(dTb_RT_z) ** 0 * PS_dTb[RT_iz] / 2 / np.pi ** 2, color='r', alpha=0.5, ls=':',
                 label='PS(dTb)', lw=4)
    print('mean dTb is ', dTb_RT_z)
    # test_HM = PS['k']**3 * dTb_HM_z**2 * PS['P_mu0'][HM_iz]/2/np.pi**2
    # test_RT = abs(dTb_RT_z)**2 * PS_dTb/2/np.pi**2

    #### mm
    axis3.loglog(PS['k'], PS['k'] ** 3 * PS['P_mm'][HM_iz] / 2 / np.pi ** 2, color='k', label='PS_mm')
    axis3.loglog(kk, kk ** 3 * PS_rho[RT_iz] / 2 / np.pi ** 2, color='k', alpha=0.5, ls='--', lw=4)
    #### TT
    axis3.loglog(PS['k'], PS['k'] ** 3 * dTb_HM_z ** 0 * PS['P_TT'][HM_iz] / 2 / np.pi ** 2, color='g', label='PS_TT')
    axis3.loglog(kk, kk ** 3 * abs(dTb_RT_z) ** 0 * PS_T[RT_iz] / 2 / np.pi ** 2, color='g', alpha=0.5, ls='--', lw=4)
    #### aa
    axis3.loglog(PS['k'], PS['k'] ** 3 * dTb_HM_z ** 0 * PS['P_aa'][HM_iz] / 2 / np.pi ** 2, color='b', label='PS_aa')
    axis3.loglog(kk, kk ** 3 * abs(dTb_RT_z) ** 0 * PS_xal[RT_iz] / 2 / np.pi ** 2, color='b', alpha=0.5, ls='--', lw=4)
    ##### bubbles
    axis3.loglog(PS['k'], PS['k'] ** 3 * dTb_HM_z ** 0 * PS['P_rr'][HM_iz] / 2 / np.pi ** 2, color='darkorange',
                 label='PS_bbub')
    axis3.loglog(kk, kk ** 3 * abs(dTb_RT_z) ** 0 * PS_xHI[RT_iz] / 2 / np.pi ** 2, color='darkorange', ls='--')
    print('beta_T is ', beta_T[RT_iz])

    axis3.set_xlim(3e-2, 1e1)
    axis3.set_ylim(3e-4, 2e0)
    axis3.legend(title='z={}'.format(z), loc='best')
    axis3.set_xlabel('k [h/Mpc]', fontsize=14)
    axis3.set_ylabel('$\Delta^{2} = k^{3}P(k)/(2\pi^{2})$   ', fontsize=14)

    ################################################ PS(z)
    axis4 = fig.add_subplot(424)
    ax = plt.gca()
    for k0 in k_array:
        color = next(ax._get_lines.prop_cycler)['color']
        ind_k_RT = np.argmin(np.abs(PS_RT['k'] - k0))
        axis4.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * RT_dTb_GS ** 2 * PS_dTb_HMstyle[:, ind_k_RT] / 2 / np.pi ** 2,
                       color=color, alpha=0.5, ls='--', lw=4)
        axis4.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * RT_dTb ** 2 * PS_dTb[:, ind_k_RT] / 2 / np.pi ** 2,
                       color=color, ls='-.')

        ind_k = np.argmin(np.abs(PS['k'] - PS_RT['k'][ind_k_RT]))
        axis4.semilogy(PS['z'], PS['k'][ind_k] ** 3 * HM_dTb ** 2 * PS['P_mu0'][:, ind_k] / 2 / np.pi ** 2, color=color,
                       label='k={}'.format(k0))
        # axis4.semilogy(PS['z'],PS['k'][ind_k]**3 * HM_dTb**2 * PS['P_angav'][:,ind_k]/2/np.pi**2,ls=':',lw=4,alpha=0.7,color=color,label='k={}'.format(k0))
        # print(PS['P_angav'][:, ind_k] / PS['P_mu0'][:, ind_k])
        print('k RT is ', round(PS_RT['k'][ind_k_RT], 4), 'k HM is,', PS['k'][ind_k])

    # axis4.semilogy(PS['z'],PS['k'][ind_k]**3 * HM_dTb**2* (PS['P_aa'][:,ind_k]+2*(PS['P_ba'][:,ind_k]+PS['P_Ta'][:,ind_k]+PS['P_ra'][:,ind_k]))/2/np.pi**2,color=color,label='k={}'.format(k0))
    # axis4.semilogy(RT_zz, k0**3 * RT_dTb_GS**2 * (PS_aa[:,ind_k_RT]+2*(PS_aT[:,ind_k_RT]+PS_ab[:,ind_k_RT]+PS_ra[:,ind_k_RT]))/2/np.pi**2,color=color,alpha=0.5,ls='--',lw=4)
    axis4.semilogy([], [], color='gray', label='PS Perturbtiv', alpha=0.5, ls='--', lw=4)
    axis4.semilogy([], [], color='gray', label='PS(dTb)', ls='-.')

    axis4.set_ylim(1e-1, 1e3)
    axis4.set_xlim(6, 20)

    axis4.legend(title='$ k^{3}P(k)/(2\pi^{2})$')  # dTb^{2}
    axis4.set_xlabel('z', fontsize=14)

    ################################################ PS_lymanalpha(z)
    axis5 = fig.add_subplot(425)
    ax = plt.gca()
    for k0 in [k_array[0]]:
        color = next(ax._get_lines.prop_cycler)['color']
        ind_k_RT = np.argmin(np.abs(PS_RT['k'] - k0))
        axis5.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * RT_dTb_GS ** 0 * PS_xal[:, ind_k_RT] / 2 / np.pi ** 2,
                       color=color, alpha=0.5, ls='--', lw=4)

        axis5.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * np.abs(PS_ab[:, ind_k_RT]) / 2 / np.pi ** 2, color='C1',
                       alpha=0.5, ls='--', lw=4)
        # axis5.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * np.abs(PS_Tb[:, ind_k_RT]) / 2 / np.pi ** 2, color='C1',alpha=0.5, ls='--', lw=4)
        axis5.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * np.abs(PS_aT[:, ind_k_RT]) / 2 / np.pi ** 2, color='C2',
                       alpha=0.5, ls='--', lw=4)
        # axis5.semilogy(RT_zz, PS_RT['k'][ind_k_RT]**3 * np.abs(PS_rb[:,ind_k_RT])/2/np.pi**2,color='C3',alpha=0.5,ls='--',lw=4)
        axis5.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * np.abs(PS_ra[:, ind_k_RT]) / 2 / np.pi ** 2, color='C3',
                       alpha=0.5, ls='--', lw=4)
        # axis5.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * np.abs(PS_rT[:, ind_k_RT]) / 2 / np.pi ** 2, color='C5',alpha=0.5, ls='--', lw=4)

        ind_k = np.argmin(np.abs(PS['k'] - PS_RT['k'][ind_k_RT]))

        axis5.semilogy(PS['z'], PS['k'][ind_k] ** 3 * HM_dTb ** 0 * PS['P_aa'][:, ind_k] / 2 / np.pi ** 2, color=color,
                       label='lya-lya k={}'.format(k0))
        axis5.semilogy(PS['z'], PS['k'][ind_k] ** 3 * np.abs(PS['P_ba'][:, ind_k]) / 2 / np.pi ** 2, color='C1',
                       label='k=am'.format(k0))
        # axis5.semilogy(PS['z'], PS['k'][ind_k] ** 3 * np.abs(PS['P_bT'][:, ind_k]) / 2 / np.pi ** 2, color='C1', label='k=Tm'.format(k0))
        axis5.semilogy(PS['z'], PS['k'][ind_k] ** 3 * np.abs(PS['P_Ta'][:, ind_k]) / 2 / np.pi ** 2, color='C2',
                       label='k=aT'.format(k0))
        # axis5.semilogy(PS['z'],PS['k'][ind_k]**3 * np.abs(PS['P_rb'][:,ind_k])/2/np.pi**2,color='C2',label='k=rb'.format(k0))
        axis5.semilogy(PS['z'], PS['k'][ind_k] ** 3 * np.abs(PS['P_ra'][:, ind_k]) / 2 / np.pi ** 2, color='C3',
                       label='k=ra'.format(k0))
        # axis5.semilogy(PS['z'], PS['k'][ind_k] ** 3 * np.abs(PS['P_rT'][:, ind_k]) / 2 / np.pi ** 2, color='C5',    label='k=rT'.format(k0))

        print('k RT is ', round(PS_RT['k'][ind_k_RT], 2), 'k HM is,', PS['k'][ind_k])

    axis5.set_ylim(1e-5, 1e1)
    axis5.set_xlim(6, 18)

    axis5.set_ylabel('$\Delta^{2} = k^{3}P(k)/(2\pi^{2})$   ', fontsize=14)

    axis5.legend(title='lyal PS ')
    axis5.set_xlabel('z', fontsize=14)

    ################################################ Beta factors
    axis6 = fig.add_subplot(426)
    ax = plt.gca()

    axis6.semilogy(RT_zz, np.abs(beta_a), 'r', ls='--')
    axis6.semilogy(RT_zz, np.abs(beta_T), 'b', ls='--')
    axis6.semilogy(RT_zz, np.abs(beta_r), 'g', ls='--')

    axis6.semilogy(HM_zz, np.abs(beta_a_HM), 'r', label='|beta_a|')
    axis6.semilogy(HM_zz, np.abs(beta_T_HM), 'b', label='|beta_T|')
    axis6.semilogy(HM_zz, np.abs(beta_r_HM), 'g', label='|beta_r|')
    axis6.legend()
    axis6.set_ylim(1e-5, 1e2)
    axis6.set_xlim(6, 18)

    ################################################ PS_TT(z)
    axis7 = fig.add_subplot(427)
    ax = plt.gca()

    for k0 in k_array:
        color = next(ax._get_lines.prop_cycler)['color']
        ind_k_RT = np.argmin(np.abs(PS_RT['k'] - k0))
        axis7.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * RT_dTb ** 0 * PS_TT[:, ind_k_RT] / 2 / np.pi ** 2,
                       color=color, alpha=0.5, ls='--', lw=4)

        ind_k = np.argmin(np.abs(PS['k'] - PS_RT['k'][ind_k_RT]))
        axis7.semilogy(PS['z'], PS['k'][ind_k] ** 3 * HM_dTb ** 0 * PS['P_TT'][:, ind_k] / 2 / np.pi ** 2, color=color,
                       label='k={}'.format(k0))
        # PS_CC = ((1+PS['Plin']**0.5)**(2/3) - 1)**2
        # axis7.semilogy(PS['z'],PS['k'][ind_k]**3 * HM_dTb**0* PS_CC[:,ind_k]/2/np.pi**2,color=color,label='k={}'.format(k0))
        print('k RT is ', round(PS_RT['k'][ind_k_RT], 2), 'k HM is,', PS['k'][ind_k])

    axis7.set_ylim(1e-4, 1e3)
    axis7.set_xlim(6, 22)

    axis7.set_ylabel('$\Delta^{2} = dTb^{2} k^{3}P(k)/(2\pi^{2})$   ', fontsize=14)

    axis7.legend(title='TT PS ')
    axis7.set_xlabel('z', fontsize=14)

    ################################################ PS_reio_reio(z)
    axis8 = fig.add_subplot(428)
    ax = plt.gca()

    for k0 in k_array:
        color = next(ax._get_lines.prop_cycler)['color']

        ind_k_RT = np.argmin(np.abs(PS_RT['k'] - k0))
        axis8.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * RT_dTb ** 0 * PS_rr[:, ind_k_RT] / 2 / np.pi ** 2,
                       color=color, alpha=0.5, ls='--', lw=4)

        ind_k = np.argmin(np.abs(PS['k'] - PS_RT['k'][ind_k_RT]))
        axis8.semilogy(PS['z'], PS['k'][ind_k] ** 3 * HM_dTb ** 0 * PS['P_rr'][:, ind_k] / 2 / np.pi ** 2, color=color,
                       label='k={}'.format(np.round(PS_RT['k'][ind_k_RT], 2)))

        axis8.set_ylim(1e-7, 1e0)
        print('k RT is ', round(PS_RT['k'][ind_k_RT], 2), 'k HM is,', PS['k'][ind_k])

    axis8.set_xlim(6, 18)

    axis8.set_ylabel('$\Delta^{2} = dTb^{2} k^{3}P(k)/(2\pi^{2})$   ', fontsize=14)

    axis8.legend(title='reio-reio PS ')
    axis8.set_xlabel('z', fontsize=14)

    if save_loc is not None:
        plt.savefig(save_loc)


def Plotting_PS_TT(k_array, physics, PS_RT, GS, PS, save_loc=None):
    """""""""""
    k_array,z : values for the power spectra plot
    physics,PS_Dict : GS history and PS dictionnary data from radtrans
    GS, PS : from halo model
    """""""""""
    ################################################ loading
    from matplotlib import pyplot as plt
    RT_zz = physics['z']  #
    fig = plt.figure(figsize=(10, 10))
    Beta_T_HM = GS['Tcmb'] / (GS['Tgas'] - GS['Tcmb'])
    ################################################ P_TT (z)
    axis4 = fig.add_subplot(111)
    ax = plt.gca()
    for k0 in k_array:
        color = next(ax._get_lines.prop_cycler)['color']
        ind_k_RT = np.argmin(np.abs(PS_RT['k'] - k0))
        axis4.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * PS_RT['PS_T'][:, ind_k_RT] / 2 / np.pi ** 2, color=color,
                       alpha=0.5, ls='--', lw=4)

        ind_k = np.argmin(np.abs(PS['k'] - PS_RT['k'][ind_k_RT]))
        axis4.semilogy(PS['z'], PS['k'][ind_k] ** 3 * PS['P_TT'][:, ind_k] / Beta_T_HM ** 2 / 2 / np.pi ** 2,
                       color=color, label='k={}'.format(k0))
        print('k RT is ', round(PS_RT['k'][ind_k_RT], 4), 'k HM is,', PS['k'][ind_k])

    axis4.semilogy([], [], color='gray', label='Beorn', alpha=0.5, ls='--', lw=4)
    axis4.semilogy([], [], color='gray', label='HM', ls='-.')

    # axis4.set_ylim(1e-1, 1e3)
    axis4.set_xlim(6, 20)

    axis4.legend(title='$ k^{3}P(k)/(2\pi^{2})$')  # dTb^{2}
    axis4.set_xlabel('z', fontsize=14)

    if save_loc is not None:
        plt.savefig(save_loc)


def Plot_heat_profiles(profile, HM_PS, zz, label, color, rho='heat'):
    """""""""
    RT_profile : profile dictionnary from radtrans
    HM_PS : output dic of coda.halomodel
    rho : str, either xray or heat
    """""""""
    # profile = load_profile(RT_profile)
    PS = HM_PS
    ind_z = np.argmin(np.abs(profile.z_history - zz))
    z_RT = profile.z_history[ind_z]
    M_halo = profile.Mh_history[ind_z]
    print('solid is HaloModel dashed is RadTrans')
    print('in RT z is', z_RT, 'Mhalo is {:2e}'.format(M_halo))
    if rho == 'heat':
        plt.loglog(profile.r_grid_cell, profile.T_history[str(round(z_RT, 2))], color=color, ls='--')
    elif rho == 'xray':
        plt.loglog(profile.r_grid_cell, profile.rhox_history[str(round(z_RT, 2))], color=color, ls='--')

    rmin, rmax, Nr = 0.1, 6000, 50
    rr = np.logspace(np.log10(rmin), np.log10(rmax), Nr)

    M0, rho_heat, rho_xray = PS['M0'], PS['rho_heat_prof'], PS['rho_X_prof']
    ind_z_HM = np.argmin(np.abs(PS['z'] - z_RT))

    if rho == 'xray':
        ind_M_HM = np.argmin(np.abs(PS['M_accr'][ind_z_HM] - M_halo))
        plt.loglog(rr, rho_xray[ind_z_HM, :, ind_M_HM] * eV_per_erg, label=label, color=color, ls='-')
        print('in HM z is', PS['z'][ind_z_HM], 'Mhalo is {:2e}'.format(PS['M_accr'][ind_z_HM, ind_M_HM]))
        plt.ylim(1e-30, 1e-12)
    if rho == 'heat':
        ind_M_HM = np.argmin(np.abs(PS['M0'] - M_halo))
        plt.loglog(rr, rho_heat[ind_z_HM, :, ind_M_HM], label=label, color=color, ls='-')
        print('in HM z is', PS['z'][ind_z_HM], 'Mhalo is {:2e}'.format(PS['M0'][ind_M_HM]))
        plt.ylim(1e-7, 1e6)


########### FUNCTIONS TO DO COMPARISON PLOTS WITH ------>21CM_FAST<--------
def plot_FAST(Fast_Model, qty='zz', xlim=None, ylim=None, label='', color='C0', ls='-'):
    """""""""
    This functions plots 21cmFAST global quantities.
    
    Parameters
    ----------    
    Fast_Model : dictionnary, output of GlogalSignal_21cmFast (function below)
    qty : str. quantity to plot
    """""""""
    try:
        zz, quantity = Fast_Model['zz'], Fast_Model[qty]
        plt.plot(zz, quantity, label=label, color=color, ls=ls)
    except Exception:
        print('available qty in Fast: ', Fast_Model.keys())
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()


def GlogalSignal_21cmFast(path):
    """""""""
    Transform the 21cmFAST globalsignal file into a clear dictionnary

    Parameters
    ----------    
    path : global_evolution data file output by 21cmFAST run (int Output_files folder)
    """""""""
    file = np.loadtxt(path)
    zz = file[:, 0]
    xHI = file[:, 1]
    Tk = file[:, 2]
    Tspin = file[:, 4]
    Tcmb = file[:, 5]
    Om, Ob, h0 = 0.31, 0.045, 0.68
    factor = 27 * (1 / 10) ** 0.5 * (Ob * h0 ** 2 / 0.023) * (Om * h0 ** 2 / 0.15) ** (-0.5)
    dTb = factor * np.sqrt(1 + zz) * (1 - Tcmb / Tspin) * xHI
    return {'zz': zz, 'xHI': xHI, 'Tk': Tk, 'Tspin': Tspin, 'Tcmb': Tcmb, 'dTb': dTb}


def plot_FAST_RT_PS_of_z(k, k_values_fast, path_to_FAST_PS, GS_Beorn, PS_Beorn, color='b', RSD=False, option='dTb_GS',
                         label=''):
    """""""""
    Comparison Plot Between BEORN and 21cmFAST. Find the indice to get the same k value in Mpc.
    """""""""
    ind_k_fast = np.argmin(np.abs(k_values_fast - k))
    FAST_PS_of_z = np.loadtxt(path_to_FAST_PS + 'PS_k' + str(ind_k_fast))
    plt.semilogy(FAST_PS_of_z[:, 0], FAST_PS_of_z[:, 1], ls='-', alpha=0.4, lw=8)
    print('k fast is', k_values_fast[ind_k_fast])

    ind_k = np.argmin(np.abs(PS_Beorn['k'] * 0.68 - k))
    print('k RT is', PS_Beorn['k'][ind_k] * 0.68)

    kk, PS_dTb_RT = PS_Beorn['k'][ind_k], PS_Beorn['PS_dTb'][:, ind_k]

    print('For Beorn, mutliplying by ', option)
    dTb_RT = GS_Beorn[option]
    plt.xlabel('k [1/Mpc]', fontsize=14)
    plt.ylabel('$\Delta^{2} = dTb^{2} k^{3}P(k)/(2\pi^{2})$   ', fontsize=14)
    plt.semilogy(GS_Beorn['z'], kk ** 3 * dTb_RT ** 2 * PS_dTb_RT / 2 / np.pi ** 2,
                 label='k=' + str(round(k_values_fast[ind_k_fast], 2)) + 'Mpc$^{-1}$' + ' ' + label, color=color, lw=2)
    if RSD:
        dTb_RSD, PS_dTb_RT_RSD = GS_Beorn['dTb_RSD'], PS_Beorn['PS_dTb_RSD'][:, ind_k]
        plt.semilogy(GS_Beorn['z'], kk ** 3 * dTb_RSD ** 2 * PS_dTb_RT_RSD / 2 / np.pi ** 2,
                     label='k=' + str(round(k_values_fast[ind_k_fast], 2)), color=color)
        print(PS_dTb_RT_RSD / PS_dTb_RT)


def plot_HM_PS_of_z(k, PS, color, label='', ls='--'):
    """""""""
    Plot a HM PS as a fct of k
    PS : output of coda.halomodel

    """""""""
    kk, zz = PS['k'], PS['z']
    ind_k = np.argmin(np.abs(kk * 0.68 - k))
    print('k HM is', PS['k'][ind_k] * 0.68)
    try:
        plt.semilogy(zz, kk[ind_k] ** 3 * PS['dTb'] ** 2 * np.abs(PS['P_mu0'][:, ind_k]) / 2 / np.pi ** 2, ls=ls, lw=1,
                     alpha=1, label=label, color=color)
    except Exception:
        print('RT:', PS.keys())
    plt.legend()
    plt.xlabel('k [1/Mpc]')


def plot_PS_fast(z, file, color, ax, Beta=1, label='', qty='xHII', ls='--', alpha=0.5, lw=3):
    """""""""
    Plot a 21cm FAST power spectrum as a function of k. 
    
    Parameters
    ---------
    file : should be the Power spectrum pickle file computed from the Boxes in 21cmFAST
    Beta : float, the beta factor in the perturbative expansion of dTb. Set to 1 by default.
    """""""""
    print('-------------FAST -- plotting power spectrum of ', qty, 'at redshift ', z)
    PS = load_f(file)
    kk, zz = PS['k'], PS['z']
    ind_z = np.argmin(np.abs(zz - z))
    try:
        ax.loglog(kk, kk ** 3 * Beta * np.abs(PS['PS_' + qty][ind_z]) / 2 / np.pi ** 2, lw=lw, alpha=alpha, label=label,
                  color=color, ls=ls)
    except Exception:
        print('Fast :', PS.keys())
    plt.legend()
    plt.xlabel('k [1/Mpc]')


def plot_PS_HM(z, PS, color, ax, label='', qty='rr', with_dTb=False):
    """""""""
    Plot a HM PS as a fct of k
    PS : output of coda.halomodel
    
    """""""""
    print('-------------HaloModel -- plotting power spectrum of ', qty, 'at redshift ', z)
    kk, zz = PS['k'], PS['z']
    ind_k = np.intersect1d(np.where(kk > 8e-2), np.where(kk < 5))
    kk = kk[ind_k]
    ind_z = np.argmin(np.abs(zz - z))
    coef = 1
    if with_dTb:
        coef = PS['dTb'][ind_z] ** 2
    try:
        ax.loglog(kk * 0.68, coef * kk ** 3 * np.abs(PS['P_' + qty][ind_z][ind_k]) / 2 / np.pi ** 2, ls='-', lw=1,
                  alpha=1, label=label, color=color)
    except Exception:
        print('RT:', PS.keys())
    plt.legend()
    plt.xlabel('k [1/Mpc]')


def plot_FAST_PS_k(z, z_liste_fast, path_to_deldel, ax, color='b', with_dTb=True):
    """""""""
    Plot a 21cm FAST power spectrum as a function of k from the power spectra files output by 21cmFAST (in Deldel_T_power_spec folder)

    Parameters
    ---------
    z_liste_fast : liste of 21cmFAST redshift values 
    """""""""
    import os
    directory = os.fsencode(path_to_deldel)
    z_liste = load_f(z_liste_fast)
    z_fast = z_liste[np.argmin(np.abs(z_liste - z))]
    print('z_fast is ', z_fast)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if z_fast < 10:
            if filename.startswith("ps_z00" + "{:.2f}".format((z_fast))):
                FAST_Model_1_PS = np.loadtxt(path_to_deldel + filename)
                break
        else:
            if filename.startswith("ps_z0" + "{:.2f}".format((z_fast))):
                FAST_Model_1_PS = np.loadtxt(path_to_deldel + filename)
                break
    ind = filename.find('aveTb') + 5  ## find in the stringname where dTb is written
    dTb = float(filename[ind:ind + 6])
    print('at z = ', z_fast, 'dTb FAST IS : ', dTb)

    if with_dTb:
        ax.loglog(FAST_Model_1_PS[:, 0], FAST_Model_1_PS[:, 1], color=color, ls='--')
    else:
        ## here we extract the dTb value
        ax.loglog(FAST_Model_1_PS[:, 0], FAST_Model_1_PS[:, 1] / dTb ** 2, color=color, ls='--')

        print(filename)
    ax.legend()
    ax.set_xlabel('k[1/Mpc]', fontsize=13)
    ax.set_ylabel('$\Delta^{2} $', fontsize=13)


def Beta_Beorn(z, GS, qty='xHII'):
    """""""""
    Compute the beta factors at a given redshift

    Parameters
    ---------
    GS : global signal dictionnary from Beorn
    """""""""
    ind_z = np.argmin(np.abs(GS['z'] - z))
    if qty == 'xHII':
        return GS['beta_r'][ind_z]
    if qty == 'xal':
        return GS['beta_a'][ind_z]
    if qty == 'T':
        return GS['beta_T'][ind_z]
    if qty == 'Tspin':
        return Tcmb0 * (1 + z) / (GS['T_spin'][ind_z] - Tcmb0 * (1 + z))
    else:
        print('Beta_Beorn qty should be xHII, xal, T or Tspin')


def Beta_21cmFast(z, GS, qty='x_HII'):
    """""""""
    Compute the beta factors at a given redshift

    Parameters
    ---------
    GS : output of GlogalSignal_21cmFast
    """""""""
    ind_z = np.argmin(np.abs(GS['zz'] - z))
    if qty == 'xHII':
        return -(1 - GS['xHI'][ind_z]) / (GS['xHI'][ind_z])
    if qty == 'Tspin':
        return GS['Tcmb'][ind_z] / (GS['Tspin'][ind_z] - GS['Tcmb'][ind_z])
    if qty == 'T':
        return GS['Tcmb'][ind_z] / (GS['Tk'][ind_z] - GS['Tcmb'][ind_z])
    else:
        print('Beta_21cmFast qty should be xHII, T, or Tspin')


def horizontal_plot_for_lightcone(Beorn_GS, Beorn_PS):
    """""""""
    Do a nice horizontal plot to show next to lightcone
    Smoothes the globaldTb if needed

    Parameters
    ---------
    Beorn_GS,Beorn_PS : output of compute_GS and compute_PS from Beorn.
    
    """""""""

    from scipy.signal import savgol_filter as savitzky_golay

    plt.figure(figsize=(20, 3))
    # plt.plot(1/(Beorn_GS['z']+1),Beorn_GS['dTb'],lw=6,alpha=0.5) # to check that savitzky_golay is doing the right thing
    yhat = savitzky_golay(Beorn_GS['dTb'], 11, 3)
    plt.plot(1 / (Beorn_GS['z'] + 1), yhat, lw=3, alpha=1)
    plt.ylabel('$dT_{b}$ (mK)', fontsize=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.savefig('./dTb_For_Lightcone_Plot.pdf')

    plt.figure(figsize=(20, 3))
    for k in [0.1, 0.5, 1]:
        ind_k = np.argmin(np.abs(Beorn_PS['k'] * 0.68 - k))
        label = ''

        # plt.plot(1/(Beorn_GS['z']+1),Beorn_GS['dTb'],lw=6,alpha=0.5)
        # yhat = savitzky_golay(Beorn_GS['dTb'], 11, 3)
        # plt.plot(1/(Beorn_GS['z']+1),Beorn_PS['PS_dTb'][],lw=3,alpha=1)

        PS_dTb = Beorn_PS['PS_dTb'][:, ind_k]
        indices_ = np.where(np.invert(np.isnan(PS_dTb)))
        kk, PS_dTb_RT = Beorn_PS['k'][ind_k], PS_dTb[
            indices_]  # 10**savitzky_golay(np.log10(Beorn_PS['PS_dTb'][:, ind_k]), 11, 3)
        dTb_RT = Beorn_GS['dTb'][indices_]
        plt.xlabel('k [1/Mpc]', fontsize=14)
        plt.ylabel('$\Delta^{2} = dTb^{2} k^{3}P(k)/(2\pi^{2})$   ', fontsize=14)
        plt.semilogy(1 / (Beorn_GS['z'][indices_] + 1), kk ** 3 * dTb_RT ** 2 * PS_dTb_RT / 2 / np.pi ** 2, lw=5,
                     alpha=0.4, label='k=' + str(round(Beorn_PS['k'][ind_k] * 0.68, 2)) + 'Mpc$^{-1}$')

        plt.ylabel('$dT_{b}$ (mK)', fontsize=20)
        plt.xticks(size=15)
        plt.yticks(size=15)
    plt.legend(fontsize=15, loc='upper left')
    plt.savefig('./Model_3_Beorn_P(z)_For_Lightcone_Plot.pdf')
