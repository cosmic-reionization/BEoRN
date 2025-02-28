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
from .functions import Beta, find_nearest
from beorn.cosmo import dTb_fct

def Delta_21cm_PS_fixed_k(k,PS,plot=True):
    kk, zz = PS['k'], PS['z']
    k_value, ind_k = find_nearest(kk,k)
    if plot:
        print('k-value picked is ',k_value,'h/Mpc.')
    Delta_sq = k_value**3*PS['dTb']**2 * PS['PS_dTb'][:,ind_k]/2/np.pi**2
    return zz, Delta_sq, k_value

def nu_MHz(aa):
    # observed 21cm frequency in Mhz for a given scale fac aa
    return 1420*aa


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


def plot_Beorn_PS_of_z(k, GS_Beorn, PS_Beorn, ls='-', lw=1, color='b', RSD=False, label='', qty='dTb', alpha=1, ax=plt,expansion=False):
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

    if expansion :
        dTb_PS_HM_style = expansion_dTb_PS(PS_Beorn)
        ax.semilogy(GS_Beorn['z'], kk ** 3 * dTb_RT ** 2 * dTb_PS_HM_style[:, ind_k] / 2 / np.pi ** 2, ls='-', lw=4, alpha=0.5,
                    color=color,label='expansion')


def plot_2d_map(grid,  Lbox=None, slice_nbr=None, qty='label', scale='lin'):
    # Ncell : int, nbr of grid pixels
    # slice_nbr : int, slice to plot
    # Lbox : int, Box size in Mpc/h
    Ncell = grid.shape[0]
    print('Ncell is ',Ncell)
    if Lbox is None :
        Lbox = 100
        print('No Lbox provided, assuming 100Mpc/h (only relevant for the label)')
    if slice_nbr is None:
        slice_nbr = int(Ncell/2)
        print('No slice number provided, plotting the slice (:,Ncell/2,:)')
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




def expansion_terms_beorn(PS):
    from beorn.functions import Beta

    kk,zz = PS['k'],PS['z']
    beta_r,beta_T,beta_a = Beta(zz,PS,qty='reio'), Beta(zz,PS,qty='Tk'), Beta(zz,PS,qty='lyal')

    beta_a_2 = - PS['x_al'] * beta_a**2 # coef in front of delta_a^2 in Taylor expansion of dTb

    for key, value in PS.items():  # change lists to numpy arrays
        PS[key] = np.nan_to_num(PS[key])

    PS_rr = PS['PS_xHII'] * beta_r[:, None] ** 2  ### sizes are size(kk)# reio
    PS_TT = PS['PS_T'] * beta_T[:, None] ** 2 # temp
    PS_bb = PS['PS_rho'] # Matter
    PS_aa = PS['PS_xal'] * beta_a[:, None] ** 2 # lyal

    PS_ab = beta_a[:, None] * PS['PS_rho_xal']
    PS_Tb = beta_T[:, None] * PS['PS_rho_T']
    PS_aT = beta_a[:, None] * beta_T[:, None] * PS['PS_T_lyal']
    PS_rb = beta_r[:, None] * PS['PS_rho_xHII']
    PS_ra = beta_r[:, None] * beta_a[:, None] * PS['PS_lyal_xHII']
    PS_rT = beta_r[:, None] * beta_T[:, None] * PS['PS_T_xHII']
    print('Returining: PS_bb ,PS_aa ,PS_TT ,PS_rr ,PS_ab , PS_Tb , PS_aT , PS_rb , PS_ra , PS_rT')

    return PS_bb ,PS_aa ,PS_TT ,PS_rr ,PS_ab , PS_Tb , PS_aT , PS_rb , PS_ra , PS_rT

def expansion_terms_beorn_HO_reio(PS):
    from beorn.functions import Beta

    kk, zz = PS['k'], PS['z']
    beta_r, beta_T, beta_a = Beta(zz, PS, qty='reio'), Beta(zz, PS, qty='Tk'), Beta(zz, PS, qty='lyal')

    for key, value in PS.items():  # change lists to numpy arrays
        PS[key] = np.nan_to_num(PS[key])

    PS_rbb = PS['PS_rb_b'] * (2 * beta_r)[:, None]
    PS_rrb = PS['PS_br_r'] * (2 * beta_r ** 2)[:, None]
    PS_rbrb = PS['PS_rb_rb'] * (beta_r ** 2)[:, None]

    return PS_rbb ,PS_rrb ,PS_rbrb


def expansion_dTb_PS(PS,higher_order=False ,HO_r_b=False, coef=1,coef_3th_order=0,coef_4th_order=0,remove_T_2nd_order = 1):
    """
    Parameters
    ----------
    PS : BEoRN output power spectra, with all cross correlations computed.
    higher_order : Bool. Includes higher order PS term including xHII field. (r)
    coef : 1 or 0, used to suppress the terms originating from axpansion to 2nd order of xal/(1+xal) and 1/T

    remove_T_2nd_order : if equals to zero, removes all terms coming from second order T expansion in dTb formula :  -beta_T * delta_T**2

    Returns
    ----------
    The dTb PS using HaloModel formula.
    """

    from beorn.functions import Beta

    kk,zz = PS['k'],PS['z']
    beta_r,beta_T,beta_a = Beta(zz,PS,qty='reio'), Beta(zz,PS,qty='Tk'), Beta(zz,PS,qty='lyal')

    beta_a_2 = - PS['x_al'] * beta_a**2 # coef in front of delta_a^2 in Taylor expansion of dTb

    for key, value in PS.items():  # change lists to numpy arrays
        PS[key] = np.nan_to_num(PS[key])

    PS_rr = PS['PS_xHII'] * beta_r[:, None] ** 2  ### sizes are size(kk)# reio
    PS_TT = PS['PS_T'] * beta_T[:, None] ** 2 # temp
    PS_bb = PS['PS_rho'] # Matter
    PS_aa = PS['PS_xal'] * beta_a[:, None] ** 2 # lyal

    PS_ab = beta_a[:, None] * PS['PS_rho_xal']
    PS_Tb = beta_T[:, None] * PS['PS_rho_T']
    PS_aT = beta_a[:, None] * beta_T[:, None] * PS['PS_T_lyal']
    PS_rb = beta_r[:, None] * PS['PS_rho_xHII']
    PS_ra = beta_r[:, None] * beta_a[:, None] * PS['PS_lyal_xHII']
    PS_rT = beta_r[:, None] * beta_T[:, None] * PS['PS_T_xHII']

    PS_dTb_HM_style = PS_bb + PS_aa + PS_TT + PS_rr + 2 * (PS_ab + PS_Tb + PS_aT + PS_rb + PS_ra + PS_rT)

    if HO_r_b:
        PS_rrb = PS['PS_br_r'] * (2 * beta_r ** 2)[:, None]
        PS_rbb = PS['PS_rb_b'] * (2 * beta_r)[:, None]
        PS_rbrb = PS['PS_rb_rb'] * (beta_r ** 2)[:, None]
        PS_dTb_HM_style += PS_rbb +PS_rrb +PS_rbrb

    if higher_order:
        print('computing PS to 3rd order in delta_r')
        PS_raa = PS['PS_ra_a'] * 2 * (beta_r * beta_a ** 2)[:, None] + coef* PS['PS_r_aa'] * 2 * (beta_r * beta_a_2)[:, None]
        PS_rTT = PS['PS_rT_T'] * 2 * (beta_r * beta_T ** 2)[:, None] - remove_T_2nd_order * coef * PS['PS_r_TT'] * 2 * (beta_r * beta_T)[:, None]
        PS_rbb = PS['PS_rb_b'] * (2 * beta_r)[:, None]

        #PS_raa = PS['PS_raa'] * (2 * beta_r * beta_a ** 2 + coef * 2 * beta_r * beta_a_2)[:, None]
        #PS_rTT = PS['PS_rTT'] * (2 * beta_r * beta_T ** 2 - coef * 2 * beta_r * beta_T)[:, None]

        PS_abr = (PS['PS_ab_r']+PS['PS_rb_a']+PS['PS_ra_b']) * (2 * beta_a * beta_r)[:, None]  ## instead of factor 6.
        PS_rTb = (PS['PS_rT_b']+PS['PS_rb_T']+PS['PS_Tb_r']) * (2 * beta_T * beta_r)[:, None]
        PS_aTr = (PS['PS_aT_r']+PS['PS_ar_T']+PS['PS_Tr_a']) * (2 * beta_T * beta_r * beta_a)[:, None]

        PS_rra = PS['PS_ar_r'] * (2 * beta_r ** 2 * beta_a)[:, None]
        PS_rrb = PS['PS_br_r'] * (2 * beta_r ** 2)[:, None]
        PS_rrT = PS['PS_Tr_r'] * (2 * beta_r ** 2 * beta_T)[:, None]

        #PS_rTrT = PS['PS_rTrT'] * (beta_T * beta_r ** 2 * (beta_T - 2))[:, None]
        #PS_rara = PS['PS_rara'] * (beta_a ** 2 * beta_r ** 2 * (1 - 2 * PS['x_al']))[:, None]
        PS_rara = PS['PS_ra_ra'] * (beta_r **2 * beta_a ** 2)[:, None] + coef* PS['PS_raa_r'] * 2 * (beta_r ** 2 * beta_a_2)[:, None]
        PS_rTrT = PS['PS_rT_rT'] * (beta_T **2 * beta_r ** 2)[:, None] - remove_T_2nd_order * coef* PS['PS_rTT_r'] * 2 * (beta_r ** 2 * beta_T)[:, None]
        PS_rbrb = PS['PS_rb_rb'] * (beta_r ** 2)[:, None]

        PS_rarb = (2*PS['PS_rba_r']+2*PS['PS_rb_ra']) * (beta_r ** 2 * beta_a)[:, None]
        PS_rTra = (2*PS['PS_rTa_r']+2*PS['PS_rT_ra']) * (beta_r ** 2 * beta_T * beta_a)[:, None]
        PS_rTrb = (2*PS['PS_rTb_r']+2*PS['PS_rT_rb']) * (beta_r ** 2 * beta_T)[:, None]

        PS_dTb_HM_style += PS_rTT + PS_raa + PS_rbb + PS_rTb + PS_abr + PS_aTr + PS_rrT + PS_rra + PS_rrb + \
                           PS_rTrT + PS_rara + PS_rbrb + PS_rarb + PS_rTrb + PS_rTra


    if coef_3th_order>0:
        ## 3rd order in a, T and b (aaT,aaa, TTT,TTa...)
        PS_aa_a = PS['PS_aa_a'] * (2 * beta_a * beta_a_2)[:, None]
        PS_aa_T = PS['PS_aa_T'] * (2 * beta_T * beta_a_2)[:, None]
        PS_aT_a = PS['PS_aT_a'] * (2 * beta_T * beta_a**2)[:, None]
        PS_aT_T = PS['PS_aT_T'] * (2 * beta_T**2 * beta_a)[:, None]
        PS_TT_a = - remove_T_2nd_order * PS['PS_TT_a'] * (2 * beta_T * beta_a)[:, None]
        PS_TT_T = - remove_T_2nd_order * PS['PS_TT_T'] * (2 * beta_T**2)[:, None]

        PS_ab_a = PS['PS_ab_a'] * (2 * beta_a**2)[:, None]
        PS_aa_b = PS['PS_aa_b'] * (2 * beta_a_2)[:, None]
        PS_ab_b = PS['PS_ab_b'] * (2 * beta_a)[:, None]

        PS_abT = (PS['PS_ab_T'] + PS['PS_aT_b'] + PS['PS_bT_a']) * (2 * beta_a * beta_T)[:, None]

        PS_bT_b = PS['PS_bT_b'] * (2 * beta_T)[:, None]
        PS_Tb_T = PS['PS_Tb_T'] * (2 * beta_T**2)[:, None]
        PS_TT_b = - remove_T_2nd_order * PS['PS_TT_b'] * (2 * beta_T)[:, None]




        ##### 4th order terms in a T and b
        PS_aa_aa= PS['PS_aa_aa'] * (beta_a_2 **2)[:, None]
        PS_TT_TT= remove_T_2nd_order * PS['PS_TT_TT'] * (beta_T **2)[:, None]

        PS_aaba = (PS['PS_aa_ba'] + PS['PS_aab_a']) * (2 * beta_a * beta_a_2)[:, None]
        PS_ab_ab = PS['PS_ab_ab'] * (beta_a ** 2)[:, None]
        PS_aab_b = PS['PS_aab_b'] * (2 * beta_a_2)[:, None]
        PS_aaTa = (PS['PS_aaT_a'] + PS['PS_aa_Ta']) * (2 * beta_T * beta_a * beta_a_2)[:, None]

        PS_baTa = (PS['PS_baT_a'] + PS['PS_ba_aT']) * (2 * beta_T * beta_a ** 2)[:, None]
        PS_aaTb = (PS['PS_aaT_b'] + PS['PS_aab_T'] + PS['PS_aa_Tb']) * (2 * beta_T * beta_a_2)[:, None]

        PS_abTb = (PS['PS_ba_Tb'] + PS['PS_abT_b']) * (2 * beta_T * beta_a)[:, None]
        PS_aTT_a = remove_T_2nd_order * PS['PS_aTT_a'] * (-2 * beta_T * beta_a ** 2)[:, None]
        PS_aa_TT = remove_T_2nd_order * PS['PS_aa_TT'] * (-2 * beta_T * beta_a_2)[:, None]
        PS_aT_aT = PS['PS_aT_aT'] * (beta_T ** 2 * beta_a ** 2)[:, None]

        PS_aaT_T = PS['PS_aaT_T'] * (2 * beta_T ** 2 * beta_a_2)[:, None]
        PS_aTTb = remove_T_2nd_order * (PS['PS_aTT_b'] + PS['PS_bTT_a'] + PS['PS_TT_ab']) * (-2 * beta_T * beta_a)[:, None]

        PS_bTaT = (PS['PS_baT_T'] + PS['PS_bT_aT']) * (2 * beta_T ** 2 * beta_a)[:, None]

        PS_bTT_b = remove_T_2nd_order * PS['PS_bTT_b'] * (-2 * beta_T)[:, None]

        PS_Tb_Tb = PS['PS_Tb_Tb'] * (beta_T ** 2)[:, None]
        PS_TTaT = remove_T_2nd_order * (PS['PS_aTT_T'] + PS['PS_TT_aT']) * (-2 * beta_T ** 2 * beta_a)[:, None]
        PS_TTbT = remove_T_2nd_order * (PS['PS_bTT_T'] + PS['PS_TT_bT']) * (-2 * beta_T ** 2)[:, None]



        PS_dTb_HM_style += coef_3th_order * (PS_aa_a + PS_aT_a + PS_aT_T + PS_aa_T + PS_TT_a + PS_TT_T + \
                           PS_ab_a + PS_aa_b + PS_ab_b + PS_abT + PS_bT_b + PS_Tb_T + PS_TT_b) + \
                           coef_4th_order * (PS_aa_aa + PS_TT_TT+PS_aaba +PS_ab_ab+PS_aab_b+PS_aaTa +PS_baTa +\
                           PS_aaTb +PS_abTb +PS_aTT_a+PS_aa_TT+PS_aT_aT+PS_aaT_T+PS_aTTb \
                           +PS_bTaT +PS_bTT_b+PS_Tb_Tb+PS_TTaT +PS_TTbT)



    return PS_dTb_HM_style



def expansion_dTb_globalsignal(param,PS,corr_fct):
    """
    Parameters
    ----------
    PS : BEoRN output power spectra, with all cross correlations computed.
    corr_fct : The real space correlation function for the unsmoothed field. Output of compute_corr_fct in run.py.

    Returns
    ----------
    The mean dTb including cross correlation, to second order
    """
    from beorn.functions import Beta
    from beorn.cosmo import dTb_fct
    kk,zz = PS['k'],PS['z']
    beta_r,beta_T,beta_a = Beta(zz,PS,qty='reio'),Beta(zz,PS,qty='Tk'),Beta(zz,PS,qty='lyal')

    for key, value in PS.items():  # change lists to numpy arrays
        PS[key] = np.nan_to_num(PS[key])
    for key, value in corr_fct.items():  # change lists to numpy arrays
        corr_fct[key] = np.nan_to_num(corr_fct[key])

    Xi_TT = corr_fct['Xi_TT']
    Xi_aa = corr_fct['Xi_aa']
    Xi_Tb = corr_fct['Xi_Tb']
    Xi_rT = corr_fct['Xi_rT']
    Xi_ar = corr_fct['Xi_ar']
    Xi_aT = corr_fct['Xi_aT']
    Xi_rb = corr_fct['Xi_rb']
    Xi_ab = corr_fct['Xi_ab']

    Xi_rba = corr_fct['Xi_rba']
    Xi_rbT = corr_fct['Xi_rbT']
    Xi_raT = corr_fct['Xi_raT']
    Xi_raa = corr_fct['Xi_raa']
    Xi_rTT = corr_fct['Xi_rTT']

    Xi_aTb  = corr_fct['Xi_aTb']
    Xi_aTrb = corr_fct['Xi_aTrb']
    Xi_aab  = corr_fct['Xi_aab']
    Xi_aarb = corr_fct['Xi_aarb']
    Xi_TTb  = corr_fct['Xi_TTb']
    Xi_TTrb = corr_fct['Xi_TTrb']

    x_al = PS['x_al']

    zero_order_correction = (1 + beta_r * Xi_rb)
    print('0. Zero Order correction just includes 1+ <delta_r*delta_b>')

    first_order_correction = 1 + beta_r * beta_T * Xi_rT + beta_T * Xi_Tb + beta_r * Xi_rb  + beta_a * beta_T * Xi_aT +\
                         beta_a * beta_r * Xi_ar  + beta_a * Xi_ab + beta_r * beta_T * Xi_rbT + beta_r * beta_T * beta_a * Xi_raT + beta_r * beta_a * Xi_rba
    print('1. Computing correction to GS including first order expansion of lyal and Tk, with all nonlin terms in xHII.')

    print('2. Second order includes 2nd order expansion of lyal and Tk, with all nonlin terms in xHII. Only keeping terms to 2nd order in lyal and Tk')

    sec_order_correction = first_order_correction - beta_a ** 2 * x_al * beta_r * Xi_raa - beta_a ** 2 * x_al * Xi_aa - beta_T * beta_r * Xi_rTT  - beta_T * Xi_TT

    dTb_approx = dTb_fct(PS['z'], PS['Tk'], PS['x_coll'] + PS['x_al'], 0, PS['x_HII'] , param)

    #if third_order_w_reio:  ## add up terms to 3rd order that include xHII
   #     sec_order_correction += beta_r * beta_a * Xi_rba + beta_r * beta_T * Xi_rbT + beta_r * beta_a * beta_T * Xi_raT \
   #                             - beta_a ** 2 * x_al * beta_r * Xi_raa - beta_T * beta_r * Xi_rTT
   # if third_order :## add up ALL 3rd order terms
   #     sec_order_correction += beta_a * beta_T * Xi_aTb + beta_a * beta_T * beta_r * Xi_aTrb - beta_a ** 2 * x_al * Xi_aab - beta_a ** 2 * x_al * beta_r *  Xi_aarb\
   #                             - beta_T * Xi_TTb - beta_T * beta_r * Xi_TTrb
    print('returning 4 arrays : uncorrected GS, zero, first, and second order corrections to dTb GS.')
    return dTb_approx, dTb_approx * zero_order_correction, dTb_approx * first_order_correction, dTb_approx * sec_order_correction


def plot_var_z(k, var,PS, ax=plt, ls='-', legend=True):
    """""""""
    Plot a Beorn variances (sigma(k,z)) as a function of z, for lyal xHII, and Tk fields
    """""""""
    ind_k = np.argmin(np.abs(var['k'] - k))
    print('k is', var['k'][ind_k], 'Mpc/h')
    var_lyal, var_xHII, var_Temp = var['var_lyal'][:, ind_k], var['var_xHII'][:, ind_k], var['var_Temp'][:, ind_k]

    if legend:
        label_1 = r'$\sigma$ ly-al'
        label_2 = r'$\sigma$ Tk'
        label_3 = r'$\sigma$ xHII'
       # label_4 = r'$\sigma$ T+lyal'

    else:
        label_1, label_2, label_3 = '', '', ''
    ax.semilogy(var['z'], np.sqrt(var_lyal)*PS['x_al']/(1+PS['x_al']), label=label_1, color='C0', ls=ls)
    ax.semilogy(var['z'], np.sqrt(var_Temp), label=label_2, color='C1', ls=ls)
   # ax.semilogy(var['z'], np.sqrt(var_Temp) + np.sqrt(var_lyal), label=label_4, color='C5', ls=ls)
    ax.semilogy(var['z'], np.sqrt(var_xHII), label=label_3, color='C2', ls=ls)
    ax.semilogy([], [], label='k={} h/Mpc'.format(round(var['k'][ind_k], 3)), color='gray', ls=ls)
    ax.legend()





def plot_Compare_expansion_with_true_PS(param,k, PS_Beorn, color='b', qty='dTb',variances=None,dTb_approx=True,coef=1,PS_Beorn_2=None,coef_3th_order=0,coef_4th_order=0,remove_T_2nd_order=1):
    """""""""
    Plot a Beorn Power Spectrum as a function of z. 
    """""""""
    import matplotlib.gridspec as gridspec
    matplotlib.rc('xtick', labelsize=15)
    matplotlib.rc('ytick', labelsize=15)
    fig = plt.figure(constrained_layout=True)
    fig.set_figwidth(10)
    fig.set_figheight(6)
    gs = gridspec.GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[:-1,0])
    ax2 = fig.add_subplot(gs[-1,0])
    ax3 = fig.add_subplot(gs[:-1,1])

    ind_k = np.argmin(np.abs(PS_Beorn['k'] - k))
    print('k RT is', PS_Beorn['k'][ind_k], 'Mpc/h')
    kk, PS_dTb_RT = PS_Beorn['k'][ind_k], PS_Beorn['PS_' + qty][:, ind_k]
    dTb_RT = PS_Beorn['dTb']
    ax1.semilogy(PS_Beorn['z'], kk ** 3 * dTb_RT ** 2 * PS_dTb_RT / 2 / np.pi ** 2, ls='-', lw='4', alpha=0.5,
                label='True PS, k='+str(round(kk,3)), color=color)


    dTb_PS_HM_style = expansion_dTb_PS(PS_Beorn,higher_order=False,coef=coef,coef_3th_order=coef_3th_order,coef_4th_order=coef_4th_order,remove_T_2nd_order=remove_T_2nd_order)
    dTb_PS_HM_style_sec_order = expansion_dTb_PS(PS_Beorn,higher_order=True,coef=coef,coef_3th_order=coef_3th_order,coef_4th_order=coef_4th_order,remove_T_2nd_order=remove_T_2nd_order)

    if dTb_approx:
        dTb_approx = dTb_fct(PS_Beorn['z'], PS_Beorn['Tk'], PS_Beorn['x_coll'] + PS_Beorn['x_al'], 0, PS_Beorn['x_HII'], param)
    else :
        dTb_approx = dTb_RT
    ax1.semilogy(PS_Beorn['z'], kk ** 3 * dTb_approx ** 2 * dTb_PS_HM_style[:, ind_k] / 2 / np.pi ** 2, ls='--', lw=2, alpha=1,
                    color='gray',label='First order')
    ax1.semilogy(PS_Beorn['z'], kk ** 3 * dTb_approx ** 2 * dTb_PS_HM_style_sec_order[:, ind_k] / 2 / np.pi ** 2, ls='-', lw=2,
                 alpha=1,
                 color='gray', label='Second order')


    ax2.plot(PS_Beorn['z'], PS_dTb_RT/dTb_PS_HM_style[:, ind_k]* dTb_RT ** 2 /dTb_approx**2 , ls='--', lw=2, alpha=1,
                    color=color)
    ax2.plot(PS_Beorn['z'], PS_dTb_RT / dTb_PS_HM_style_sec_order[:, ind_k] * dTb_RT ** 2 / dTb_approx ** 2, ls='-', lw=2,
             alpha=1,
             color=color)

    if PS_Beorn_2 is not None:
        kk, PS_dTb_RT = PS_Beorn_2['k'][ind_k], PS_Beorn_2['PS_' + qty][:, ind_k]
        dTb_RT = PS_Beorn_2['dTb']
        ax1.semilogy(PS_Beorn_2['z'], kk ** 3 * dTb_RT ** 2 * PS_dTb_RT / 2 / np.pi ** 2, ls='--', lw='2', alpha=0.5,
                     label='True PS, k=' + str(round(kk, 3)), color=color)


    ax1.legend()


    ax2.set_ylim(0.5,1.5)
    ax1.set_ylim(1e-2,6e2)
    ax1.set_xlim(5,20)
    ax2.set_xlim(5,20)
    ax2.hlines(y=1,xmin=5,xmax=20)

    if variances is not None:
        plot_var_z(k, variances,PS_Beorn, ax=ax3, ls='-', legend=True)


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

def Plotting_GS(physics, sfrd_beorn, GS, PS, save_loc=None,param=None):
    """""""""""
    physics : GS history dictionnary data from radtrans
    x__approx : approximation dictionnary (when we compute the global quantities from halo catalogs + profiles, without using a grid.)
    GS, PS : from halo model
    """""""""""
    ################################################ loading
    from matplotlib import pyplot as plt

    RT_dTb_GS = dTb_fct(physics['z'], physics['Tk'], physics['x_coll'] + physics['x_al'], 0, physics['x_HII'], param)

    RT_dTb_GS_Tkneutr = physics['dTb_GS_Tkneutral']
    RT_dTb = physics['dTb']  # physics['dTb']
    RT_zz = physics['z']
    #x_tot = physics['xal_coda_style'] + physics['x_coll']
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
    Tbg = Tcmb0 * (1 + 135) * (1 + RT_zz) ** 2 / (1 + 135) ** 2
    axis2.semilogy(RT_zz, Tbg, 'cyan', ls='--', label='Tadiab')
    #axis2.semilogy(RT_zz, physics['T_spin'], 'r', ls='--')
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
    axis3.semilogy(sfrd_beorn[0], sfrd_beorn[1], 'b', ls='--')
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






def Plotting_PS(k_array, z, physics, PS_Dict, GS, PS, save_loc=None,param=None,plot_expansion_beorn=False):
    """""""""""
    k_array,z : values for the power spectra plot
    physics,PS_Dict : GS history and PS dictionnary data from radtrans
    GS, PS : from halo model
    """""""""""
    ################################################ loading
    from matplotlib import pyplot as plt
    RT_zz = physics['z']  #
    Tcmb = (1 + RT_zz) * Tcmb0
    RT_dTb_GS = dTb_fct(physics['z'], physics['Tk'], physics['x_coll'] + physics['x_al'], 0, physics['x_HII'], param) #physics['dTb_GS']  # *(1-Tcmb/physics['Tk'])/(1-Tcmb/physics['Tk_neutral'])
    x_tot = physics['x_al'] + physics['x_coll']
    RT_dTb = physics[
        'dTb']  # * x_tot/(1 + x_tot) * (physics['x_al']+physics['x_coll']+1) / (physics['x_al']+physics['x_coll'])

    beta_a = Beta(RT_zz,physics,qty='lyal') # physics['xal_coda_style'] / x_tot / (1 + x_tot)  # physics['beta_a'] #
    beta_a_HM = GS['x_al'] / (GS['x_al'] + GS['x_coll']) / (1 + GS['x_al'] + GS['x_coll'])

    Tbg = Tcmb0 * (1 + 135) * (1 + RT_zz) ** 2 / (1 + 135) ** 2
    fr = 1  # (physics['Tk']-Tbg)/physics['Tk'] ### Halomodel primordial/heating decomp
    beta_T = Beta(RT_zz,physics,qty='Tk')  # physics['beta_T'] * fr  #

    beta_T_HM = (1 + GS['z']) * Tcmb0 / (GS['Tgas'] - (1 + GS['z']) * Tcmb0)
    beta_r = Beta(RT_zz,physics,qty='reio') # -physics['x_HII']/(1-physics['x_HII'])
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
        if plot_expansion_beorn:
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




def Plotting_PS_GS(k_array, z, physics, PS_Dict, GS, PS, save_loc=None,param=None,plot_expansion_beorn=False):
    """""""""""
    k_array,z : values for the power spectra plot
    physics,PS_Dict : GS history and PS dictionnary data from radtrans
    GS, PS : from halo model
    """""""""""
    ################################################ loading
    from matplotlib import pyplot as plt
    RT_zz = physics['z']  #
    Tcmb = (1 + RT_zz) * Tcmb0
    RT_dTb_GS = dTb_fct(physics['z'], physics['Tk'], physics['x_coll'] + physics['x_al'], 0, physics['x_HII'], param) #physics['dTb_GS']  # *(1-Tcmb/physics['Tk'])/(1-Tcmb/physics['Tk_neutral'])
    x_tot = physics['x_al'] + physics['x_coll']
    RT_dTb = physics[
        'dTb']  # * x_tot/(1 + x_tot) * (physics['x_al']+physics['x_coll']+1) / (physics['x_al']+physics['x_coll'])

    beta_a = Beta(RT_zz,physics,qty='lyal') # physics['xal_coda_style'] / x_tot / (1 + x_tot)  # physics['beta_a'] #
    beta_T = Beta(RT_zz,physics,qty='Tk')  # physics['beta_T'] * fr  #

    beta_r = Beta(RT_zz,physics,qty='reio') # -physics['x_HII']/(1-physics['x_HII'])
    RT_iz = np.argmin(np.abs(RT_zz - z))

    HM_zz, HM_dTb = GS['z'], GS['dTb']
    HM_iz = np.argmin(np.abs(PS['z'] - RT_zz[RT_iz]))

    fig = plt.figure(figsize=(15, 5))

    ################################################ dTb
    axis1 = fig.add_subplot(131)
    axis1.plot(HM_zz, HM_dTb, label='HM')
    axis1.plot(RT_zz, RT_dTb_GS, label='BEORN dTb_GS f(mean)', ls='--')
    axis1.plot(RT_zz, RT_dTb, label='BEORN mean(dTb) ', ls='--')
    # axis1.plot(RT_zz , RT_dTb,label='radtrans',ls='-.')
    axis1.set_xlim(6, 21)
    axis1.set_ylabel('dTb')
    axis1.legend()

    ################################################ xHII
    axis2 = fig.add_subplot(132)
    axis2.plot(HM_zz, 1 - GS['x_HI'])
    axis2.plot(RT_zz, physics['x_HII'], ls='--')
    axis2.set_xlim(6, 20)
    axis2.set_ylabel('xHII')
    axis2.set_xlabel('z')

    ################################################ PS(k)
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

    ################################################ PS(z)
    axis4 = fig.add_subplot(133)
    ax = plt.gca()
    for k0 in k_array:
        color = next(ax._get_lines.prop_cycler)['color']
        ind_k_RT = np.argmin(np.abs(PS_RT['k'] - k0))
        if plot_expansion_beorn:
            axis4.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * RT_dTb_GS ** 2 * PS_dTb_HMstyle[:, ind_k_RT] / 2 / np.pi ** 2,
                       color=color, alpha=0.5, ls='-.', lw=4)
        axis4.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * RT_dTb ** 2 * PS_dTb[:, ind_k_RT] / 2 / np.pi ** 2,
                       color=color, ls='--')

        ind_k = np.argmin(np.abs(PS['k'] - PS_RT['k'][ind_k_RT]))
        axis4.semilogy(PS['z'], PS['k'][ind_k] ** 3 * HM_dTb ** 2 * PS['P_mu0'][:, ind_k] / 2 / np.pi ** 2, color=color,
                       label='k={} h/Mpc'.format(k0))
        # axis4.semilogy(PS['z'],PS['k'][ind_k]**3 * HM_dTb**2 * PS['P_angav'][:,ind_k]/2/np.pi**2,ls=':',lw=4,alpha=0.7,color=color,label='k={}'.format(k0))
        # print(PS['P_angav'][:, ind_k] / PS['P_mu0'][:, ind_k])
        print('k RT is ', round(PS_RT['k'][ind_k_RT], 4), 'k HM is,', PS['k'][ind_k])

    # axis4.semilogy(PS['z'],PS['k'][ind_k]**3 * HM_dTb**2* (PS['P_aa'][:,ind_k]+2*(PS['P_ba'][:,ind_k]+PS['P_Ta'][:,ind_k]+PS['P_ra'][:,ind_k]))/2/np.pi**2,color=color,label='k={}'.format(k0))
    # axis4.semilogy(RT_zz, k0**3 * RT_dTb_GS**2 * (PS_aa[:,ind_k_RT]+2*(PS_aT[:,ind_k_RT]+PS_ab[:,ind_k_RT]+PS_ra[:,ind_k_RT]))/2/np.pi**2,color=color,alpha=0.5,ls='--',lw=4)
    axis4.semilogy([], [], color='gray', label='HM', alpha=0.5, ls='-')
    axis4.semilogy([], [], color='gray', label='PS(dTb) BEORN', ls='--')

    axis4.set_ylim(1e-1, 1e3)
    axis4.set_xlim(6, 20)

    axis4.legend(title='$ k^{3}P(k)/(2\pi^{2})$')  # dTb^{2}
    axis4.set_xlabel('z', fontsize=14)

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


def plot_HM_PS_of_z(k, PS, color, label='', ls='--',lw=2):
    """""""""
    Plot a HM PS as a fct of k
    PS : output of coda.halomodel

    """""""""
    kk, zz = PS['k'], PS['z']
    ind_k = np.argmin(np.abs(kk * 0.68 - k))
    print('k HM is', PS['k'][ind_k] * 0.68)
    try:
        plt.semilogy(zz, kk[ind_k] ** 3 * PS['dTb'] ** 2 * np.abs(PS['P_mu0'][:, ind_k]) / 2 / np.pi ** 2, ls=ls, lw=lw,
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
    plt.ylim(1e-1, 1e3)
    plt.xlim(6, 22)
    plt.legend(title='$ k^{3}P(k)/(2\pi^{2})$')
    plt.xlabel('z', fontsize=14)

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



def plot_reio_constraints():
    """
    Combined constrainted on averaged xHII
    TAKEN FROM  : --> The short ionizing photon mean free path at z=6 in Cosmic Dawn III,
                    a new fully-coupled radiation-hydrodynamical simulation of the Epoch of Reionization

    """
    # plt.errorbar([7.143946073101785], [0.591468163757487],yerr=[[0.591-0.4014],[0.7998-0.591]],markersize=6,marker = 'x')

    ## LyA E Fraction
    ## the point with yerr + and -
    plt.errorbar([6.980], [0.60060], yerr=[[0.6006 - 0.524], [0.69103 - 0.6006]], color='lightcoral', markersize=6,
                 marker='d', label='LAE fraction')
    plt.errorbar([7.037], [0.4027], yerr=[[0.4027 - 0.255], [0.51280 - 0.4027]], color='lightcoral', markersize=6,
                 marker='d')
    plt.errorbar([6.98420], [0.2428], yerr=[[0.2428 - 0.1956], [0.2939 - 0.2428]], color='lightcoral', markersize=6,
                 marker='d')
    ## one wiht only yerr -->
    plt.errorbar([6.954], [0.4787], yerr=[[0.4787 - 0.428], [0]], color='lightcoral', markersize=6, marker='d',
                 uplims=True, )
    plt.errorbar([7.9937], [0.7], yerr=[[0.688 - 0.633], [0]], color='lightcoral', markersize=6, marker='d',
                 uplims=True, )
    plt.errorbar([7.9787], [0.3439], yerr=[[0.3439 - 0.286283], [0]], color='lightcoral', markersize=6, marker='d',
                 uplims=True, )
    ## one with x and y err
    plt.errorbar([7.5916], [0.117122], yerr=[[0.117122 - 0.064], [0.21279 - 0.117122]],
                 xerr=([7.5916 - 6.99439], [8.188973 - 7.5916]), color='lightcoral', markersize=6, marker='d')
    ### QSO damping wings
    plt.errorbar([6.191], [0.888], yerr=[[0.888 - 0.833], [0]], color='darkseagreen', markersize=6, marker='s',
                 uplims=True, label='QSO damping wings')
    plt.errorbar([7.08696], [0.88765], yerr=[[0.887654 - 0.8352], [0]], color='darkseagreen', markersize=6, marker='s',
                 uplims=True, )
    plt.errorbar([7.07], [0.660914], yerr=[[0.660914 - 0.6045], [0]], color='darkseagreen', markersize=6, marker='s',
                 uplims=True, )
    plt.errorbar([7.0701], [0.74217], yerr=[[0.74217 - 0.68975033], [0.7893549 - 0.74217]], color='darkseagreen',
                 markersize=6, marker='s')
    plt.errorbar([6.965], [0.29129], yerr=[[0.29129 - .06456], [0.4892027 - 0.29129]], color='darkseagreen',
                 markersize=6, marker='s')
    plt.errorbar([7.53], [0.3936], yerr=[[0.3936 - 0.28225], [0.50243 - 0.3936]], color='darkseagreen', markersize=6,
                 marker='s')
    ## GRB Damping wings
    plt.plot([5.91], [0.94], color='cyan', markersize=6, marker='o', label='GRB Damping wings')
    ##Lyal LF
    # plt.errorbar([6.6030974], [0.7427821],yerr = [[0],[0.797900-0.7427821]],color='purple',markersize=6,marker='X',lolims=True,label='Lyα LF')
    #  plt.errorbar([6.6], [0.92],yerr = [[0.05],[0.08]],color='purple',markersize=6,marker='X',label='Lyα LF') #2101.01205v3
    # plt.errorbar([7], [0.72],yerr = [[0.5],[0.05]],color='purple',markersize=6,marker='X')
    # plt.errorbar([7.3], [0.17],yerr = [[0.07],[0.06]],color='purple',markersize=6,marker='X')

    # Dark Pixel Fraction
    plt.errorbar([5.61], [0.89107611], yerr=[[0], [0.944881 - 0.891076115]], color='y', markersize=6, marker='*',
                 lolims=True, label='Dark pixel fraction')
    plt.errorbar([5.917], [0.880577], yerr=[[0], [0.93569553 - 0.880577]], color='y', markersize=6, marker='*',
                 lolims=True, )
    # Lyal EW
    plt.errorbar([7.6], [0.51], yerr=[[0.19], [0.19]], color='thistle', markersize=6, marker='H', label='Lyα EW')
    # Lyal EW 2303.03419 Bruton et al
    plt.errorbar([10.6], [0.12], yerr=[[0], [0.1]], color='thistle', markersize=6, marker='H', lolims=True)


