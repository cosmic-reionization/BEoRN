from scipy.optimize import curve_fit
import numpy as np
from .functions import *
from .constants import *
from .cosmo import *

def eps_compt(z, Tk, x_e):
    # eq 12 in meisinger 2011
    # eV/sec
    return 3 / 2 * (1 + x_e) * coef_compton(z, x_e) * kb_eV_per_K * (T_cmb(z) - Tk)

def eps_compt_Licorice(param,z,Tk,x_e):
    # eq 12 in meisinger 2011
    # K/sec
    n_atom = rhoc0*0.045*0.68**2/m_p_in_Msun/cm_per_Mpc**3 #cm-3
    return 1e-37* T_cmb(z)**4*(T_cmb(z)-Tk) * 2/3/kb * x_e * n_atom * param.source.coef_compton


def coef_compton(z, x_e):
    # sigma_T : Thomson cross section in m^2
    # M : Surface Power of CMB photons in eV.s^-1.m^-2
    # return a coef in s^-1

    M = sigm_steph_boltz * T_cmb(z) ** 4 / Joules_per_eV  # eV/s/m2
    m_e = 511e3  # eV, electron mass
    coef = x_e / (1 + f_He_nbr + x_e) * 8 / 3 * sigma_T * M / m_e
    return coef


from scipy.integrate import odeint

# let's solve the full global Tk eq

# Define the function to solve(f)

def compute_Tad_licorice(param,z_input):
    # Compute a correction factor (or output directly the adiab temperature)to account for COmpton heating of IGM gas by CMB photons
    # This is to compare to licorice.
    # This function is called in beorn/cosmo/T_adiab(z,param)
    # Highly approximative. Method : Solve global evolution of Tk, with adiabatic cooling
    # and with and without Compton heating
    # Take the difference. Add this to BEoRN temperature.

    def hubble_sec(z):
        return Hubble(z, param) / sec_per_year

    def rhs_heat_with_Compton(Tk, z):
        return 2 * Tk / (1 + z) - 2 / 3 * eps_compt(z, Tk, x_e=2e-4) / (1 + z) / hubble_sec(z) / kb_eV_per_K  # Replace with your expression

    def rhs_heat_w_o_Compton(Tk, z):
        return 2 * Tk / (1 + z)  # Replace with your expression

    def rhs_w_Compton_Licorice(Tk, z):
        #  compton heting as done in Licorice. Romain gave me the expression.
        return 2 * Tk / (1 + z) - eps_compt_Licorice(param,z, Tk, x_e=2e-4) / (1 + z) / hubble_sec(z)

    # Initial condition
    zi = 135
    T0 = Tcmb0 * (1 + zi) ** 2 / (1 + param.cosmo.z_decoupl)  # Replace with your initial value

    # Array of z
    zz = np.linspace(zi, 6, 100)

    # Solve the ODE
    #solution_with_Compton = odeint(rhs_heat_with_Compton, T0, zz)[:, 0]
    #solution_without_Compton = odeint(rhs_heat_w_o_Compton, T0, zz)[:, 0]
    solution_with_Compton = odeint(rhs_w_Compton_Licorice, T0, zz)[:, 0]
    #return np.interp(z_input, np.flip(zz), np.flip(solution_with_Compton - solution_without_Compton)) # if you just want the boost
    return np.interp(z_input, np.flip(zz), np.flip(solution_with_Compton))

















###### Relative to reading and writing cubes
def X_Y_Z_flat_cube(nGrid):
    depth = nGrid  # Number of 2D arrays along the first axis

    X_cube = np.arange(depth).reshape((depth, 1, 1)) * np.ones((nGrid, nGrid))
    Y_cube = np.arange(depth).reshape((1, depth, 1)) * np.ones((nGrid, 1, nGrid))
    Z_cube = np.arange(depth).reshape((1, 1, depth)) * np.ones((nGrid, nGrid, 1))

    X_cube = X_cube.T.reshape(int(nGrid ** 3))
    Y_cube = Y_cube.T.reshape(int(nGrid ** 3))
    Z_cube = Z_cube.T.reshape(int(nGrid ** 3))

    return X_cube, Y_cube, Z_cube


def read_cube(path, type=np.float):
    try:
        cube = np.fromfile(path, dtype=type)
        print('reading cube of shape :', cube.shape)
        if np.shape(cube)[0] > 0:
            print("moving on...")
            shape = np.shape(cube)[0]
            length = int(shape ** (1 / 3)) + 1

            cube = np.reshape(cube, (length, length, length)).T
            shape = np.shape(cube)
        else:
            cube = np.zeros((256, 256, 256))

    except FileNotFoundError:
        print(" !!!!!! file not found : " + path)
        cube = 'file not found'

    return cube


def read_Mstar(path,Vcell):
    h0 = 0.647
    M_unit = 2.19*1e9
    rho_star = read_cube(path, type = np.float)*M_unit/h0**2 * 1e9  ## (Msol/h)/(Mpc/h)**3
    Mstar = rho_star * Vcell ## (Msol/h)
    return Mstar

def str_snap(snap):
    if snap<10:
        str_sn = '00'+str(snap)
    else :
        str_sn = '0'+str(snap)
    return str_sn


def param_adapted_to_licorice():
    Om = 0.315
    h = 0.647
    Ob = 0.0492
    import beorn
    param = beorn.par()
    # Halo Mass bins
    param.sim.Mh_bin_min = 1e-3
    param.sim.Mh_bin_max = 1e11
    param.sim.binn = 30  # nbr of halo mass bin
    param.sim.average_profiles_in_bin = False

    # name your simulation
    param.sim.model_name = 'test_simuXXX'
    param.source.alpha_MAR = 0.79

    # Nbr of cores to use
    param.sim.cores = 1
    # simulation redshifts
    # np.concatenate((z_liste[5:25][::3],np.array([7.79,6.07])))#z_liste[5:25][//2]

    # cosmo
    param.cosmo.Om = Om
    param.cosmo.Ob = Ob
    param.cosmo.Ol = 1 - Ob - Om
    param.cosmo.h = h


    # Source parameters
    # lyman-alpha
    param.source.N_al = 7400   # 9690 #1500
    param.source.alS_lyal = 0.0

    # ion
    param.source.Nion = 14700
    # xray
    param.source.E_min_xray = 100
    param.source.E_max_xray = 2000
    param.source.E_min_sed_xray = 100
    param.source.E_max_sed_xray = 2000
    param.source.alS_xray = 1.6
    param.source.cX = 3.4e40

    param.source.xray_type = 'Licorice'
    param.source.sed_XRB = './simu12577/XRB_sed_Fragos.txt'
    # param.source.z_thresh_f_esc = 6.4
    param.source.f_esc_type = 'Licorice_bis'
    param.source.min_xHII = 1.5e-4
    param.source.fX_AGN = 1
    param.sim.licorice = True

    # fesc
    param.source.f0_esc = 0.275
    param.source.pl_esc = 0

    # fstar
    param.source.f_st = 1
    param.source.g1 = 0
    param.source.g2 = 0
    param.source.g3 = 4
    param.source.g4 = -4
    param.source.Mp = 1.6e11 * param.cosmo.h
    param.source.Mt = 1e-10

    param.cosmo.z_decoupl = 131

    # Minimum star forming halo
    param.source.M_min = 1e-10  # 1e8
    # Mass Accretion Rate model (EXP or EPS)
    param.source.MAR = 'EXP'

    param.cosmo.clumping = 1
    # fXh
    param.solver.fXh = 'Schull'

    param.sim.halo_catalogs = './NEW_Licorice_Halos/halo_dict_z'  ## path to dir with halo catalogs + filename

    return param



