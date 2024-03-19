"""
External Parameters
"""
import pkg_resources

class Bunch(object):
    """
    translates dic['name'] into dic.name 
    """

    def __init__(self, data):
        self.__dict__.update(data)


def source_par():
    par = {
        "alpha_MAR" : 0.79,              # coefficient for exponential MAR
        "MAR" : 'EXP',                   # MAR model. Can be EXP or EPS.
        "type": 'SED',                   # str. source type. SED, Ghara, Ross, constant

        "E_min_sed_xray": 500,           # minimum energy of normalization of xrays in eV
        "E_max_sed_xray": 2000,          # minimum energy of normalization of xrays in eV

        "E_min_xray": 500,
        "E_max_xray": 2000,              # energy cutoff for the xray band.
        "alS_xray": 2.5,                 ##PL sed Xray part N ~ nu**-alS [nbr of photons/s/Hz]
        "cX":  3.4e40,                   # Xray normalization [(erg/s) * (yr/Msun)] (astro-ph/0607234 eq22)



        ### To compare to Licorice
        "xray_type": "PL",  # Can be PL (power-law) or Licorice
        "fX_AGN": 0.4 * 0.1 ,
        "fX_XRB": 0.6 * 0.1  ,
        "sed_XRB":None    ,    ## 2d array containing energy and  sed normalized to 1 (eV, sed)
        "f_esc_type": 'cst' , ## cst : redshift indep. Licorice : 2 valeurs en fonction du z.
        "z_thresh_f_esc":None, ## redshif at which the escape fraction changes value.
        "min_xHII":0, ## set all pixels where xHII=0 to this value


        "E_min_sed_xray": 500,  # minimum energy of normalization of xrays in eV
        "E_max_sed_xray": 2000,  # minimum energy of normalization of xrays in eV

        "N_al"    : 9690,                # nbr of lyal photons per baryons in stars
        "alS_lyal": 1.001,               ## PL for lyal

        "M_min" : 1e5,                   # Minimum mass of star forming halo. Mdark in HM
        "M_max" : 1e16,                   # Maximum mass of star forming halo
        'f_st': 0.05,
        'Mp': 1e11,
        'g1': 0.49,
        'g2': -0.61,
        'Mt': 1e7,
        'g3': 4,
        'g4': -1,

        'Nion'  : 2665,
        "f0_esc": 0.15,                   # photon escape fraction f_esc = f0_esc * (M/Mp)^pl_esc
        "Mp_esc": 1e10,
        "pl_esc": 0.0,
    }

    return Bunch(par)






def solver_par():
    par = {
        "z_max" : 40,                ## Starting redshift
        "z_min" : 6,             ## Only for MAR. Redshift where to stop the solver
        "Nz": 500,               ## Array or path to a text file.
        "fXh": 'constant',       ## if fXh is constant here, it will take the value 0.11. Otherwise, we will compute the free e- fraction in neutral medium and take the fit fXh = xe**0.225
    }
    return Bunch(par)

def sim_par(): ## used when computing and painting profiles on a grid
    par = {
        "Mh_bin_min" : 1e5,
        "Mh_bin_max" : 1e14,
        "binn" : 12,                # to define the initial halo mass at z_ini = solver.z
        "average_profiles_in_bin" : True,  # inside one mass bin, halos are un-evenly distributed. The profile corresponding to the mass bin should hence be a weidghted average...
        "HR_binning":400 ,          # finer mass binning used to compute accurate average profile in coarse binning.
        "model_name": 'SED',        # Give a name to your sim, will be used to name all the files created.
        "Ncell" : 128,              # nbr of pixels of the final grid.
        "Lbox" : 100,               # Box lenght, in [Mpc/h]
        "halo_catalogs": None,      # path to the directory containing all the halo catalogs.
        "store_grids": True,        # whether or not to store the grids. If not, will just store the power spectra.
        "dens_field": None,         # path and name of the input density field on a grid. Used in run.py to compute dTb maps.
        "dens_field_type": 'pkdgrav',  # Can be either 21cmFAST of pkdgrav. It adapts the format and normalization of the density field...
        "Nh_part_min":50,           # Halo with less than Mh_part_min are excluded.
        "cores" : 2,                # number of cores used in parallelisation
        "kmin": 3e-2,
        "kmax": 4,
        "kbin": 30,                ## either a path to a text files containing kbins edges values or an int (nbr of bins to measure PS)
        "thresh_pixel" : None,      ## when spreading the excess ionisation fraction, we treat all the connected regions with less that "thresh_pixel" as a single connected region(to speed up)
        "approx" : True,            ## when spreading the excess ionisation fraction and running distance_tranform_edt, whether or not to do the subgrid approx.
        "nGrid_min_heat": 4,             ## stacked_T_kernel
        "nGrid_min_lyal": 16,            ## stacked_lyal_kernel
        "random_seed": 12345,            ## when using 21cmFAST 2LPT solver.

        "T_saturated":False, ## If True, we will assum Tk>>Tcmb
        "reio": True,        ## If False, we will assume xHII = 0
    }
    return Bunch(par)


def cosmo_par():
    par = {
    'Om' : 0.31,
    'Ob' : 0.045,
    'Ol' : 0.68,
    'rho_c' : 2.775e11,
    'h' : 0.68,
    's8': 0.83,
    'ns': 0.96,
    'ps': pkg_resources.resource_filename('beorn', "files/PCDM_Planck.dat"),      ### This is the path to the input Linear Power Spectrum
    'corr_fct' : pkg_resources.resource_filename('beorn', "files/corr_fct.dat"),  ### This is the path where the corresponding correlation function will be stored. You can change it to anything.
    'HI_frac' : 1-0.08,       # HI number fraction fraction of HI. Only used when running H_He_Final. 1-fraction is Helium then.  0.2453 of total mass is in He according to BBN, so in terms of number density it is  1/(1+4*(1-f_He_bymass)/f_He_bymass)  ~0.075.
    "clumping" : 1,         # to rescale the background density. set to 1 to get the normal 2h profile term.
    "z_decoupl" : 135,      # redshift at which the gas decouples from CMB and starts cooling adiabatically according to Tcmb0*(1+z)**2/(1+zdecoupl)
    }
    return Bunch(par)



def excursion_set_par():
    par = {
        ### SemiNumerical Parameters
        "R_max": 40,                  # Mpc/h. The scale at which we start the excursion set.
        "n_rec": 3,                   # mean number of recombination per baryon.
        "stepping":1.1,               # When doing the exc set, we smooth the field over varying scales. We loop and increase this scale logarithmically (R=1.1*R)
     }
    return Bunch(par)


def hmf_par(): ## Parameters related to analytical halo mass function (PS formalism. Used to compute variance in EPS_MAR, and for subhalo MF in excursion set).
    par = {
        ### HMF parameters that we use to normalise the collapsed fraction.
        "filter": 'tophat',  # tophat, sharpk or smoothk
        "c": 1,              # scale to halo mass relation (1 for tophat, 2.5 for sharp-k, 3 for smooth-k)
        "q": 0.85,           # q for f(nu) [0.707,1,1] for [ST,smoothk or sharpk,PS] (q = 0.8 with tophat fits better the high redshift z>6 HMF)
        "p": 0.3,            # p for f(nu) [0.3,0.3,0] for [ST,smoothk or sharpk,PS]
        "delta_c": 1.686,    # critical density
        "A": 0.322,          # A = 0.322 except 0.5 for PS Spherical collapse (to double check)
        "m_min": 1e4,
        "m_max": 1e16,
        "Mbin": 300,
        'z': [0],            # output z values. Should be a list.
        }
    return Bunch(par)



def par():
    par = Bunch({
        "source": source_par(),
        "solver": solver_par(),
        "cosmo" : cosmo_par(),
        "sim" : sim_par(),
        "exc_set" : excursion_set_par(),
        "hmf" : hmf_par(),
        })
    return par
