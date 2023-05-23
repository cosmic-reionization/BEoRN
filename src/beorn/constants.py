
sec_per_year = 3600*24*365.25
M_sun = 1.988 * 10 ** 30 # [kg]
m_H    = 1.6 * 10 ** - 27
km_per_Mpc = 3.08568 * 10 ** 19
cm_per_Mpc = 3.08568 * 10 ** 24
m_p_in_Msun= 8.4119e-58     # Proton mass in [Msun]
eV_per_erg = 6.242 * 10 ** 11   #(1 * u.erg).to(u.eV).value


Tcmb0 = 2.725   # [K]
Tstar = 0.068   # [K] (astro-ph/0608032 below Eq.11)
A10 = 2.85e-15  # [1/s] (astro-ph/0608032 below Eq.14)


nu21  = 1420.4                      # 21cm frequency [MHz]
nu_LL = 3.2898e15                   # Lyman-limit frequency [Hz]
nu_al = 2.4674e15                   # Lyman-alpha frequency [Hz]
nu_be = 2.9243e15                   # Lyman-beta frequency [Hz]


h__ = 6.626e-34         ## [J.s] or [m2 kg / s]
c__ = 2.99792e+8        ## [m/s]
k__ = 1.380649e-23      ## [m 2 kg s-2 K-1]
h_eV_sec = 4.135667e-15 ## [eV.s]
c_km_s   = 2.99792e5   # Speed of light [km/s]

# Hydro density and mass
n_H_0  = 1.87 * 10 ** -12        # [cm**-3]
m_H    = 1.6726219 * 10 ** - 27       # [kg]
m_He   = 6.6464731 * 10 ** - 27 # [kg]
rhoc0    = 2.775e11              # Critical density at z=0 [h^2 Msun/Mpc^3]


# Energy limits and ionization energies, in [eV]
E_0 = 10.4
E_HI = 13.6
E_HeI = 24.5
E_HeII = 54.42

# Constants
c = 2.99792 * 10 ** 10    # [cm/s]
kb = 1.380649 * 10 ** -23 # [J/K] or [kg.m2.s-2.K-1 ]
kb_eV_per_K = 8.61733e-5  # [eV/K]

m_e = 9.10938 * 10 ** -31 # [kg]
m_e_eV = 511e3            # [eV], 511keV

##Thomson scattering cross-section
sigma_s = 6.6524 * 10 ** -25 # [cm**2]
Hz_per_eV = 241799050402293

##Thomson scattering cross-section in meters
sigma_T = 6.6524 * 10 ** -29