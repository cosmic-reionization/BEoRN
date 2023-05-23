"""
Here we write all the functions needed for the solver describing interactions, such as cross sections, cooling and collisional coupling coefficients etc...
"""

import numpy as np
from .constants import *





############ Used for Hydrogen
def sigma_HI(E):
    """
    Input : E is in eV.
    Returns : bound free photo-ionization cross section ,  [cm ** 2]
    """
    sigma_0 = 5.475 * 10 ** 4 * 10 ** -18 ## cm**2 1Mbarn = 1e-18 cm**2
    E_01 = 4.298 * 10 ** -1
    y_a = 3.288 * 10 ** 1
    P = 2.963
    y_w = y_0 = y_1 = 0
    x = E / E_01 - y_0
    y = np.sqrt(x ** 2 + y_1 ** 2)
    F = ((x - 1) ** 2 + y_w ** 2) * y ** (0.5 * P - 5.5) * (1 + np.sqrt(y / y_a)) ** -P
    sigma = sigma_0 * F
    return sigma


# Ionization and Recombination coefficients. Expressions taken from Fukugita and Kawasaki 1994.
def alpha_HII(T):
    """
    Case B recombination coefficient for Hydrogen :  [cm3.s-1]
    Input : temperature in K
    """
    return 2.6 * 10 ** -13 * (T / 10 ** 4) ** -0.85


def beta_HI(T):
    """
    Collisional ionization coefficient for Hydrogen :  [cm3.s-1]
    Input : temperature in K
    """
    return 5.85 * 10 ** -11 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * np.exp(-1.578 * 10 ** 5 / T)


def zeta_HI(T):
    """
    Collisional ionization cooling (see Fukugita & Kawazaki 1994) [eV.cm3.s-1]
    """
    return eV_per_erg * 1.27 * 10 ** -21 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * np.exp(-1.58 * 10 ** 5 / T)

def eta_HII(T):
    """
    Recombination cooling [eV.cm3.s-1].
    T ** 0.5 * (T / 10 ** 3) ** -0.2
    """
    return eV_per_erg * 6.5 * 10 ** -27 * T ** 0.3 * (1 / 10 ** 3) ** -0.2 * (1 + (T / 10 ** 6) ** 0.7) ** -1


def psi_HI(T):
    """
    Collisional excitation cooling [eV.cm3.s-1]
    """
    return eV_per_erg * 7.5 * 10 ** -19 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * np.exp(-1.18 * 10 ** 5 / T)


def theta_ff(T):
    """
    Free-free cooling coefficient [eV.cm3.s-1]
    """
    return eV_per_erg * 1.3 * 1.42 * 10 ** -27 * (T) ** 0.5




def f_H(x_ion):
    """
    Factor for secondary ionization, due to kinetic energy carried by secondary e- (see Shull & van Steenberg 1985). [Dimensionless]
    Input : x is the ionized fraction of hydrogen
    """
    return np.nan_to_num(0.3908 * (1 - np.maximum(np.minimum(x_ion,1),0) ** 0.4092) ** 1.7592)

def f_He(x_ion):
    return np.nan_to_num(0.0554 * (1 - np.maximum(np.minimum(x_ion,1),0) ** 0.4614) ** 1.6660)

def f_Heat(x_ion):
    """
    Amount of heat deposited by secondary electrons. (Shull & van Steenberg (1985) fig.3 - according to Thomas&Zaroubi Miniqso). [Dimensionless]
    """
    return np.maximum(0.9971 * (1 - (1 - np.maximum(np.minimum(x_ion, 1), 0) ** 0.2663) ** 1.3163), 0.11)  ## this 0.11 is a bit random, it should be 0.15 for xion<1e-4, but we do it like this to vectorize.


############ What follows is used when adding Helium to the calculations

def alpha_HeII(T):
    return 1.5 * 10 ** -10 * T ** -0.6353
def alpha_HeIII(T):
    return 3.36 * 10 ** -10 * T ** -0.5 * (T / 10 ** 3) ** -0.2 * (1 + (T / (4 * 10 ** 6)) ** 0.7) ** -1
def beta_HeI(T):
    return 2.38 * 10 ** -11 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * np.exp(-2.853 * 10 ** 5 / T)
def beta_HeII(T):
    return 5.68 * 10 ** -12 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * np.exp(-6.315 * 10 ** 5 / T)
def zeta_HeI(T):
    return eV_per_erg * 9.38 * 10 ** -22 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * np.exp(-2.85 * 10 ** 5 / T)

def zeta_HeII(T):
    return eV_per_erg * 4.95 * 10 ** -22 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * np.exp(-6.31 * 10 ** 5 / T)

def eta_HeII(T):
    return eV_per_erg * 1.55 * 10 ** -26 * T ** 0.3647
def eta_HeIII(T):
    return eV_per_erg * 3.48 * 10 ** -26 * T ** 0.5 * (T / 10 ** 3) ** -0.2 * (1 + (T / (4 * 10 ** 6)) ** 0.7) ** -1

def psi_HeI(T, ne, n_HeII):
    """
    Collisional excitation cooling coefficient for Hydrogen :  [eV.s-1]
    This is actually psi_HeI * nHeI . See B4.3(b) Fukugita
    Multiply this by ne and you get the correct term in heating eq.
    """
    return eV_per_erg * 9.1 * 10 ** -27 * T ** -0.1687 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * np.exp(-1.31 * 10 ** 4 / T) * ne * n_HeII

def psi_HeII(T):
    return eV_per_erg * 5.54 * 10 ** -17 * T ** -0.397 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * np.exp(-4.73 * 10 ** 5 / T)
def omega_HeII(T):
    return eV_per_erg * 1.24 * 10 ** -13 * T ** -1.5 * np.exp(-4.7 * 10 ** 5 / T) * (1 + 0.3 * np.exp(-9.4 * 10 ** 4 / T))

def sigma_HeI(E):
    sigma_0 = 9.492 * 10 ** 2 * 10 ** -18
    E_01 = 1.361 * 10 ** 1
    y_a = 1.469
    P = 3.188
    y_w = 2.039
    y_0 = 4.434 * 10 ** -1
    y_1 = 2.136
    x = E / E_01 - y_0
    y = np.sqrt(x ** 2 + y_1 ** 2)
    F = ((x - 1) ** 2 + y_w ** 2) * y ** (0.5 * P - 5.5) * (1 + np.sqrt(y / y_a)) ** -P
    sigma = sigma_0 * F
    return sigma

def sigma_HeII(E):
    sigma_0 = 1.369 * 10 ** 4 * 10 ** -18  # cm**2
    E_01 = 1.72
    y_a = 3.288 * 10 ** 1
    P = 2.963
    y_w = 0
    y_0 = 0
    y_1 = 0
    x = E / E_01 - y_0
    y = np.sqrt(x ** 2 + y_1 ** 2)
    F = ((x - 1) ** 2 + y_w ** 2) * y ** (0.5 * P - 5.5) * (1 + np.sqrt(y / y_a)) ** -P
    sigma = sigma_0 * F
    return sigma
