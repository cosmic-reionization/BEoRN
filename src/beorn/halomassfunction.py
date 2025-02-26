
"""
Halo mass function from the Press-Schechter formalism
"""

import numpy as np
from math import pi

import scipy.differentiate
from .cosmo import *
import scipy
from beorn.profiles_on_grid import log_binning,bin_edges_log

class HMF:
    def __init__(self, param):
        data = np.loadtxt(param.cosmo.ps)
        self.k_lin = data[:, 0]
        self.P_lin = data[:, 1]
        self.tab_M = np.logspace(np.log10(param.hmf.m_min), np.log10(param.hmf.m_max), param.hmf.Mbin, base=10)   # [Msol/h]
        self.tab_R = ((3 * self.tab_M / (4 * rhoc0 * param.cosmo.Om * pi)) ** (1. / 3)) / param.hmf.c     # [Mpc/h]
        self.z = np.array(param.hmf.z)

    def sigma_square(self,param):
        if param.hmf.filter == 'tophat':
            self.sigma2 = np.trapz(self.k_lin ** 2 * self.P_lin * W_tophat(self.k_lin * self.tab_R[:,None]) ** 2 / (2 * pi ** 2) , self.k_lin, axis = 1)
            self.dlnsigmdlnR = np.trapz(self.k_lin ** 3 * self.tab_R[:,None] * self.P_lin * W_tophat(self.k_lin * self.tab_R[:, None]) * derivative_W_tophat(self.k_lin * self.tab_R[:, None])/(2 *self.sigma2[:,None] * pi ** 2), self.k_lin, axis=1)
        elif param.hmf.filter == 'sharpk':
            self.sigma2 = np.trapz(self.k_lin ** 2 * self.P_lin * W_sharpk(self.k_lin * self.tab_R[:, None]) ** 2 / (2 * pi ** 2), self.k_lin, axis=1)
            self.dlnsigmdlnR = np.interp(1/self.tab_R, self.k_lin, self.P_lin) /4 /np.pi**2 / self.tab_R**3 / self.sigma2
        elif param.hmf.filter == 'smoothk':
            self.sigma2 = np.trapz(self.k_lin ** 2 * self.P_lin * W_smooth_k(self.k_lin * self.tab_R[:, None],param) ** 2 / (2 * pi ** 2), self.k_lin, axis=1)
            self.dlnsigmdlnR = np.trapz(self.k_lin ** 3 * self.tab_R[:,None] * self.P_lin * W_smooth_k(self.k_lin * self.tab_R[:, None],param) * derivative_W_smooth(self.k_lin * self.tab_R[:, None],param)/(2 * self.sigma2[:,None] * pi ** 2), self.k_lin, axis=1)
        else :
            print('filter should be tophat, sharpk or smoothk')

    def generate_HMF(self,param):
        self.sigma_square(param)
        self.sigma_z = D(1. / (self.z  + 1), param)[:,None] * np.sqrt(self.sigma2)
        self.f_ST = crossing_f_ST(self.sigma_z, param)
        self.HMF = self.f_ST * rhoc0 * param.cosmo.Om * np.abs(self.dlnsigmdlnR) / 3 / self.tab_M




#define window function top hat filter (in Fourier space)
def W_tophat(x):
    return 3*(np.sin(x)-x*np.cos(x))/x**3

def derivative_W_tophat(x):
    return scipy.differentiate.derivative(W_tophat,x).df

#sharp-k filter
def W_sharpk(x):
    return np.heaviside(1-x,0)
def derivative_W_sharpk(x):
    return scipy.differentiate.derivative(W_sharp,x).df


#smooth-k filter
def W_smooth_k(x,param):
    return ((1+x**param.hmf.Beta)**-1)
def derivative_W_smooth(x,param):
    Beta = param.hmf.Beta
    return -(Beta*x**(Beta-1))/((1+x**Beta)**2)
def W_times_W_prime_smooth(x,param):
    Beta = param.hmf.Beta
    return Beta*x**(Beta-1)/((1+x**Beta)**3)




#define the crossing distribution function
def crossing_f_ST(sigm,param):
    """""
    First crossing distribution
    """""
    A = param.hmf.A
    q = param.hmf.q
    delta_c = param.hmf.delta_c
    p = param.hmf.p
    return A*np.sqrt(2*q*(delta_c**2/sigm**2)/np.pi)*(1+(sigm**2/(q*delta_c**2))**p)*(np.exp(-q*delta_c**2/(2*sigm**2)))


def from_catalog_to_hmf(dictionnary, Lbox=None, Mmax=None, Mmin=None,bin_nbr=None):
    """""
    Lbox : Box size (Mpc/h)
    Read HMF dictionnary and output binned masses, halo mass fct (Mpc/h)^-3,  and Poisson error in each bins.
    
    """""
    Mh = dictionnary['M']
    if Lbox is None:
        Lbox = dictionnary['Lbox']
    if Mmax is None and Mmin is None:
        Mmin, Mmax = np.min(Mh), np.max(Mh)

    if bin_nbr is None:
        bin_nbr = np.log10(Mmax / Mmin) * 10  # 10 bin per order of mag of Mh

    tab_M = np.logspace(np.log(Mmin - 1), np.log(Mmax + 1), int(bin_nbr), base=np.exp(1))
    dlnm = np.log(tab_M[1]) - np.log(tab_M[0])

    #indices = np.digitize(Mh, tab_M)
    indices = log_binning(Mh, bin_edges_log(tab_M))
    indices = indices - 1

    count = np.unique(indices, return_counts=True)
    hmf__ = np.zeros(len(tab_M))
    error = np.zeros(len(tab_M))
    hmf__[count[0]] = count[1] / Lbox ** 3 / dlnm
    error[count[0]] = hmf__[count[0]] / np.sqrt(count[1])

    print('redshift is', dictionnary['z'], 'Lbox is :', Lbox)

    return tab_M, hmf__, error
