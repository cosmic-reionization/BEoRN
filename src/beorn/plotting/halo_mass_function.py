from ..structs import HaloCatalog
import numpy as np
import matplotlib.pyplot as plt

def plot_halo_mass_function(ax: plt.Axes, halo_catalog: HaloCatalog, bin_count: int = None, label: str = None, color: str = None) -> None:
    """
    Plots the halo mass function (HMF) from the given halo catalog on the provided Axes.
    """
    bin_edges, hmf, hmf_err = halo_catalog.halo_mass_function(bin_count)
    # the bin centers are the geometric mean of the edges
    bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    ax.errorbar(bin_centers, hmf, yerr=hmf_err, fmt="*", label=label, color=color)
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Halo Mass $M$ [$M_\odot$]')
    ax.set_ylabel(r'$\frac{dn}{d\ln M}$ [$(\mathrm{Mpc}/h)^{-3}$]')
