import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
from ..structs import Parameters
from ..astro import f_star_Halo, f_esc


def draw_star_formation_rate(ax: plt.Axes, parameters: Parameters, label=None, color=None):
    """
    TODO
    """
    range = [parameters.source.halo_mass_min, parameters.source.halo_mass_max]

    # restrict the simulated mass range
    bins = parameters.simulation.halo_mass_bins

    keep = (bins >= range[0]) & (bins <= range[1])
    keep_bins = bins[keep]

    fstar = f_star_Halo(parameters, keep_bins)


    ax.loglog(keep_bins, fstar, label=label, color=color)
    # ax.set_ylim(0.1, 1.1)
    ax.set_ylabel(r'$f_*$ = $\dot{M}_{*}/\dot{M}_{\mathrm{h}}$')
    ax.set_xlabel(r'M$_*$ $[M_{\odot}]$')



def draw_f_esc(ax: plt.Axes, parameters: Parameters, label=None, color=None):
    """
    TODO
    """
    range = [parameters.source.halo_mass_min, parameters.source.halo_mass_max]

    # restrict the simulated mass range
    bins = parameters.simulation.halo_mass_bins

    keep = (bins >= range[0]) & (bins <= range[1])
    keep_bins = bins[keep]

    fesc = f_esc(parameters, keep_bins)

    ax.loglog(keep_bins, fesc, label=label, color=color)
    # ax.set_ylim(0.1, 1.1)
    ax.set_ylabel(r'$f_{esc}(M_h)$')
    ax.set_xlabel(r'M$_h$ [$M_{\odot}$]')

