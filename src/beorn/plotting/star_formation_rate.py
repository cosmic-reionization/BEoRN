"""Plotting routines for star-formation rate related quantities."""
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
from ..structs import Parameters
from ..astro import f_star_Halo, f_esc


def draw_star_formation_rate(ax: plt.Axes, parameters: Parameters, label=None, color=None):
    """Draw the star-formation efficiency as a function of halo mass.

    The function queries the halo-mass grid from ``parameters`` and
    computes the star-formation fraction ``f_*`` using
    :func:`beorn.astro.f_star_Halo`, restricting the plotted range to
    the configured source mass limits.

    Args:
        ax (matplotlib.axes.Axes): Axis to draw the curve onto.
        parameters (Parameters): Simulation parameters used to obtain the halo mass grid and source mass limits.
        label (str, optional): Optional legend label.
        color (str|tuple, optional): Color for the plotted line.
    """
    mass_range = [parameters.source.halo_mass_min, parameters.source.halo_mass_max]

    # restrict the simulated mass range
    bins = parameters.simulation.halo_mass_bins

    keep = (bins >mass_= mass_range[0]) & (bins <= mass_range[1])
    keep_bins = bins[keep]

    fstar = f_star_Halo(parameters, keep_bins)


    ax.loglog(keep_bins, fstar, label=label, color=color)
    # ax.set_ylim(0.1, 1.1)
    ax.set_ylabel(r'$f_*$ = $\dot{M}_{*}/\dot{M}_{\mathrm{h}}$')
    ax.set_xlabel(r'M$_*$ $[M_{\odot}]$')



def draw_f_esc(ax: plt.Axes, parameters: Parameters, label=None, color=None):
    """Draw the escape fraction as a function of halo mass.

    The escape fraction ``f_esc`` is computed by
    :func:`beorn.astro.f_esc` on the simulation halo-mass grid and the
    result is plotted over the configured source mass range.

    Args:
        ax (matplotlib.axes.Axes): Axis to draw the curve onto.
        parameters (Parameters): Simulation parameters used to obtain the halo mass grid and source mass limits.
        label (str, optional): Optional legend label.
        color (str|tuple, optional): Color for the plotted line.
    """
    mass_range = [parameters.source.halo_mass_min, parameters.source.halo_mass_max]

    # restrict the simulated mass range
    bins = parameters.simulation.halo_mass_bins

    keep = (bins >= mass_range[0]) & (bins <= mass_range[1])
    keep_bins = bins[keep]

    fesc = f_esc(parameters, keep_bins)

    ax.loglog(keep_bins, fesc, label=label, color=color)
    # ax.set_ylim(0.1, 1.1)
    ax.set_ylabel(r'$f_{esc}(M_h)$')
    ax.set_xlabel(r'M$_h$ [$M_{\odot}$]')
