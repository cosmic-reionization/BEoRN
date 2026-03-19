import numpy as np
import matplotlib.pyplot as plt

from ..structs import HaloCatalog


def plot_halo_mass_function(
    ax: plt.Axes,
    halo_catalog: HaloCatalog,
    bin_count: int = None,
    label: str = None,
    color: str = None,
    analytical: bool = True,
    analytical_model: str = 'Tinker08',
) -> None:
    """Plot the halo mass function from a :class:`HaloCatalog`.

    Optionally overlays an analytical reference curve computed with the
    ``hmf`` package (Tinker et al. 2008 by default).

    Args:
        ax (matplotlib.axes.Axes): Axis to draw the HMF on.
        halo_catalog (HaloCatalog): Halo catalog providing ``halo_mass_function``.
        bin_count (int, optional): Number of bins to use; if ``None`` the catalog default is used.
        label (str, optional): Legend label for the simulation data points.
        color (str, optional): Line/marker color for the simulation data points.
        analytical (bool): If ``True`` (default), overlay an analytical HMF curve.
            Requires the ``hmf`` package (``pip install hmf``).
        analytical_model (str): ``hmf`` model name passed to
            :class:`hmf.MassFunction`. Default is ``'Tinker08'``.
    """
    bin_edges, hmf_sim, hmf_err = halo_catalog.halo_mass_function(bin_count)
    bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    ax.errorbar(bin_centers, hmf_sim, yerr=hmf_err, fmt="*", label=label, color=color)
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Halo Mass $M$ [$M_\odot$]')
    ax.set_ylabel(r'$\frac{dn}{d\ln M}$ [$(\mathrm{Mpc}/h)^{-3}$]')

    if not analytical:
        return

    try:
        import hmf as hmf_pkg
    except ImportError:
        return

    cosmo = halo_catalog.parameters.cosmology
    z = halo_catalog.parameters.solver.redshifts[halo_catalog.redshift_index]
    mf = hmf_pkg.MassFunction(
        z = z,
        Mmin = np.log10(bin_edges[0]),
        Mmax = np.log10(bin_edges[-1]),
        hmf_model = analytical_model,
        cosmo_params = dict(Om0=cosmo.Om, Ob0=cosmo.Ob, H0=cosmo.h * 100),
    )
    ax.plot(mf.m, mf.dndlnm, ls='--', color='grey', label=f'{analytical_model} (analytical)')
