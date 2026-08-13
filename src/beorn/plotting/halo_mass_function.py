import numpy as np
import matplotlib.pyplot as plt

from ..structs import HaloCatalog
from ..mass_function import HaloMassFunction


def plot_halo_mass_function(
    ax: plt.Axes,
    halo_catalog: HaloCatalog,
    bin_count: int = None,
    label: str = None,
    color: str = None,
    analytical: bool = True,
    analytical_model: str = 'ST',
) -> None:
    """Plot the halo mass function from a :class:`HaloCatalog`.

    Optionally overlays an analytical reference curve computed with BEoRN's
    own :class:`~beorn.mass_function.HaloMassFunction` — no external
    dependency required.

    Args:
        ax (matplotlib.axes.Axes): Axis to draw the HMF on.
        halo_catalog (HaloCatalog): Halo catalog providing ``halo_mass_function``.
        bin_count (int, optional): Number of bins to use; if ``None`` the catalog default is used.
        label (str, optional): Legend label for the simulation data points.
        color (str, optional): Line/marker color for the simulation data points.
        analytical (bool): If ``True`` (default), overlay an analytical HMF curve.
        analytical_model (str): Model passed to
            :class:`~beorn.mass_function.HaloMassFunction` (case-insensitive).
            One of ``'ST'``/``'sheth_tormen'`` (default, matching
            :attr:`~beorn.structs.HaloSimParameters.hmf_model`'s own default),
            ``'PS'``/``'press_schechter'``, or ``'ellipsoidal'``.
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

    z = halo_catalog.redshift
    if z is None:
        raise ValueError(
            "HaloCatalog.redshift is not set — cannot compute the analytical HMF. "
            "Pass analytical=False or use a loader that sets HaloCatalog.redshift."
        )
    hmf = HaloMassFunction(
        halo_catalog.parameters,
        model=analytical_model.lower(),
        delta_c=halo_catalog.parameters.halo_sim.delta_c,
    )
    M_grid = np.logspace(np.log10(bin_edges[0]), np.log10(bin_edges[-1]), 200)
    n = hmf.dndlnm(M_grid, z)
    ax.plot(M_grid, n, ls='--', color='grey', label=f'{analytical_model} (analytical)')
