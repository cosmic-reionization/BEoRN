import numpy as np
import matplotlib.pyplot as plt
import tools21cm as t2c

from ..structs import TemporalCube, Parameters
from ..structs.statistics import StatisticsEstimator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean(grid_or_stats, field: str, parameters: Parameters | None = None):
    """Return ``(z, values)`` for *field* from either a TemporalCube or a StatisticsEstimator."""
    if isinstance(grid_or_stats, StatisticsEstimator):
        r = grid_or_stats.results
        return r['z'], r[f'mean_{field}']
    grid = grid_or_stats
    return grid.z[:], grid.global_mean(field)


def _ps_dTb(grid_or_stats, parameters: Parameters | None, k_value=None, k_index=1):
    """Return ``(z, k_bins, ps_matrix, mean_dTb)`` for the dTb power spectrum."""
    if isinstance(grid_or_stats, StatisticsEstimator):
        r = grid_or_stats.results
        z = r['z']
        k_bins = r['k']
        # ps_dTb stored as Delta^2 already
        ps_c = r['ps_dTb']
        mean_dTb = r['mean_dTb']
        return z, k_bins, ps_c, mean_dTb
    grid = grid_or_stats
    z = grid.z[:]
    mean_dTb = grid.global_mean('Grid_dTb')
    ps, k_bins = grid.power_spectrum(grid.Grid_dTb, parameters)
    ps_c = ps * k_bins[np.newaxis, :] ** 3 * mean_dTb[:, np.newaxis] ** 2 / (2 * np.pi ** 2)
    return z, k_bins, ps_c, mean_dTb


# ---------------------------------------------------------------------------
# Public draw functions
# ---------------------------------------------------------------------------

def draw_dTb_signal(ax: plt.Axes, grid, label=None, color=None, **kwargs):
    """Plot the global mean differential brightness temperature dTb(z).

    Args:
        ax (matplotlib.axes.Axes): Axis to draw on.
        grid (TemporalCube | StatisticsEstimator): Data source.
        label (str, optional): Legend label.
        color (str|tuple, optional): Line color.
        **kwargs: Additional keyword arguments forwarded to ``ax.plot``.
    """
    z_range, mean_dtb = _mean(grid, 'dTb')
    ax.plot(z_range, mean_dtb, color=color, label=label, **kwargs)
    ax.set_xlim(z_range.min() - 0.2, z_range.max())
    ax.set_xlabel('z')
    ax.set_ylabel(r'$dT_b$ [mK]')


def draw_x_alpha_signal(ax: plt.Axes, grid, label=None, color=None, **kwargs):
    """Plot the mean Lyman-alpha coupling history x_alpha(z).

    Args:
        ax (matplotlib.axes.Axes): Axis to draw on.
        grid (TemporalCube | StatisticsEstimator): Data source.
        label (str, optional): Legend label.
        color (str|tuple, optional): Line color.
        **kwargs: Additional keyword arguments forwarded to ``ax.semilogy``.
    """
    z_range, mean_x_alpha = _mean(grid, 'xal')
    ax.semilogy(z_range, mean_x_alpha, color=color, label=label, **kwargs)
    ax.set_xlim(z_range.min() - 0.2, z_range.max())
    ax.set_xlabel('z')
    ax.set_ylabel(r'$x_\alpha$')


def draw_Temp_signal(ax: plt.Axes, grid, label=None, color=None, **kwargs):
    """Plot the mean kinetic temperature history T_k(z).

    Args:
        ax (matplotlib.axes.Axes): Axis to draw on.
        grid (TemporalCube | StatisticsEstimator): Data source.
        label (str, optional): Legend label.
        color (str|tuple, optional): Line color.
        **kwargs: Additional keyword arguments forwarded to ``ax.semilogy``.
    """
    z_range, mean_tk = _mean(grid, 'Temp')
    ax.semilogy(z_range, mean_tk, color=color, label=label, **kwargs)
    ax.set_xlim(z_range.min() - 0.2, z_range.max())
    ax.set_ylabel(r'$T_{k}$ [K]')
    ax.set_xlabel('z')


def draw_xHII_signal(ax: plt.Axes, grid, label=None, color=None, **kwargs):
    """Plot the mean ionized fraction x_HII(z).

    Args:
        ax (matplotlib.axes.Axes): Axis to draw on.
        grid (TemporalCube | StatisticsEstimator): Data source.
        label (str, optional): Legend label.
        color (str|tuple, optional): Line color.
        **kwargs: Additional keyword arguments forwarded to ``ax.plot``.
    """
    z_range, mean_x_HII = _mean(grid, 'xHII')
    ax.plot(z_range, mean_x_HII, color=color, label=label, **kwargs)
    ax.set_xlim(z_range.min() - 0.2, z_range.max())
    ax.set_ylabel(r'$x_{\mathrm{HII}}$')
    ax.set_xlabel('z')


def draw_dTb_power_spectrum_of_z(ax: plt.Axes, grid, parameters: Parameters = None, label=None, color=None, k_index=1, k_value=None, **kwargs):
    """Plot the evolution of the dTb power spectrum at a fixed k.

    Computes the power spectrum for each snapshot and plots the
    dimensionless power at the requested wavenumber index as a
    function of redshift.

    Args:
        ax (matplotlib.axes.Axes): Axis to draw on.
        grid (TemporalCube | StatisticsEstimator): Data source.
        parameters (Parameters, optional): Required when *grid* is a
            :class:`~beorn.structs.TemporalCube`; not needed for
            :class:`~beorn.structs.statistics.StatisticsEstimator`.
        label (str, optional): Legend label.
        color (str|tuple, optional): Line color.
        k_index (int, optional): Index of the k-bin to plot (ignored when *k_value* is set).
        k_value (float, optional): Target k value in Mpc^-1; nearest bin is used.
        **kwargs: Additional keyword arguments forwarded to ``ax.semilogy``.

    Returns:
        float: The k value actually plotted.
    """
    z_range, k_bins, ps_c, mean_dTb = _ps_dTb(grid, parameters, k_value=k_value, k_index=k_index)
    k = k_bins[k_index] if k_value is None else k_bins[np.abs(k_bins - k_value).argmin()]
    ki = np.abs(k_bins - k).argmin()
    ax.semilogy(z_range, ps_c[:, ki], label=label, color=color, **kwargs)
    ax.set_ylim(1e-1, 1e3)
    ax.set_ylabel(r'$\Delta_\mathrm{21}^{{2}}$ [mK]$^2$')
    ax.set_xlabel('z')
    ax.set_xlim(z_range.min() - 0.2, z_range.max())
    print(f'k={k:.2f} Mpc$^{{-1}}$')
    return k


def draw_dTb_power_spectrum_of_k(ax: plt.Axes, grid: TemporalCube, parameters: Parameters, z_index=1, z_value=None, label=None, color=None):
    """Plot the dTb power spectrum as a function of k at a given z.

    Args:
        ax (matplotlib.axes.Axes): Axis to draw on.
        grid (TemporalCube): Temporal cube providing ``Grid_dTb`` and ``z``.
        parameters (Parameters): Simulation parameters containing kbins and box size.
        z_index (int): Index of the redshift slice to analyse.
        label (str, optional): Legend label.
        color (str, optional): Color used for plotted lines.
    """
    z = grid.z[z_index] if z_value is None else grid.z[np.abs(grid.z-z_value).argmin()]
    current_grid = grid.Grid_dTb[z_index, ...]
    mean_dtb = np.mean(current_grid)

    delta_quantity = current_grid / mean_dtb - 1
    bin_number = parameters.simulation.kbins.size
    # Power spectrum itself is always computed in h-units (matches every other
    # internal consumer of Lbox_hunits); only the displayed bins/label below
    # follow `use_hunits` — see issue #49.
    box_dims = parameters.Lbox_hunits

    ps, bins = t2c.power_spectrum.power_spectrum_1d(delta_quantity, box_dims=box_dims, kbins=bin_number)
    ps_c = ps * bins ** 3 * mean_dtb ** 2 / (2 * np.pi ** 2)

    k_unit = 'h/Mpc' if parameters.simulation.use_hunits else 'Mpc$^{-1}$'
    bins_display = bins if parameters.simulation.use_hunits else bins * parameters.cosmology.h0

    ax.semilogy(bins_display, ps_c, ls='-', label=f"{label} (z={z:.2f})", color=color)
    ax.set_ylim(1e-1, 1e3)
    ax.set_ylabel(r'$\Delta_\mathrm{21}^{{2}}$ [mK]$^2$')
    ax.set_xlabel(f'k [{k_unit}]')
    print(f'z={z:.2f}')
    return z


def full_diff_plot(fig: plt.Figure, grid: TemporalCube, baseline_grid: TemporalCube = None, label: str = None, color: str = None):
    """Create a multi-panel comparison plot of global quantities.

    If ``baseline_grid`` is provided the routine also plots fractional
    deviations between ``grid`` and ``baseline_grid`` for the set of
    global quantities.

    Args:
        fig (matplotlib.figure.Figure): Figure to draw on; axes will be created if the figure is empty.
        grid (TemporalCube): Primary temporal cube to visualise.
        baseline_grid (TemporalCube, optional): Reference temporal cube for fractional deviation plots.
        label (str, optional): Label applied to plotted lines.
        color (str, optional): Color used for plotted lines.
    """
    if fig.axes:
        axs = fig.axes
    else:
        axs = fig.subplots(2, 4, sharex=True)
        axs = axs.flatten()

    draw_x_alpha_signal(axs[0], grid, label=label, color=color)
    draw_Temp_signal(axs[1], grid, label=label, color=color)
    draw_xHII_signal(axs[2], grid, label=label, color=color)
    draw_dTb_signal(axs[3], grid, label=label, color=color)

    if baseline_grid is not None:
        if grid == baseline_grid:
            print("Not comparing baseline grid to itself.")
            return

        for ax, field, ylabel in [
            (axs[4], 'Grid_xal',  r'$\Delta x_\alpha$ / $x_\alpha$'),
            (axs[5], 'Grid_Temp', r'$\Delta T_k$ / $T_k$'),
            (axs[6], 'Grid_xHII', r'$\Delta x_{\mathrm{HII}}$ / $x_{\mathrm{HII}}$'),
            (axs[7], 'Grid_dTb',  r'$\Delta dT_b$ / $dT_b$'),
        ]:
            grid_value = grid.global_mean(field)
            baseline_value = baseline_grid.global_mean(field)
            deviation = (grid_value - baseline_value) / baseline_value
            ax.plot(grid.z[:], deviation, color=color, label=label)
            ax.set_xlabel('z')
            ax.set_ylabel(ylabel)
