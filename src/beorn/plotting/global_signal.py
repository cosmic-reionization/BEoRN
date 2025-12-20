import numpy as np
import matplotlib.pyplot as plt
import tools21cm as t2c

from ..structs import TemporalCube, Parameters


def draw_dTb_signal(ax: plt.Axes, grid: TemporalCube, label=None, color=None):
    """Plot the global mean differential brightness temperature dTb(z).

    Args:
        ax (matplotlib.axes.Axes): Axis to draw on.
        grid (TemporalCube): Temporal cube providing ``Grid_dTb`` and ``z`` arrays.
        label (str, optional): Legend label.
        color (str|tuple, optional): Line color.
    """
    z_range = grid.z[:]
    mean_dtb = np.mean(grid.Grid_dTb, axis=(1, 2, 3))
    ax.plot(z_range, mean_dtb, color=color, label=label)
    ax.set_xlim(z_range.min() - 0.2, z_range.max())
    ax.set_xlabel('z')
    ax.set_ylabel(r'$dT_b$ [mK]')


def draw_x_alpha_signal(ax: plt.Axes, grid: TemporalCube, label=None, color=None):
    """Plot the mean Lyman-alpha coupling history x_alpha(z).

    Args:
        ax (matplotlib.axes.Axes): Axis to draw on.
        grid (TemporalCube): Temporal cube providing ``Grid_xal`` and ``z`` arrays.
        label (str, optional): Legend label.
        color (str|tuple, optional): Line color.
    """
    z_range = grid.z[:]
    mean_x_alpha = np.mean(grid.Grid_xal, axis=(1, 2, 3))
    ax.semilogy(z_range, mean_x_alpha, color=color, label=label)
    ax.set_xlim(z_range.min() - 0.2, z_range.max())
    ax.set_xlabel('z')
    ax.set_ylabel(r'$x_\alpha$')


def draw_Temp_signal(ax: plt.Axes, grid: TemporalCube, label=None, color=None):
    """Plot the mean kinetic temperature history T_k(z).

    Args:
        ax (matplotlib.axes.Axes): Axis to draw on.
        grid (TemporalCube): Temporal cube providing ``Grid_Temp`` and ``z`` arrays.
        label (str, optional): Legend label.
        color (str|tuple, optional): Line color.
    """
    z_range = grid.z[:]
    mean_tk = np.mean(grid.Grid_Temp, axis=(1, 2, 3))
    ax.plot(z_range, mean_tk, color=color, label=label)
    ax.semilogy([], [])
    ax.set_xlim(z_range.min() - 0.2, z_range.max())
    ax.set_ylabel(r'$T_{k}$ [K]')
    ax.set_xlabel('z')


def draw_xHII_signal(ax: plt.Axes, grid: TemporalCube, label=None, color=None):
    """Plot the mean ionized fraction x_HII(z).

    Args:
        ax (matplotlib.axes.Axes): Axis to draw on.
        grid (TemporalCube): Temporal cube providing ``Grid_xHII`` and ``z`` arrays.
        label (str, optional): Legend label.
        color (str|tuple, optional): Line color.
    """
    z_range = grid.z[:]
    mean_x_HII = np.mean(grid.Grid_xHII, axis=(1, 2, 3))
    ax.plot(z_range, mean_x_HII, color=color, label=label)
    ax.set_xlim(z_range.min() - 0.2, z_range.max())
    ax.set_ylabel(r'$x_{\mathrm{HII}}$')
    ax.set_xlabel('z')


def draw_dTb_power_spectrum_of_z(ax: plt.Axes, grid: TemporalCube, parameters: Parameters, label=None, color=None, k_index=1):
    """Plot the evolution of the dTb power spectrum at a fixed k.

    Computes the power spectrum for each snapshot and plots the
    dimensionless power at the requested wavenumber index as a
    function of redshift.

    Args:
        ax (matplotlib.axes.Axes): Axis to draw on.
        grid (TemporalCube): Temporal cube providing ``Grid_dTb`` and ``z``.
        parameters (Parameters): Simulation parameters passed to the cube's power spectrum routine.
        label (str, optional): Legend label.
        color (str|tuple, optional): Line color.
        k_index (int, optional): Index of the k-bin to plot.
    """
    z_range = grid.z[:]
    mean_dtb = np.mean(grid.Grid_dTb, axis=(1, 2, 3))
    ps, bins = grid.power_spectrum(grid.Grid_dTb, parameters)
    k = bins[k_index]
    ps_k = ps[..., k_index]
    ps_c = ps_k * k ** 3 * mean_dtb ** 2 / (2 * np.pi ** 2)
    ax.semilogy(z_range, ps_c, label=label, color=color)
    ax.set_ylim(1e-1, 1e3)
    ax.set_ylabel(rf'$| dT_b |^2 \, \Delta_{{tot}}^{{2}} (k_0, z)$ [mK]$^2$')
    ax.set_xlabel('z')
    ax.set_xlim(z_range.min() - 0.2, z_range.max())
    # write the k value inside the plot
    ax.text(0.05, 0.05, f'k={k:.2f} Mpc$^{{-1}}$', transform=ax.transAxes)


def draw_dTb_power_spectrum_of_k(ax: plt.Axes, grid: TemporalCube, parameters: Parameters, z_index: int, label=None, color=None):
    """Plot the dTb power spectrum as a function of k at a given z.

    Args:
        ax (matplotlib.axes.Axes): Axis to draw on.
        grid (TemporalCube): Temporal cube providing ``Grid_dTb`` and ``z``.
        parameters (Parameters): Simulation parameters containing kbins and box size.
        z_index (int): Index of the redshift slice to analyse.
        label (str, optional): Legend label.
        color (str|tuple, optional): Line color.
    """
    z = grid.z[z_index]
    current_grid = grid.Grid_dTb[z_index, ...]
    # TODO - fail on nan values instead of ignoring them
    mean_dtb = np.mean(current_grid)

    delta_quantity = current_grid / mean_dtb - 1
    bin_number = parameters.simulation.kbins.size
    box_dims = parameters.simulation.Lbox

    # TODO - is this the correct quantity?
    ps, bins = t2c.power_spectrum.power_spectrum_1d(delta_quantity, box_dims=box_dims, kbins=bin_number)
    ps_c = ps * bins ** 3 * mean_dtb ** 2 / (2 * np.pi ** 2)

    ax.semilogy(bins, ps_c, ls='-', label=f"{label} (z={z:.2f})", color=color)
    ax.set_ylim(1e-1, 1e3)
    ax.set_ylabel(rf'$| dT_b |^2 \, \Delta_{{tot}}^{{2}} (k, z_0)$ [mK]$^2$')
    ax.set_xlabel('k [cMpc$^{-1}$]')



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
    # get or create the axes
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

        grid_value = np.mean(grid.Grid_xal, axis=(1, 2, 3))
        baseline_value = np.mean(baseline_grid.Grid_xal, axis=(1, 2, 3))
        deviation = (grid_value - baseline_value) / baseline_value
        axs[4].plot(grid.z[:], deviation, color=color, label=label)
        axs[4].set_xlabel('z')
        axs[4].set_ylabel(r'$\Delta x_\alpha$ / $x_\alpha$')

        grid_value = np.mean(grid.Grid_Temp, axis=(1, 2, 3))
        baseline_value = np.mean(baseline_grid.Grid_Temp, axis=(1, 2, 3))
        deviation = (grid_value - baseline_value) / baseline_value
        axs[5].plot(grid.z[:], deviation, color=color, label=label)
        axs[5].set_xlabel('z')
        axs[5].set_ylabel(r'$\Delta T_k$ / $T_k$')

        grid_value = np.mean(grid.Grid_xHII, axis=(1, 2, 3))
        baseline_value = np.mean(baseline_grid.Grid_xHII, axis=(1, 2, 3))
        deviation = (grid_value - baseline_value) / baseline_value
        axs[6].plot(grid.z[:], deviation, color=color, label=label)
        axs[6].set_xlabel('z')
        axs[6].set_ylabel(r'$\Delta x_{\mathrm{HII}}$ / $x_{\mathrm{HII}}$')

        grid_value = np.mean(grid.Grid_dTb, axis=(1, 2, 3))
        baseline_value = np.mean(baseline_grid.Grid_dTb, axis=(1, 2, 3))
        deviation = (grid_value - baseline_value) / baseline_value
        axs[7].plot(grid.z[:], deviation, color=color, label=label)
        axs[7].set_xlabel('z')
        axs[7].set_ylabel(r'$\Delta dT_b$ / $dT_b$')
