import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LogNorm, Normalize
import numpy as np
import logging
logger = logging.getLogger(__name__)
from ..structs.lightcone import Lightcone

# Edit this gradient at https://eltos.github.io/gradient/#0:78E4FF-20:006DC2-49:001250-50:000000-51:562500-71.9:CF8400-100:FFEC33
#Good sharp gradient for dTb : https://eltos.github.io/gradient/#0:78E4FF-20:006DC2-49:001250-50:000000-51:562500-71.9:CF5D00-100:FFEC33
# is this an alternative?
COLOR_GRADIENT = LinearSegmentedColormap.from_list(
    'my_gradient',
    (
        (0.000, (0.471, 0.894, 1.000)),
        (0.200, (0.000, 0.427, 0.761)),
        (0.490, (0.000, 0.071, 0.314)),
        (0.500, (0.000, 0.000, 0.000)),
        (0.510, (0.337, 0.145, 0.000)),
        (0.719, (0.812, 0.518, 0.000)),
        (1.000, (1.000, 0.925, 0.200))
    )
)


def define_norm_cbar_label(data: np.ndarray, quantity: str) -> tuple:
    if quantity == 'Tk':
        norm, cmap, label = LogNorm(), plt.get_cmap('plasma'), r'$T_{\mathrm{k}} [K]$'
    elif quantity == 'lyal':
        norm, cmap, label = LogNorm(vmin=np.min(data[data > 0]), vmax=np.max(data)), plt.get_cmap('cividis'), r'$x_{\mathrm{al}}$'
    elif quantity == 'matter':
        norm, cmap, label = Normalize(vmin=-1,vmax=5),plt.get_cmap('viridis'), r'$\delta_{\mathrm{m}}$'
    elif quantity == 'Grid_xHII':
        norm, cmap, label = Normalize(vmin=0,vmax=1),plt.get_cmap('binary'), r'$x_{\mathrm{HII}}$'
    elif quantity == 'Grid_dTb':
        norm = TwoSlopeNorm(vmin = np.min(data), vcenter = 0, vmax = max(np.max(data), 0.001))
        cmap = COLOR_GRADIENT
        label = r'$\overline{dT}_{\mathrm{b}}$ [mK]'
    else:
        raise ValueError(f"Unknown quantity '{quantity}' for lightcone plotting.")

    return norm, cmap, label


def plot_lightcone(lightcone: Lightcone, ax: plt.Axes, description: str, slice_number: int = None) -> None:
    lbox = lightcone.parameters.simulation.Lbox
    redshifts = 1 / lightcone.redshifts - 1

    logger.debug(f"Lightcone dynamic range is {lightcone.data.min():.2e} to {lightcone.data.max():.2e}")
    # the lightcone.redshifts are actually scale factors, so we need to convert them

    if slice_number is None:
        slice_number = lightcone.data.shape[0] // 2

    norm, cmap, label = define_norm_cbar_label(lightcone.data[slice_number, ...], lightcone.quantity)

    xi = np.tile(lightcone.redshifts, (lightcone.data.shape[1], 1))
    yi = np.tile(np.linspace(0, lbox, lightcone.data.shape[1]).reshape(-1, 1), (1, lightcone.redshifts.size))
    zj = (
        lightcone.data[slice_number,1:,1:] +
        lightcone.data[slice_number,1:,:-1] +
        lightcone.data[slice_number,:-1,1:] +
        lightcone.data[slice_number,:-1,:-1]
    ) / 4

    logger.debug(f"Plotting slice {slice_number} with {xi.shape=} x {yi.shape=} x {zj.shape=}")

    im = ax.pcolormesh(xi, yi, zj, cmap=cmap, norm=norm)
    # show the description as white text in the top left corner
    ax.text(0.02, 0.1, description, color='white', fontweight="bold", transform=ax.transAxes)

    ax.set_xlabel('a')
    ax.set_ylabel('L (Mpc)')

    # Add a secondary x-axis to show the redshifts
    ax2 = ax.twiny()
    ax2.set_xlim(redshifts[0], redshifts[-1])
    redshift_ticks = np.linspace(redshifts[0], redshifts[-1], 10, endpoint=True)
    ax2.set_xticks(redshift_ticks, labels=np.round(redshift_ticks, 2))
    ax2.set_xlabel("z")
    # ax2.tick_params()

    # Plot the colorbar directly
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label)
