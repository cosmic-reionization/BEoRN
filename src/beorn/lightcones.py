"""
Scripts to plot lightcones.
"""
import tools21cm as t2c
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.pyplot as plt
import cmasher as cmr
from .functions import *


class lightcone:
    def __init__(self, param, qty='dTb',slice_nbr = 64):
        self.path = './grid_output/'
        self.mean_array = []
        self.coeval_set = {}
        self.zs_set = []
        self.nGrid = param.sim.Ncell
        self.qty = qty  # the quantity you want to plot
        self.Lbox = param.sim.Lbox
        self.z_liste = param.solver.Nz
        self.param = param
        self.slice_nbr = slice_nbr
        print('nGrid is', self.nGrid, '. Lbox is', self.Lbox, 'Mpc.', 'Plotting lightcone for z =', self.z_liste,'and slice nbr',slice_nbr)

    def load_boxes(self):
        print('Loading boxes...')
        for iz, zi in enumerate(self.z_liste):
            Grid = load_grid(self.param, zi, type=self.qty)
            self.coeval_set[z_string_format(zi)] = Grid
            self.zs_set.append(self.z_liste[iz])
            self.mean_array.append(np.mean(Grid))
        self.zs_set = np.array(self.zs_set)

    def reading_function(self, name):
        return self.coeval_set[name]

    def generate_lightcones(self):
        print('Generating lightcones...')
        filenames = [z_string_format(zi) for zi in self.zs_set]
        self.scale_fac = 1 / (self.zs_set + 1)
        print('scale_fac :', self.scale_fac, 'z : ', self.zs_set)
        self.xf_lc, self.zs_lc = t2c.make_lightcone(
            filenames,
            z_low=None,
            z_high=None,
            file_redshifts=self.scale_fac,  ## FOR dTb LIGHTCONE
            # file_redshifts=zs_set, ## FOR xHII
            cbin_bits=self.nGrid,
            cbin_order='c',
            los_axis=2,
            raw_density=False,
            # interpolation='sigmoid',
            reading_function=self.reading_function,
            box_length_mpc=self.Lbox,
        )

    def slice_av(self, grid, nbr, center):
        av = grid[center, :, :]
        for i in range(1, nbr + 1):
            print(i)
            av += grid[center + i, :, :] + grid[center - i, :, :]
        av = av / (2 * nbr + 1)
        print((2 * nbr + 1))
        return av

    def plotting_lightcone(self, save='./lightcone.jpg'):
        import cmasher as cmr
        from matplotlib.colors import TwoSlopeNorm

        print('Range for Lightcone plot is :', np.min(self.mean_array), np.max(self.mean_array))

        if self.qty == 'bubbles':
            norm = TwoSlopeNorm(vmin=0, vcenter=0.5,vmax=1)
            cmap = plt.get_cmap('viridis')
            label = 'xHII'
        elif self.qty == 'dTb':
            norm = TwoSlopeNorm(vmin=np.min(self.mean_array), vcenter=0, vmax=np.maximum(0.1, np.max(self.mean_array)))
            cmap = plt.get_cmap('cmr.iceburn')
            label ='dTb [mK]'
        else:
            norm = TwoSlopeNorm(vmin=np.min(self.mean_array), vcenter=0, vmax=np.maximum(0.1, np.max(self.mean_array)))

        xi = np.array([self.zs_lc for i in range(self.xf_lc.shape[1])])
        yi = np.array([np.linspace(0, int(self.Lbox), self.xf_lc.shape[1]) for i in range(xi.shape[1])]).T
        zj = self.slice_av(self.xf_lc, 0, self.slice_nbr)  # self.xf_lc[64,:,:] #slice_av(self.xf_lc, 1, 64)

        fig, axs = plt.subplots(1, 1, figsize=(20, 6))
        ax2 = axs.twiny()
        im = axs.pcolor(xi, yi, zj, cmap=cmap, norm=norm)

        axs.set_xlabel('a(t)', fontsize=18)
        axs.set_ylabel('L (Mpc)', fontsize=18)

        indices = []
        for zzz in [20, 17, 14, 10, 8,6]:  ### for the upper x tick-axis of the plot, find the correct indices corresponding to desired redshifts.
            indices.append(np.argmin(np.abs(self.zs_set - zzz)))
        # indices = np.array(indices)

        ax2.set_xlim(axs.get_xlim())
        ax2.set_xticks(self.scale_fac[indices], )
        ax2.set_xticklabels(np.round(self.zs_set[indices], 2))
        ax2.set_xlabel("z", fontsize=18)
        ax2.tick_params(labelsize=18)
        axs.tick_params(labelsize=18)
        # axs.set_xticks(np.arange(6.5,13,1))
        # axs.set_yticks(np.arange(0,350,100))
        fig.subplots_adjust(bottom=0.11, right=0.91, top=0.9, left=0.06)
        cax = plt.axes([0.92, 0.15, 0.02, 0.75])

        cb = fig.colorbar(im, cax=cax, label=label)
        cb.ax.tick_params(labelsize=18)
        cb.set_label(label=label, fontsize=18)

        # plt.tight_layout()
        plt.savefig(save)  # , bbox_inches='tight')
        plt.show()

