"""
Scripts to plot lightcones.
"""
import tools21cm as t2c
import os
import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.pyplot as plt
import cmasher as cmr
from .functions import *


class lightcone:
    def __init__(self, param, qty='dTb',slice_nbr = None,path='./grid_output/'):
        self.path = path
        self.mean_array = []
        self.coeval_set = {}
        self.zs_set = []
        self.nGrid = param.sim.Ncell
        self.qty = qty  # the quantity you want to plot
        self.Lbox = param.sim.Lbox
        self.z_liste = def_redshifts(param)
        self.param = param
        if slice_nbr is None: self.slice_nbr = int(self.nGrid/2)
        else :  self.slice_nbr = slice_nbr

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



    def define_norm_cbar_label(self):
        if self.qty == 'Tk':
            norm, cmap, label = matplotlib.colors.LogNorm(), plt.get_cmap('plasma'), r'$T_{\mathrm{k}} [K]$'
        elif self.qty == 'lyal':
            norm, cmap, label = matplotlib.colors.LogNorm(vmin=np.min(self.mean_array[self.mean_array > 0]), vmax=np.max(self.mean_array)), plt.get_cmap('cividis'), r'$x_{\mathrm{al}}$'
        elif self.qty == 'matter':
            norm, cmap, label = matplotlib.colors.Normalize(vmin=-1,vmax=5),plt.get_cmap('viridis'), r'$\delta_{\mathrm{m}}$'
        elif self.qty == 'bubbles':
            norm, cmap, label = matplotlib.colors.Normalize(vmin=0,vmax=1),plt.get_cmap('binary'), r'$x_{\mathrm{HII}}$'
        elif self.qty == 'dTb':
            norm, cmap, label = TwoSlopeNorm(vmin=np.min(self.mean_array), vcenter=0, vmax=max(np.max(self.mean_array),0.001)),my_color_gradient_(),'$\overline{dT}_{\mathrm{b}}$ [mK]'
        else:
            norm = matplotlib.colors.LogNorm(vmin=np.min(self.mean_array) + 1, vmax=np.max(self.mean_array) + 1)
        return norm, cmap, label


    def plotting_lightcone(self, save='./lightcone.jpg',save_data_slice = None ):
        import cmasher as cmr
        from matplotlib.colors import TwoSlopeNorm

        print('Range for Lightcone plot is :', np.min(self.mean_array), np.max(self.mean_array))

        norm, cmap, label = self.define_norm_cbar_label()

        xi = np.array([self.zs_lc for i in range(self.xf_lc.shape[1])])
        yi = np.array([np.linspace(0, int(self.Lbox), self.xf_lc.shape[1]) for i in range(xi.shape[1])]).T
        zj = self.slice_av(self.xf_lc, 0, self.slice_nbr)  # self.xf_lc[64,:,:] #slice_av(self.xf_lc, 1, 64)

        #zj_average_1 = self.slice_av(self.xf_lc, 1, self.slice_nbr)
        #zj_average_2 = self.slice_av(self.xf_lc, 2, self.slice_nbr)
        #zj_average_3 = self.slice_av(self.xf_lc, 3, self.slice_nbr)
        #zj_average_4 = self.slice_av(self.xf_lc, 4, self.slice_nbr)

        if save_data_slice is not None :
            save_f(file=save_data_slice,obj={'xi':xi,'yi':yi,'zj':zj,'mean':self.mean_array})#,'zj_av_1':zj_average_1,\
                                            # 'zj_av_2':zj_average_2,'zj_av_3':zj_average_3,'zj_av_4':zj_average_4})

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


def my_color_gradient_():
    my_gradient = \
        LinearSegmentedColormap.from_list('my_gradient', (
            # Edit this gradient at https://eltos.github.io/gradient/#0:78E4FF-20:006DC2-49:001250-50:000000-51:562500-71.9:CF8400-100:FFEC33
            (0.000, (0.471, 0.894, 1.000)),
            (0.200, (0.000, 0.427, 0.761)),
            (0.490, (0.000, 0.071, 0.314)),
            (0.500, (0.000, 0.000, 0.000)),
            (0.510, (0.337, 0.145, 0.000)),
            (0.719, (0.812, 0.518, 0.000)),
            (1.000, (1.000, 0.925, 0.200))))
    #Good sharp gradient for dTb : https://eltos.github.io/gradient/#0:78E4FF-20:006DC2-49:001250-50:000000-51:562500-71.9:CF5D00-100:FFEC33
    return my_gradient