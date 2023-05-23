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

#
class lightcone_21cmFAST:
    def __init__(self,nGrid = 300,Lbox = 147,path = './Boxes',type='21cmFAST',zliste = None,qty = 'delta_T_z'):
        #Om, Ob, h0 = 0.31, 0.045, 0.68
        #factor = 27 * Ob * h0 ** 2 / 0.023 * np.sqrt(0.15 / Om / h0 ** 2 / 10)
        #Tcmb0 = 2.725
        self.path = path
        self.mean_array = []
        self.coeval_set = {}
        self.zs_set = []
        self.nGrid = nGrid
        self.qty = qty  # the quantity you want to plot
        self.type = type # the type of sim output we are dealing with
        self.Lbox = Lbox
        print('nGrid is ', nGrid, 'Lbox is', Lbox, 'Mpc')
        if self.type == '21cmFAST':
            z_liste = []
            for file in os.listdir(path):
                    filename = os.fsdecode(file)
                    if filename.startswith("delta_T_z"):
                            z_liste.append(float(filename[10:15]))
            z_liste.sort()
            self.z_liste = np.flip(np.array(z_liste))

        elif self.type == 'Beorn':
            self.z_liste = zliste


    def load_boxes(self):
        for iz, zi in enumerate(self.z_liste):
            if 25 > zi > 6:
                if zi<10:
                    string_z = '00'+"{:.2f}".format(zi)
                else :
                    string_z = '0' + "{:.2f}".format(zi)
                # try :
                print(zi)
                #T_gamma_z = Tcmb0 * (1 + zi)
                #T_adiab = Tcmb0 * (1 + zi) ** 2 / (1 + 250)
                for file in os.listdir(self.path):
                    filename = os.fsdecode(file)
                    if self.type == '21cmFAST':
                        if filename.startswith(self.qty + string_z):   ####choose the qty you want to plot here
                            Grid = t2c.read_21cmfast_files(filename)
                    elif self.type == 'Beorn':
                        if filename.startswith(self.qty + str(iz)):   ####choose the qty you want to plot here
                            Grid = load_f(filename)

                self.coeval_set['{:.2f}'.format(zi)] = Grid
                self.zs_set.append(self.z_liste[iz])
                self.mean_array.append(np.mean(Grid))

        self.zs_set = np.array(self.zs_set)


    def reading_function(self,name):
        return self.coeval_set[name]


    def generate_lightcones(self):
        filenames = ['{:.2f}'.format(zi) for zi in self.zs_set]
        self.scale_fac = 1/(self.zs_set+1)
        print('scale_fac :',self.scale_fac,'z : ',self.zs_set)
        self.xf_lc, self.zs_lc = t2c.make_lightcone(
                                    filenames,
                                    z_low=None,
                                    z_high=None,
                                    file_redshifts=self.scale_fac, ## FOR dTb LIGHTCONE
                                    #file_redshifts=zs_set, ## FOR xHII
                                    cbin_bits=self.nGrid,
                                    cbin_order='c',
                                    los_axis=2,
                                    raw_density=False,
                                    #interpolation='sigmoid',
                                    reading_function = self.reading_function,
                                    box_length_mpc = self.Lbox,
                                )

    def slice_av(self,grid, nbr, center):
        av = grid[center, :, :]
        for i in range(1, nbr + 1):
            print(i)
            av += grid[center + i, :, :] + grid[center - i, :, :]
        av = av / (2 * nbr + 1)
        print((2 * nbr + 1))
        return av

    def plotting_lightcone(self, save='./lightcone.jpg'):
        cmap = plt.get_cmap('cmr.iceburn')
        norm = TwoSlopeNorm(vmin=np.min(self.mean_array), vcenter=0, vmax=np.max(self.mean_array))

        xi = np.array([self.zs_lc for i in range(self.xf_lc.shape[1])])
        yi = np.array([np.linspace(0, int(self.Lbox), self.xf_lc.shape[1]) for i in range(xi.shape[1])]).T
        zj = self.slice_av(self.xf_lc, 1,64) #self.xf_lc[64,:,:] #slice_av(self.xf_lc, 1, 64)

        fig, axs = plt.subplots(1, 1, figsize=(20, 6))
        ax2 = axs.twiny()
        im = axs.pcolor(xi, yi, zj, cmap=cmap, norm=norm)

        axs.set_xlabel('a(t)', fontsize=18)
        axs.set_ylabel('L (Mpc)', fontsize=18)

        indices = []
        for zzz in [20,17,14,10,8,6]: ### for the upper x tick-axis of the plot, find the correct indices corresponding to desired redshifts.
            indices.append(np.argmin(np.abs(self.zs_set-zzz)))
        #indices = np.array(indices)

        ax2.set_xlim(axs.get_xlim())
        ax2.set_xticks(self.scale_fac[indices],)
        ax2.set_xticklabels(np.round(self.zs_set[indices], 2))
        ax2.set_xlabel("z", fontsize=18)
        ax2.tick_params(labelsize=18)
        axs.tick_params(labelsize=18)
        # axs.set_xticks(np.arange(6.5,13,1))
        # axs.set_yticks(np.arange(0,350,100))
        fig.subplots_adjust(bottom=0.11, right=0.91, top=0.9, left=0.06)
        cax = plt.axes([0.92, 0.15, 0.02, 0.75])

        cb = fig.colorbar(im, cax=cax, label='dTb [mK]')
        cb.ax.tick_params(labelsize=18)
        cb.set_label(label='dTb [mK]', fontsize=18)

        #plt.tight_layout()
        plt.savefig(save)#, bbox_inches='tight')
        plt.show()

        plt.plot(self.zs_set,self.mean_array)
        plt.show()






lightcone = lightcone_21cmFAST(path='./')
lightcone.load_boxes()
lightcone.generate_lightcones()
lightcone.plotting_lightcone()




















path =
mean_array = []
coeval_set = {}
zs_set = []
type =
Lbox =
print('nGrid is ', nGrid, 'Lbox is', Lbox, 'Mpc')
if type == '21cmFAST':
    z_liste = []
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.startswith("delta_T_z"):
            z_liste.append(float(filename[10:15]))
    z_liste.sort()
    z_liste = np.flip(np.array(z_liste))

for iz, zi in enumerate(z_liste):
    if 25 > zi > 6:
        if zi < 10:
            string_z = '00' + "{:.2f}".format(zi)
        else:
            string_z = '0' + "{:.2f}".format(zi)
        print(zi)
        for file in os.listdir(path):
            filename = os.fsdecode(file)
            if filename.startswith(self.qty + string_z):  ####choose the qty you want to plot here
                Grid = t2c.read_21cmfast_files(filename)

            coeval_set['{:.2f}'.format(zi)] = Grid
            zs_set.append(self.z_liste[iz])
            mean_array.append(np.mean(Grid))

    zs_set = np.array(self.zs_set)




def reading_function(self, name):
    return self.coeval_set[name]


def generate_lightcones(self):
    filenames = ['{:.2f}'.format(zi) for zi in self.zs_set]
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
    cmap = plt.get_cmap('cmr.iceburn')
    norm = TwoSlopeNorm(vmin=np.min(self.mean_array), vcenter=0, vmax=np.max(self.mean_array))

    xi = np.array([self.zs_lc for i in range(self.xf_lc.shape[1])])
    yi = np.array([np.linspace(0, int(self.Lbox), self.xf_lc.shape[1]) for i in range(xi.shape[1])]).T
    zj = self.slice_av(self.xf_lc, 1, 64)  # self.xf_lc[64,:,:] #slice_av(self.xf_lc, 1, 64)

    fig, axs = plt.subplots(1, 1, figsize=(20, 6))
    ax2 = axs.twiny()
    im = axs.pcolor(xi, yi, zj, cmap=cmap, norm=norm)

    axs.set_xlabel('a(t)', fontsize=18)
    axs.set_ylabel('L (Mpc)', fontsize=18)

    indices = []
    for zzz in [20, 17, 14, 10, 8,
                6]:  ### for the upper x tick-axis of the plot, find the correct indices corresponding to desired redshifts.
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

    cb = fig.colorbar(im, cax=cax, label='dTb [mK]')
    cb.ax.tick_params(labelsize=18)
    cb.set_label(label='dTb [mK]', fontsize=18)

    # plt.tight_layout()
    plt.savefig(save)  # , bbox_inches='tight')
    plt.show()

    plt.plot(self.zs_set, self.mean_array)
    plt.show()
