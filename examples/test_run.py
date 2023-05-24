import numpy as np
import beorn
from beorn.run import *

param = beorn.par()

# Halo Mass bins
param.sim.Mh_bin_min = 1e7
param.sim.Mh_bin_max = 1e15
param.sim.binn = 40  # nbr of halo mass bin

# Name attached to the outputs
param.sim.model_name = 'test'
# Nbr of cores to use
param.sim.cores = 1  

# Redshift of the snapshot(s)
param.solver.Nz = [7.72]  

# cosmo 
param.cosmo.Om = 0.31
param.cosmo.Ob = 0.045
param.cosmo.Ol = 0.69
param.cosmo.h = 0.68

## Source parameters
# lyal
param.source.N_al = 9690  # 1500
param.source.alS_lyal = 0.0
# ion
param.source.Nion = 3000  # 5000
# xray
param.source.E_min_xray = 500
param.source.E_max_xray = 10000
param.source.E_min_sed_xray = 200
param.source.E_max_sed_xray = 10000
param.source.alS_xray = 1.5
param.source.cX = 3.4e40
# fesc
param.source.f0_esc = 0.2
param.source.pl_esc = 0.5
# fstar
param.source.f_st = 0.14
param.source.g1 = 0.49
param.source.g2 = -0.61
param.source.g3 = 4
param.source.g4 = -4
param.source.Mp = 1.6e11 * param.cosmo.h
param.source.Mt = 1e9
# Minimum star forming halo
param.source.M_min = 1e8

# Box size and Number of pixels
Ncell = 128
Lbox = 100
param.sim.Lbox = Lbox
param.sim.Ncell = Ncell
param.sim.halo_catalogs = './halo_catalog/pkdgrav_halos_z'  ## path to dir with halo catalogs + filename
param.sim.thresh_pixel = 80 * (Ncell / 256) ** 3
param.sim.dens_field =  './density_field/grid_' + str(Ncell) + '_B100_CDM.z' # None
param.sim.dens_field_type = 'pkdgrav'
param.sim.store_grids = True

# Step 1 : Compute the profiles
compute_profiles(param)

# define k bins for PS measurement
kmin = 1 / Lbox
kmax = Ncell / Lbox
kbin = int(6 * np.log10(kmax / kmin))
param.sim.kmin = kmin
param.sim.kmax = kmax
param.sim.kbin = kbin

# Step 2 : Paint Boxes and read and write GS and PS in ./physics/
# change argument if you don't want all the maps.
paint_boxes(param, RSD=False, ion=True, temp=True, dTb=True, lyal=True,check_exists = False)

# Step 3 : gather the GS_PS files at different redshifts and create a single GS_PS.pkl file.
gather_GS_PS_files(param)

# Option : Read the profiles and do a quick calculation of the global quantities. Save the file.
GS = beorn.compute_glob_qty(param)
save_f(file='./physics/GS_approx'+'_' + param.sim.model_name + '.pkl',obj = GS)




##### Plot the results (dTb, Tk, xHII, PS_dTb(z))
##### Need more than 1 halo catalog.
import matplotlib.pyplot as plt
import matplotlib
from beorn.plotting import plot_Beorn_PS_of_z
import matplotlib.gridspec as gridspec
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
fig = plt.figure(constrained_layout=True)
fig.set_figwidth(18)

fig.set_figheight(6)

gs = gridspec.GridSpec(2, 3, figure=fig)

ax1 = fig.add_subplot(gs[:,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,1],sharex=ax2)
ax4 = fig.add_subplot(gs[:,2])
    
    

GS_PS = load_f('./physics/GS_PS_' + str(param.sim.Ncell) + '_' + param.sim.model_name + '.pkl')
GS_approx = load_f('./physics/GS_approx'+'_' + param.sim.model_name + '.pkl')

ax1.plot(GS_PS['z'],GS_PS['dTb'],'*',lw=2,alpha=0.8,ls='-',color='gray',label='BEoRN')
ax1.plot(GS_approx['z'],GS_approx['dTb'],ls='--',label='')
ax1.legend(fontsize=15,loc='upper right')
ax1.set_xlim(6,20)
ax1.set_ylim(-62,13)
ax1.set_xlabel('z',fontsize=15)
ax1.set_ylabel('dTb [mK]',fontsize=15)



ax2.plot(GS_PS['z'],GS_PS['Tk'],'*',lw=2,alpha=0.8,ls='-',color='gray',label='BEoRN')
ax2.plot(GS_approx['z'],GS_approx['Tk'],ls='--',color='gray')
ax2.semilogy([],[])
ax2.set_ylim(3,5e2)
ax2.set_ylabel('$T_{k}$ [K]',fontsize=17)
#plt.show()


ax3.plot(GS_PS['z'],GS_PS['x_HII'],'*',lw=2,alpha=0.8,ls='-',color='gray',label='BEoRN')
ax3.plot(GS_approx['z'],GS_approx['x_HII'],ls='--',color='gray')
ax3.set_xlim(6.3,15)
ax3.set_ylabel('$x_{\mathrm{HII}}$',fontsize=17)
ax3.set_xlabel('z ',fontsize=17)


plot_Beorn_PS_of_z(0.1, GS_PS, GS_PS,ls='-',lw=1, color='b',RSD = False,label='',qty='dTb',alpha=1,ax=plt)


ax4.plot([1e-1],[1e-2])

ax4.set_ylim(1e-1,1e2)
ax4.set_xlim(5.8,18)
ax4.set_ylabel('$\Delta_{21}^{2}(k,z)$ [mK]$^{2}$ ',fontsize=18) # k^{3}P(k)/(2\pi^{2})
ax4.set_xlabel('z ',fontsize=17)
ax4.legend(loc='best',fontsize=15)

ax2.axes.get_xaxis().set_visible(False)
plt.show()










