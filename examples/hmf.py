### Example script to compute analytical HMF using PS formalism.

import beorn
from beorn.halomassfunction import HMF as hmf
from beorn.halomassfunction import from_catalog_to_hmf
import matplotlib.pyplot as plt
from beorn.functions import *

par = beorn.par()

# tophat HMF
z_array = [20., 15., 10., 7.72,5]
par.hmf.z = z_array#[10.19,12.19,15.39,17.91,5.9,9,8.27]
par.hmf.c = 1
par.hmf.delta_c = 1.675
par.hmf.p = 0.3
par.hmf.filter = 'tophat'
par.hmf.q = 0.85 

HMF = hmf(par)
HMF.generate_HMF(par)

# sharpk HMF
par.hmf.z = z_array
par.hmf.c = 2.7
par.hmf.delta_c = 1.675
par.hmf.p = 0.3
par.hmf.filter = 'sharpk'
par.hmf.q = 1
HMF_sharpk = hmf(par)
HMF_sharpk.generate_HMF(par)


# read data for comparison
Halo_dict = load_f('./halo_catalog/pkdgrav_halos_z07.72')
hmf_data = from_catalog_to_hmf(Halo_dict)
z = Halo_dict['z']
plt.errorbar(hmf_data[0]/0.68,hmf_data[1]*0.68**3,hmf_data[2]*0.68**3,marker = '*',linestyle='',markersize=7,color='C'+str(3),alpha=0.8,label='fof, z ='+str(round(z,2)))
  


 
# plot
for ii,zz_i in enumerate(z_array):#
    plt.loglog(HMF.tab_M/0.68,HMF.HMF[ii]*0.68**3,ls='--',color='C'+str(ii))
    plt.loglog(HMF_sharpk.tab_M/0.68,HMF_sharpk.HMF[ii]*0.68**3,ls='-',color='C'+str(ii),label='z={}'.format(zz_i))
    

plt.loglog([],[],ls='--',color='gray',label='tophat')
plt.loglog([],[],ls='-',color='gray',label='sharpk')
    
               
plt.legend()
plt.ylim(3e-7,1e2)
plt.xlim(1.45e8,4e13)
plt.ylabel('dn/dlnM $(Mpc)^{-3}$', fontsize=12)
plt.legend(fontsize=13)
plt.xlabel('M$_{h}  (M_{\odot})$', fontsize=15)
plt.tick_params(axis="both",labelsize=14)
plt.tight_layout()
plt.show()    
