import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from matplotlib import rcParams
rcParams['ps.fonttype'] = 42

mpl.rc('font',size=8)

data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"

val = np.load(os.path.join(data_folder, 'Census_DM_Flux.npy'))

B2217_pos = [43.4862, 820.]

fig = plt.figure(figsize=(3.3,2))

plt.errorbar(val[0], val[2]/1000., yerr=val[3]/1000., fmt='ok', ms=1, lw=.5)
plt.annotate('B2217+47', xy=(B2217_pos[0], B2217_pos[1]/1000.), xytext=(B2217_pos[0]+2, B2217_pos[1]/1000.-.3), horizontalalignment='left', verticalalignment='centre')
plt.plot(val[0,144], val[2,144]/1000., 'or', markeredgewidth=0., ms=2)

plt.xscale('log')
plt.yscale('log')
#plt.ylim((0,1.5))
plt.xlim((3,200))

plt.xlabel("DM (pc cm$^{-3}$)")
plt.ylabel("Flux density (Jy)")


fig.tight_layout()
fig.savefig('flux_distribution.eps', papertype='a4', orientation='portrait', format='eps', dpi=200)


plt.show()


