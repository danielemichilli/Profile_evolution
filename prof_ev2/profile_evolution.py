import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import LOFAR_span_all
import LOFAR_new

mpl.rc('font',size=8)
#mpl.rc('text',usetex=True)
#mpl.rc('font',**{'family':'serif','serif':['Computer Modern']})

fig = plt.figure(figsize=(7,8))

#gs = gridspec.GridSpec(3, 2, height_ratios=[1.-1000./2387., 0.1, 1000./2387.-.1], wspace=0.06, hspace=0.)
total_span = 6.53424657534 #yr
offset_new = 2.75 #yr
gs = gridspec.GridSpec(3, 2, height_ratios=[1.-offset_new/total_span, 0.1, offset_new/total_span-.1], width_ratios=[1., 1./3.], wspace=0.06, hspace=0.)

ax, scale = LOFAR_span_all.plot(gs[:, 0])
LOFAR_new.plot(gs[0, 1], ax_ref=ax)

gs_cb = gridspec.GridSpecFromSubplotSpec(1, 2, gs[-1, 1], width_ratios=[0.2,1.])
ax_cb = plt.subplot(gs_cb[0])

cbar = plt.colorbar(scale, cax=ax_cb)
cbar.ax.yaxis.set_label_position("right")
cbar.set_label('Flux (% peak)')
cbar.ax.tick_params(which='both', axis='y', labelleft='off', labelright='on', right='on', left='off')
cbar.ax.yaxis.set_ticks(np.arange(0, 1.01, .2))
cbar.ax.set_yticklabels(range(0,16,3))
cbar.ax.minorticks_on()

fig.savefig('evolution.eps', papertype='a4', orientation='portrait', format='eps', dpi=200)


plt.show()






