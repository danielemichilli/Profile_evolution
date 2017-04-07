import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

from mjd2date import convert

mpl.rc('font',size=8)

data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"


def plot(fig):
  res = np.load(os.path.join(data_folder, 'timing_res.npy'))
  vdot = np.load(os.path.join(data_folder, 'vdot.npy'))
  DM = np.load(os.path.join(data_folder, 'DM.npy'))
  DM_tel = np.load(os.path.join(data_folder, 'DM_tel.npy'))
  x_yr = [convert(n) for n in res[0]]

  gs = gridspec.GridSpec(3, 1, height_ratios=[1,.5,1], hspace=.1)
  ax1 = plt.subplot(gs[0])
  ax2 = plt.subplot(gs[1])
  ax3 = plt.subplot(gs[2])

  ax1.errorbar(x_yr, res[1]*1e3, yerr=res[2]*1e3, fmt='ok', markersize=2)
  ax1.set_ylabel('Residual (ms)')
  ax1.set_xlabel('Year')
  ax1.xaxis.set_label_position('top') 
  ax1.tick_params(axis='x', labelbottom='off', labeltop='on', bottom='off')
  ax1.axvline(convert(55859.43), c='k', ls='--')
  ax1.axvspan(convert(55402), convert(56258), fc='r', ec=None, alpha=.2, linewidth=0.)
  ax1.axvspan(convert(56258), convert(res[0,-1]), fc='g', ec=None, alpha=.2, linewidth=0.)
  ax1b = ax1.twinx()
  ax1b.set_ylim([ax1.get_ylim()[0]/5.384688219194, ax1.get_ylim()[1]/5.384688219194])
  ax1b.set_ylabel('Residual (% phase)')
  
  ax2.errorbar(vdot[0, 105:-10], vdot[1, 105:-10]/1e-15, yerr=vdot[2, 105:-10]/1e-15, fmt='ok', markersize=2)
  ax2.set_ylabel(r'$\dot{\nu}\ \times\ 10^{-15}$ Hz/s')
  ax2.tick_params(axis='x', labelbottom='off', labeltop='off', bottom='off', top='off')
  ax2.axvline(55859.43, c='k', ls='--')
  ax2.axvspan(55402, 56258, fc='r', ec=None, alpha=.2, linewidth=0.)
  ax2.axvspan(56258, res[0,-1], fc='g', ec=None, alpha=.2, linewidth=0.)
  ax2.ticklabel_format(useOffset=False)
  ax2.set_yticks(np.linspace(-9.541,-9.533,5))

  idx = np.argsort(DM[0, DM_tel != 'LWA1'])
  ax3.plot(DM[0, idx], DM[1, idx], 'k-', markersize=5)
  x_dm = np.array([res[0,0], res[0,-1]])
  ax3.plot(x_dm, (x_dm - res[0,0]) / 365. * -2e-4 * np.sqrt(DM[1].mean()) + DM[1].max()+0.01, 'r-')
  colors = ['b', 'k', 'g', 'r']
  for i, tel in enumerate(np.unique(DM_tel)):
    idx = np.where(DM_tel == tel)[0]
    d = DM[:, idx]
    ax3.errorbar(d[0], d[1], yerr=d[2], fmt='o', c=colors[i], label=tel, markeredgewidth=0., markersize=3)
  ax3.set_xlabel('MJD')
  ax3.tick_params(axis='x', top='off')
  ax3.legend(loc='lower left', fancybox=True)
  ax3.set_ylabel('DM (pc cm$^{-3}$)')
  ax3.ticklabel_format(useOffset=False)
  #ax3.axvline(55859.43, c='k', ls='--')
  #ax3.axvspan(56258, res[0,-1], fc='g', ec=None, alpha=.2)
  
  ax1.get_yaxis().set_label_coords(-0.09,0.5)
  ax2.get_yaxis().set_label_coords(-0.09,0.5)
  ax3.get_yaxis().set_label_coords(-0.09,0.5)

  ax1.set_xlim([convert(res[0,0]), convert(res[0,-1])])
  ax2.set_xlim([res[0,0], res[0,-1]])
  ax3.set_xlim([res[0,0], res[0,-1]])

  def label(ax, number):
    at = AnchoredText(number, prop=dict(size=15), loc=2, frameon=True, pad=.1, borderpad=0.)
    ax.add_artist(at)
    return
  label(ax1, "(a)")
  label(ax2, "(b)")
  label(ax3, "(c)")

  fig.add_subplot(ax1)
  fig.add_subplot(ax2)
  fig.add_subplot(ax3)
  return 


if __name__ == '__main__':
  fig = plt.figure(figsize=(7,4))
  plot(fig)

  fig.savefig('timing.eps', papertype='a4', orientation='portrait', format='eps', dpi=200)

  plt.show()





