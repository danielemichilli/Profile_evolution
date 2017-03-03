import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mjd2date import convert


data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"


def plot(fig, ratio=0.):
  res = np.load(os.path.join(data_folder, 'timing_res.npy'))
  vdot = np.load(os.path.join(data_folder, 'vdot.npy'))
  DM = np.load(os.path.join(data_folder, 'DM.npy'))
  DM_tel = np.load(os.path.join(data_folder, 'DM_tel.npy'))
  x_yr = [convert(n) for n in res[0]]

  gs = gridspec.GridSpec(3, 1, height_ratios=[1,.5,1], hspace=.1, left=0.6, right=.95, bottom=0.05, top=ratio-0.1)
  ax1 = plt.subplot(gs[0])
  ax2 = plt.subplot(gs[1])
  ax3 = plt.subplot(gs[2])

  ax1.errorbar(x_yr, res[1]*1e3, yerr=res[2]*1e3, fmt='ok')
  ax1.set_ylabel('Residual (ms)')
  ax1.tick_params(axis='x', labelbottom='off', labeltop='on', bottom='off')
  ax1.axvline(convert(55859.43), c='k', ls='--')
  ax1.axvspan(convert(56258), convert(res[0,-1]), fc='g', ec=None, alpha=.2)

  ax2.errorbar(vdot[0, 11:-10], vdot[1, 11:-10]/1e-15, yerr=vdot[2, 11:-10]/1e-15, fmt='ok')
  ax2.set_ylabel(r'$\dot{\nu}\ \times\ 10^{-15}$ Hz/s')
  ax2.tick_params(axis='x', labelbottom='off', labeltop='off', bottom='off', top='off')
  ax2.axvline(55859.43, c='k', ls='--')
  ax2.axvspan(56258, res[0,-1], fc='g', ec=None, alpha=.2)

  colors = ['b', 'k', 'g', 'r']
  for i, tel in enumerate(np.unique(DM_tel)):
    idx = np.where(DM_tel == tel)[0]
    d = DM[:, idx]
    ax3.errorbar(d[0], d[1], yerr=d[2], fmt='o', c=colors[i], label=tel)
  idx = np.argsort(DM[0])
  ax3.plot(DM[0, idx], DM[1, idx], 'k-')
  ax3.set_xlabel('MJD')
  ax3.tick_params(axis='x', top='off')
  ax3.legend(loc='center left', fancybox=True, framealpha=0.5)
  x_dm = np.array([res[0,0], res[0,-1]])
  ax3.plot(x_dm, (x_dm - res[0,0]) / 365. * -2e-4 * np.sqrt(DM[1].mean()) + DM[1].max()+0.01, 'r-')
  ax3.set_ylabel('DM (pc$\cdot$cm$^{-3}$)')
  ax3.ticklabel_format(useOffset=False)
  ax3.axvline(55859.43, c='k', ls='--')
  #ax3.axvspan(56258, res[0,-1], fc='g', ec=None, alpha=.2)

  ax1.get_yaxis().set_label_coords(-0.08,0.5)
  ax2.get_yaxis().set_label_coords(-0.08,0.5)
  ax3.get_yaxis().set_label_coords(-0.08,0.5)

  ax1.set_xlim([convert(res[0,0]), convert(res[0,-1])])
  ax2.set_xlim([res[0,0], res[0,-1]])
  ax3.set_xlim([res[0,0], res[0,-1]])

  fig.add_subplot(ax1)
  fig.add_subplot(ax2)
  fig.add_subplot(ax3)
  return 


if __name__ == '__main__':
  fig = plt.figure(figsize=(10,5))
  plot_grid = gridspec.GridSpec(1, 1)
  
  plot(fig, plot_grid[0])

  plt.show()
