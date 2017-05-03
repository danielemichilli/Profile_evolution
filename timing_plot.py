import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import datetime

from mjd2date import convert

mpl.rc('font',size=8)
ref_date = datetime.date(2010, 7, 25)
ref_mjd = 55402
data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"


def plot_profile(ax):
  #Load LOFAR observations
  dates = np.load(os.path.join(data_folder, 'LOFAR_profiles_dates.npy'))
  observations = np.load(os.path.join(data_folder, 'LOFAR_profiles.npy'))

  #Average Observations on the same day
  date_uniq, idx_uniq, repeated = np.unique(dates, return_index=True, return_counts=True)
  obs_uniq = observations[idx_uniq]
  idx = np.where(repeated>1)[0]
  for i in idx:
    date_rep = np.where(dates == date_uniq[i])[0]
    obs_rep = np.sum(observations[date_rep], axis=0)
    obs_uniq[i] = obs_rep

  #Create an image of the profile evolution calibrated in time
  days = (dates[-1] - dates[0]).days
  img = np.zeros((days,512))
  for i,n in enumerate(img):
    idx = np.abs(date_uniq - date_uniq[0] - datetime.timedelta(i)).argmin()
    img[i] = obs_uniq[idx]

  img -= np.median(img, axis=1, keepdims=True)
  img /= np.max(img, axis=1, keepdims=True)

  date_max = (dates[-1] - ref_date).days
  phase_min = -258. / 512. * 538.4688219194
  phase_max = (512. - 258.) / 512. * 538.4688219194
  s = ax.imshow(np.clip(img.T,0,0.15*img.T.max()),cmap='hot',origin="lower",aspect='auto',interpolation='nearest',extent=[0, date_max/365., phase_min, phase_max])
  ax.tick_params(axis='both', labelleft='off', labelbottom='off', top='off', bottom='off', left='off', right='off')
  ax.set_ylim([-20, 50])
  ax.set_xlim([(dates[0] - ref_date).days/365., (dates[-1] - ref_date).days/365.])
  ax.axis('off')
  return


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

  ax1.errorbar(x_yr, res[1]*1e3, yerr=res[2]*1e3, fmt='ok', markersize=2, capsize=0)
  ax1.set_ylabel('Residual (ms)')
  ax1.set_xlabel('Year')
  ax1.xaxis.set_label_position('top') 
  ax1.tick_params(axis='x', labelbottom='off', labeltop='on', bottom='off')
  ax1.minorticks_on()
  ax1.tick_params(axis='both', which='minor', top='on', bottom='off', left='off', right='off')
  ax1.axvline(convert(55859.43), c='k', ls='--')
  ax1.axvspan(convert(55402), convert(56258), fc='r', ec=None, alpha=.2, linewidth=0.)
  ax1.axvspan(convert(56258), convert(res[0,-1]), fc='g', ec=None, alpha=.2, linewidth=0.)
  pos = ax1.get_position()
  x_in = (55402.-res[0,0]) / (res[0,-1]-res[0,0]) * (pos.xmax - pos.xmin) + pos.xmin
  y_in = (pos.ymax - pos.ymin) * 0.4
  ax1_in = plt.axes([x_in, pos.ymin, pos.xmax-x_in, y_in])
  ax1_in.tick_params(axis='both', labelleft='off', labelbottom='off', bottom='off', top='off', left='off', right='off')
  plot_profile(ax1_in)
  ax1b = ax1.twinx()
  ax1b.tick_params(axis='both', which='minor', top='off', bottom='off', left='off', right='off')
  ax1b.set_ylim([ax1.get_ylim()[0]/5.384688219194, ax1.get_ylim()[1]/5.384688219194])
  ax1b.set_ylabel('Residual (% phase)')

  ax2.errorbar(vdot[0, 105:-10], vdot[1, 105:-10]/1e-15, yerr=vdot[2, 105:-10]/1e-15, fmt='ok', markersize=2, capsize=0)
  ax2.set_ylabel(r'$\dot{\nu}\ \times\ 10^{-15}$ Hz/s')
  ax2.tick_params(axis='x', labelbottom='off', labeltop='off', bottom='off', top='off')
  ax2.axvline(55859.43, c='k', ls='--')
  ax2.axvspan(55402, 56258, fc='r', ec=None, alpha=.2, linewidth=0.)
  ax2.axvspan(56258, res[0,-1], fc='g', ec=None, alpha=.2, linewidth=0.)
  ax2.ticklabel_format(useOffset=False)
  ax2.set_yticks(np.linspace(-9.541,-9.533,5))

  idx = np.argsort(DM[0, DM_tel != 'LWA1'])
  x_dm = np.array([res[0,0], res[0,-1]])
  ax3.plot(x_dm, (x_dm - res[0,0]) / 365. * -2e-4 * np.sqrt(DM[1].mean()) + DM[1].max()+0.01, 'r-')
  ax3.plot(DM[0, idx], DM[1, idx], 'k-', markersize=5)
  colors = ['b', 'k', 'g', 'r']
  for i, tel in enumerate(np.unique(DM_tel)):
    idx = np.where(DM_tel == tel)[0]
    d = DM[:, idx]
    ax3.errorbar(d[0], d[1], yerr=d[2], fmt='o', c=colors[i], label=tel, markeredgewidth=0., markersize=3, capsize=0, zorder=3)
  ax3.set_xlabel('MJD')
  ax3.tick_params(axis='x', top='off')
  ax3.minorticks_on()
  ax3.tick_params(axis='both', which='minor', bottom='on', top='off', left='off', right='off')
  ax3.legend(loc='lower left', fancybox=True, numpoints=1)
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





