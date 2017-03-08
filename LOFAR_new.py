import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

from mjd2date import convert


data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"
ref_date = datetime.date(2010, 7, 25)
ref_mjd = 55402

days_min = 1000.

def plot(grid, ax_ref=False):
  gs = gridspec.GridSpecFromSubplotSpec(1, 3, grid, wspace=.15, width_ratios=[1.,.5,.5])
  ax1 = plt.subplot(gs[2])
  ax2 = plt.subplot(gs[1], sharey=ax1)
  ax3 = plt.subplot(gs[0], sharey=ax1, sharex=ax_ref)

  profiles(ax3)
  DM(ax2)
  flux(ax1)

  def label(ax, number):
    at = AnchoredText(number, prop=dict(size=15), loc=2, frameon=True, pad=.1, borderpad=0.)
    ax.add_artist(at)
    return
  label(ax3, "(d)")
  label(ax2, "(e)")
  label(ax1, "(f)")

  ax1.tick_params(axis='y', labelright='on')
  ax1.yaxis.set_label_position("right")
  ax1.set_ylabel('Years after MJD 55402 (2010 July 25)')

  return



def profiles(ax):
  #Load LOFAR observations
  dates = np.load(os.path.join(data_folder, 'LOFAR_profiles_dates.npy'))
  observations = np.load(os.path.join(data_folder, 'LOFAR_profiles.npy'))
  observations = observations[dates > ref_date]
  dates = dates[dates > ref_date]

  #Average over dt
  dt = 30
  delay = dates - dates[0]
  delay = np.array( [n.days for n in delay] )
  delay /= dt
  days = np.unique( delay )
  avg = []
  for n in days:
    obs = observations[ np.where(delay==n)[0] ].sum(axis=0)
    avg.append( obs / obs.max() )
  obs_uniq = np.array(avg)

  #Plot
  days = days * dt + dt / 2
  distance = 0.0005
  for i, obs in enumerate(obs_uniq):
    y = obs / distance + days[i]
    x = (np.arange(y.size) - 258) / 512. * 538.4688219194
    ax.plot(x, y/365., 'k')
  ax.set_ylim([days_min/365., (dates[-1] - ref_date).days/365.])
  #ax.set_ylim([days_min, 0.05 / distance + days[i]])
  ax.set_xlabel('Phase (ms)')
  ax.set_xlim([-30, 80])
  ax.tick_params(axis='y', labelleft='off')
  #ax.yaxis.set_ticks(np.arange(int(days_min/365.), int((dates[-1] - ref_date).days/365.)+1, 0.5))
  return



def DM(ax):
  DM = np.load(os.path.join(data_folder, 'DM.npy'))[:, 1:]
  DM_tel = np.load(os.path.join(data_folder, 'DM_tel.npy'))[1:]

  idx = np.where(DM_tel == 'LOFAR')[0]
  d = DM[:, idx]
  idx = np.argsort(d[0])
  #ax.errorbar((d[1,idx]-43.485)*1000, (d[0,idx] - ref_mjd)/365., xerr=d[2,idx], fmt='ko', markersize=2)
  ax.plot((d[1,idx]-43.485)*1000, (d[0,idx] - ref_mjd)/365., 'ko', markersize=2)
  ax.tick_params(axis='y', labelleft='off')
  ax.set_xlabel('DM-43485\n(mpc$\cdot$cm$^{-3}$)')
  ax.ticklabel_format(useOffset=False)
  ax.locator_params(axis='x', nbins=5)

  return



def flux(ax):
  LOFAR_f = np.load(os.path.join(data_folder, 'LOFAR_FLUX.npy'))[:, 1:]

  date = np.array([(n - ref_date).days for n in LOFAR_f[0]])
  idx = np.argsort(date)
  ax.errorbar(LOFAR_f[1,idx]/1000., date[idx]/365., xerr=LOFAR_f[1,idx]/2./1000., fmt='ko-', markersize=2)
  ax.set_xlabel("Flux density\n(Jy)")
  ax.tick_params(axis='y', labelleft='off')
  ax.locator_params(axis='x', nbins=3)
  ax.set_xlim([0,2])
  return





if __name__ == '__main__':
  fig = plt.figure(figsize=(10,5))

  plot(fig)

  plt.show()

