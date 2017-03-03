import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mjd2date import convert


data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"
ref_date = datetime.date(2010, 7, 25)
ref_mjd = 55402

days_min = 1000.

def plot(fig, days_max=0., ax_ref=False):
  ratio = 0.95 - (days_max - days_min) / days_max * 0.9
  gs = gridspec.GridSpec(1, 3, wspace=.1, left=.505, right=.95, bottom=0.44198606271776997, top=.95)
  ax1 = plt.subplot(gs[0])
  ax2 = plt.subplot(gs[1], sharey=ax1)
  ax3 = plt.subplot(gs[2], sharey=ax1, sharex=ax_ref)

  profiles(ax3, days_max)
  DM(ax2)
  flux(ax1)

  return ratio



def profiles(ax, days_max):
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
    ax.plot(x, y, 'k')
  ax.set_ylim([days_min, days_max])
  #ax.set_ylim([days_min, 0.05 / distance + days[i]])
  ax.set_xlabel('Phase (ms)')
  ax.set_xlim([-30, 80])
  ax.set_ylabel('Days after MJD 55402 (2010 July 25)')
  ax.tick_params(axis='y', labelleft='off', labelright='on')
  ax.yaxis.set_label_position("right")
  return



def DM(ax):
  DM = np.load(os.path.join(data_folder, 'DM.npy'))
  DM_tel = np.load(os.path.join(data_folder, 'DM_tel.npy'))

  idx = np.where(DM_tel == 'LOFAR')[0]
  d = DM[:, idx]
  idx = np.argsort(d[0])
  ax.errorbar(d[1,idx], (d[0,idx] - ref_mjd), xerr=d[2,idx], fmt='go-')
  ax.tick_params(axis='y', labelleft='off')
  ax.set_xlabel('DM (pc$\cdot$cm$^{-3}$)')
  ax.ticklabel_format(useOffset=False)

  return



def flux(ax):
  LOFAR_f = np.load(os.path.join(data_folder, 'LOFAR_FLUX.npy'))

  date = np.array([(n - ref_date).days for n in LOFAR_f[0]])
  idx = np.argsort(date)
  ax.errorbar(LOFAR_f[1,idx], date[idx], xerr=LOFAR_f[1,idx]/2., fmt='go-')
  ax.set_xlabel("Flux density (mJy)")
  ax.tick_params(axis='y', labelleft='off')

  return





if __name__ == '__main__':
  fig = plt.figure(figsize=(10,5))

  plot(fig)

  plt.show()

