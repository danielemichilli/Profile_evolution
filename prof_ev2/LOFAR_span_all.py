import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from mjd2date import convert
from mjd2date import year
from LOFAR_new import profiles

data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"
ref_date = datetime.date(2011, 3, 13)
#ref_date = datetime.date(2010, 7, 25)

def plot(grid):
  gs = gridspec.GridSpecFromSubplotSpec(1, 4, grid, wspace=.15, width_ratios=[.5,.5,1.,1.])
  ax1 = plt.subplot(gs[1])
  ax2 = plt.subplot(gs[2], sharex=ax1, sharey=ax1)
  ax3 = plt.subplot(gs[0], sharey=ax1)
  ax4 = plt.subplot(gs[3], sharex=ax1, sharey=ax1)

  scale = JB(ax1)
  LOFAR(ax2)
  flux(ax3)
  profiles(ax4)

  def label(ax, number):
    at = AnchoredText(number, prop=dict(size=15), loc=2, frameon=True, pad=.1, borderpad=0.)
    ax.add_artist(at)
    return
  label(ax3, "(a)")
  label(ax1, "(b)")
  label(ax2, "(c)")
  label(ax4, "(d)")

  ax1.set_title('Lovell')
  ax2.set_title('LOFAR')
  ax3.set_title('Lovell')
  ax4.set_title('LOFAR')
  ax4.ticklabel_format(useOffset=False)
  ax4.yaxis.set_major_locator(MultipleLocator(1))
  ax4.yaxis.set_minor_locator(MultipleLocator(.2))

  #ax3.set_ylabel('Years after MJD 55402 (2010 July 25)')
  ax3.set_ylabel('Year')
  ax3.tick_params(axis='y', labelleft='on')

  ax1.ticklabel_format(useOffset=False)
  ax2.ticklabel_format(useOffset=False)
  ax3.ticklabel_format(useOffset=False)

  ax1.yaxis.set_major_locator(MultipleLocator(1))
  ax1.yaxis.set_minor_locator(MultipleLocator(.2))
  ax2.yaxis.set_major_locator(MultipleLocator(1))
  ax2.yaxis.set_minor_locator(MultipleLocator(.2))
  ax3.yaxis.set_major_locator(MultipleLocator(1))
  ax3.yaxis.set_minor_locator(MultipleLocator(.2))


  return ax2, scale


def JB(ax):
  #Load JB observations
  dates = np.load(os.path.join(data_folder, 'JB_profiles_dates.npy'))
  observations = np.load(os.path.join(data_folder, 'JB_profiles.npy'))

  #Average Observations on the same day
  date_uniq, idx_uniq, repeated = np.unique(dates, return_index=True, return_counts=True)
  obs_uniq = observations[idx_uniq]
  idx = np.where(repeated>1)[0]
  for i in idx:
    date_rep = np.where(dates == date_uniq[i])[0]
    obs_rep = np.sum(observations[date_rep], axis=0)
    obs_uniq[i] = obs_rep
  obs_uniq /= obs_uniq.max(axis=1)[:, np.newaxis]

  #Create an image of the profile evolution calibrated in time
  days = (dates[-1] - dates[0]).days
  img = np.zeros((days,512))
  for i,n in enumerate(img):
    idx = np.abs(date_uniq - date_uniq[0] - datetime.timedelta(i)).argmin()
    img[i] = obs_uniq[idx]

  img -= np.median(img, axis=1, keepdims=True)
  img /= np.max(img, axis=1, keepdims=True)
  #img /= np.sum(img, axis=1, keepdims=True)  #Set the total area constant

  date_max = (dates[-1] - ref_date).days
  phase_min = -353. / 512. * 538.4688219194
  phase_max = (512. - 353.) / 512. * 538.4688219194
  s = ax.imshow(np.clip(img,0,0.15*img.max()),cmap='hot',origin="lower",aspect='auto',interpolation='nearest',extent=[phase_min, phase_max, year(dates[0]), year(dates[-1])])
  ax.set_xlabel('Phase\n(ms)')
  ax.tick_params(axis='y', labelleft='off')
  #ax.locator_params(axis='x', nticks=3)
  #ax.set_xticks(range(-20,61,20))
  ax.set_xticks([-25,0,25,50])
  return s


def LOFAR(ax):
  #Load LOFAR observations
  dates = np.load(os.path.join(data_folder, 'LOFAR_profiles_dates.npy'))
  observations = np.load(os.path.join(data_folder, 'LOFAR_profiles.npy'))
  observations = observations[dates >= ref_date]
  dates = dates[dates >= ref_date]
 
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
  #img /= np.sum(img, axis=1, keepdims=True)  #Set the total area constant

  date_max = (dates[-1] - ref_date).days
  phase_min = -258. / 512. * 538.4688219194
  phase_max = (512. - 258.) / 512. * 538.4688219194
  s = ax.imshow(np.clip(img,0,0.15*img.max()),cmap='hot',origin="lower",aspect='auto',interpolation='nearest',extent=[phase_min, phase_max, year(dates[0]), year(dates[-1])])
  ax.set_xlabel('Phase (ms)')
  ax.tick_params(axis='y', labelleft='off')

  ax.axhline(year(convert(55859.)), c='m')

  ax.set_xlim([-30, 80])
  ax.set_ylim([year(dates[0]), year(dates[-1])])
  #ax.set_ylim([(dates[0] - ref_date).days/365., (dates[-1] - ref_date).days/365.])
  #ax.yaxis.set_ticks(range(0, (dates[-1] - ref_date).days, 200))
  #ax.yaxis.set_ticks(np.arange(0, int((dates[-1] - ref_date).days/365.)+1, 0.5))
  return 



def flux(ax):
  JB_f = np.load(os.path.join(data_folder, 'JB_FLUX.npy'))

  date = [year(n) for n in JB_f[0]]
  ax.errorbar(JB_f[1], date, xerr=JB_f[2]/2., fmt='ko-', markersize=2, capsize=0)
  ax.set_xlabel("Flux density\n(mJy)")
  ax.tick_params(axis='y', labelleft='off')
  ax.locator_params(axis='x', nbins=5)
  return






if __name__ == '__main__':
  fig = plt.figure(figsize=(10,5))
  plot_grid = gridspec.GridSpec(1, 1)

  plot(fig, plot_grid[0])

  plt.show()

