import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from mjd2date import convert
from mjd2date import year


data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"
ref_date = datetime.date(2010, 7, 25)
ref_mjd = 55402


def plot():
  gs = gridspec.GridSpec(2, 2, height_ratios=[.03,1.], wspace=0.1, hspace=0.05)
  ax0 = plt.subplot(gs[1,0])
  ax1 = plt.subplot(gs[1,1], sharey=ax0)
  ax0_bar = plt.subplot(gs[0,0])

  JB(ax0, saturate=[0,0.15], cbar=ax0_bar)
  flux(ax1)

  ax0.ticklabel_format(useOffset=False)
  ax1.ticklabel_format(useOffset=False)

  ax0.yaxis.set_major_locator(MultipleLocator(1))
  ax0.yaxis.set_minor_locator(MultipleLocator(.2))
  ax1.yaxis.set_major_locator(MultipleLocator(1))
  ax1.yaxis.set_minor_locator(MultipleLocator(.2))
  ax0.xaxis.set_major_locator(MultipleLocator(25))
  ax0.xaxis.set_minor_locator(MultipleLocator(5))
  ax1.xaxis.set_major_locator(MultipleLocator(1))
  ax1.xaxis.set_minor_locator(MultipleLocator(.2))

  ax0.tick_params(axis='both', which='both', direction='inout')
  ax1.tick_params(axis='both', which='both', direction='inout', labelleft='off', labelright='off')
  ax1.yaxis.set_label_position("right")

  ax0.set_ylabel('Year')
  #ax1.set_ylabel('Year')
  ax0_bar.set_xlabel('Flux (% peak)')

  ax0.set_xlim([-30, 80])
  ax1.set_xlim([0, 5])

  def label(ax, number):
    at = AnchoredText(number, prop=dict(size=15), loc=1, frameon=True, pad=.1, borderpad=0.)
    ax.add_artist(at)
    return
  label(ax0, "(a)")
  label(ax1, "(b)")

  fig.subplots_adjust(left=0.16,right=.96,bottom=.15,top=.9)

  return


def colorbar(ax, saturate=[0,1], cmap='hot', log=False):
  saturate[0] *= 100
  saturate[1] *= 100
  if log: norm = mpl.colors.LogNorm(vmin=saturate[0], vmax=saturate[1])
  else: norm = mpl.colors.Normalize(vmin=saturate[0], vmax=saturate[1])
  cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal', format='%d', ticks=np.linspace(saturate[0],saturate[1],9))
  cbar.ax.minorticks_on()
  cbar.ax.xaxis.set_label_position("top")
  cbar.ax.tick_params(axis='both', labelleft='off', labelright='off', labeltop='on', labelbottom='off', right='off', left='off', bottom='off', top='on', which='both')
  return


def flux(ax):
  JB_f = np.load(os.path.join(data_folder, 'JB_FLUX.npy'))

  date = [year(n) for n in JB_f[0]]
  ax.errorbar(JB_f[1], date, xerr=JB_f[2]/2., fmt='ko-', markersize=3, capsize=0)
  ax.set_xlabel("Flux density\n(mJy)")
  ax.locator_params(axis='x', nbins=5)
  ax.set_ylim([min(date), max(date)])
  return



def JB(ax, saturate=[-np.inf,np.inf], fit=False, cmap='hot', line=False, log=False, cbar=False):
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

  date_max = (dates[-1] - ref_date).days
  phase_min = -353. / 512. * 538.4688219194
  phase_max = (512. - 353.) / 512. * 538.4688219194
  s = ax.imshow(np.clip(img,0,0.15*img.max()),cmap='hot',origin="lower",aspect='auto',interpolation='nearest',extent=[phase_min, phase_max, year(dates[0]), year(dates[-1])])
  ax.set_xlabel('Phase\n(ms)')
  #ax.set_xticks([-25,0,25,50])

  if cbar: colorbar(cbar, saturate=saturate, cmap=cmap, log=log)
  return 



if __name__ == '__main__':
  mpl.rc('font',size=8)
  fig = plt.figure(figsize=(3.5,4))

  plot()
  plt.savefig('Lovell.eps', papertype='a4', orientation='portrait', format='eps', dpi=200)

  plt.show()


