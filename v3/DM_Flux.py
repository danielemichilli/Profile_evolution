import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import matplotlib.gridspec as gridspec
import numpy as np
import os
from matplotlib import rcParams
rcParams['ps.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import datetime

from mjd2date import convert
from mjd2date import year


data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"
#data_folder = '/data1/hessels/drg21_data1_bak_20170724/Daniele/B2217+47/Analysis/plot_data'
ref_date = datetime.date(2011, 1, 1)

def plot():
  gs = gridspec.GridSpec(2, 4, height_ratios=[.03,1.], wspace=0.2, hspace=0.05)
  ax0 = plt.subplot(gs[1,0])
  ax1 = plt.subplot(gs[1,1], sharey=ax0)
  ax2 = plt.subplot(gs[1,2], sharey=ax0)
  ax3 = plt.subplot(gs[1,3], sharey=ax0)
  ax3_bar = plt.subplot(gs[0,3])

  LOFAR_DM(ax0)
  LOFAR_flux(ax1)
  JB_flux(ax2)
  JB(ax3, saturate=[0,0.15], cbar=ax3_bar)

  ax0.ticklabel_format(useOffset=False)

  ax0.yaxis.set_major_locator(MultipleLocator(1))
  ax0.yaxis.set_minor_locator(MultipleLocator(.2))
  ax0.xaxis.set_major_locator(MultipleLocator(2))
  ax0.xaxis.set_minor_locator(MultipleLocator(.4))
  ax1.xaxis.set_major_locator(MultipleLocator(.5))
  ax1.xaxis.set_minor_locator(MultipleLocator(.1))
  ax2.xaxis.set_major_locator(MultipleLocator(1))
  ax2.xaxis.set_minor_locator(MultipleLocator(.2))
  ax3.xaxis.set_major_locator(MultipleLocator(25))
  ax3.xaxis.set_minor_locator(MultipleLocator(5))

  ax0.tick_params(axis='both', which='both', direction='inout')
  ax1.tick_params(axis='both', which='both', direction='inout', labelleft='off', labelright='off')
  ax2.tick_params(axis='both', which='both', direction='inout', labelleft='off', labelright='off')
  ax3.tick_params(axis='both', which='both', direction='inout', labelleft='off', labelright='off')
  ax0.set_ylabel('Year')
  ax0.set_ylim([2013.3, 2016.58])
  ax2.set_xlim([0, 5])
  ax3_bar.set_xlabel('Flux (% peak)')
  ax3.set_xlim([-30, 80])

  def label(ax, number):
    at = AnchoredText(number, prop=dict(size=15), loc=1, frameon=True, pad=.1, borderpad=0.)
    ax.add_artist(at)
    return
  label(ax0, "(a)")
  label(ax1, "(b)")
  label(ax2, "(c)")
  label(ax3, "(d)")

  fig.subplots_adjust(left=0.08,right=.98,bottom=.15,top=.9)
  return

def LOFAR_DM(ax):
  DM = np.load(os.path.join(data_folder, 'DM_LOFAR.npy'))[:, 1:]
  #DM_tel = np.load(os.path.join(data_folder, 'DM_tel.npy'))[1:]

  #idx = np.where(DM_tel == 'LOFAR')[0]
  #d = DM[:, idx]

  d = DM

  idx = np.argsort(d[0])
  date = [year(convert(n)) for n in d[0,idx]]
  ax.errorbar((d[1,idx]-43.485)*1000, date, xerr=d[2,idx]*1000, fmt='ko', markersize=2, capsize=0)
  ax.set_xlabel('DM - 43,485\n(10$^{-3}$pc cm$^{-3}$)')

  return


def LOFAR_flux(ax):
  LOFAR_f = np.load(os.path.join(data_folder, 'LOFAR_FLUX.npy'))[:, 1:]

  date = np.array([year(n) for n in LOFAR_f[0]])
  idx = np.argsort(date)
  ax.errorbar(LOFAR_f[1,idx]/1000., date[idx], xerr=LOFAR_f[1,idx]/2./1000., fmt='ko-', markersize=2, capsize=0)
  ax.set_xlabel("Flux density\n(Jy)")

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


def JB_flux(ax):
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
  fig = plt.figure(figsize=(7,4))

  plot()
  plt.savefig('DM_flux.pdf', format='pdf', dpi=300)

  plt.show()


