import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.interpolate import interp2d
from matplotlib.backends.backend_pdf import PdfPages

from mjd2date import convert
from mjd2date import year


data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"
ref_date = datetime.date(2011, 3, 1)


def plot():
  gs = gridspec.GridSpec(2, 2, height_ratios=[.02,1.], wspace=0.06, hspace=0.02)
  ax0 = plt.subplot(gs[1,0])
  ax1 = plt.subplot(gs[1,1], sharex=ax0, sharey=ax0)
  ax0_bar = plt.subplot(gs[0,0])

  image(ax0, saturate=[0,0.15], line=True, cbar=ax0_bar)
  profiles(ax1)

  ax0.ticklabel_format(useOffset=False)
  ax1.ticklabel_format(useOffset=False)

  ax0.yaxis.set_major_locator(MultipleLocator(1))
  ax0.yaxis.set_minor_locator(MultipleLocator(.2))
  ax1.yaxis.set_major_locator(MultipleLocator(1))
  ax1.yaxis.set_minor_locator(MultipleLocator(.2))
  ax0.xaxis.set_major_locator(MultipleLocator(20))
  ax0.xaxis.set_minor_locator(MultipleLocator(4))
  ax1.xaxis.set_major_locator(MultipleLocator(20))
  ax1.xaxis.set_minor_locator(MultipleLocator(4))

  ax0.tick_params(axis='both', which='both', direction='inout')
  ax1.tick_params(axis='both', which='both', direction='inout', labelleft='off')

  ax0.set_ylabel('Year')
  ax0_bar.set_xlabel('Flux (% peak)')

  ax0.set_xlim([-30, 60])

  def label(ax, number):
    at = AnchoredText(number, prop=dict(size=15), loc=1, frameon=True, pad=.1, borderpad=0.)
    ax.add_artist(at)
    return
  label(ax0, "(a)")
  label(ax1, "(b)")

  fig.subplots_adjust(left=0.1,right=.98,bottom=.05,top=.95)

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


def profiles(ax):
  #Load LOFAR observations
  dates = np.load(os.path.join(data_folder, 'LOFAR_profiles_dates.npy'))
  observations = np.load(os.path.join(data_folder, 'LOFAR_profiles.npy'))

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

  SNR = np.max(obs_uniq / np.std(obs_uniq[:, :200], axis=1)[:,np.newaxis], axis=1)
  idx = np.where(SNR > 1000)[0]
  days = days[idx]
  obs_uniq = obs_uniq[idx]

  #Plot
  days = days * dt + dt / 2
  distance = 0.06
  #ref_date = dates.min()
  for i, obs in enumerate(obs_uniq):
    y = obs / distance + year(dates.min()+datetime.timedelta(days[i]))
    x = (np.arange(y.size) - 258) / 512. * 538.4688219194
    if i == 0: ax.plot(x, y, 'm', lw=1.5, zorder=3)
    else: ax.plot(x, y, 'k', lw=1.5)
  ax.set_xlabel('Phase (ms)')

  return


def image(ax, saturate=[-np.inf,np.inf], cmap='hot', line=False, log=False, cbar=False, interpolate=False):
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

  #ref_date = date_uniq.min()

  #Create an image of the profile evolution calibrated in time
  if interpolate:
    obs_uniq -= np.median(obs_uniq, axis=1, keepdims=True)
    obs_uniq /= np.max(obs_uniq, axis=1, keepdims=True)
    days = np.array([(n  - ref_date).days for n in date_uniq])
    prof_arg = np.arange(obs_uniq.shape[1])
    img_f = interp2d(prof_arg, days, obs_uniq, kind='linear')
    img = img_f(prof_arg, np.arange(days.max()))
  else:
    days = (dates[-1] - ref_date).days
    img = np.zeros((days,512))
    for i,n in enumerate(img):
      idx = np.abs(date_uniq - ref_date - datetime.timedelta(i)).argmin()
      img[i] = obs_uniq[idx]

  img -= np.median(img, axis=1, keepdims=True)
  img /= np.max(img, axis=1, keepdims=True)

  date_max = (dates[-1] - ref_date).days
  phase_min = -258. / 512. * 538.4688219194
  phase_max = (512. - 258.) / 512. * 538.4688219194
  img = np.clip(img,saturate[0],saturate[1])
  if log: img = np.log(img)
  s = ax.imshow(img,cmap=cmap,origin="lower",aspect='auto',interpolation='nearest',extent=[phase_min, phase_max, year(ref_date), year(dates[-1])])
  ax.set_xlabel('Phase (ms)')

  #if line: ax.axhline(year(convert(55859.)), c='m', lw=2.)
  ax.set_ylim([year(ref_date), year(dates[-1])])

  if cbar: colorbar(cbar, saturate=saturate, cmap=cmap, log=log)
  return


if __name__ == '__main__':
  mpl.rc('font',size=8)
  fig = plt.figure(figsize=(7,8))

  plot()
  plt.savefig('LOFAR_profiles.pdf', format='pdf', dpi=300)


  plt.show()







