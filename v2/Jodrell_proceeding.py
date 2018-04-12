import psrchive
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
ref_date = datetime.date(2011, 1, 1)


def plot():
  gs = gridspec.GridSpec(1, 2, width_ratios=[3.,1.], wspace=0.1)
  ax0 = plt.subplot(gs[0])
  ax2 = plt.subplot(gs[1])
  #ax0_bar = plt.subplot(gs[1])

  image(ax0, saturate=[0,0.15], line=True)
  sp_variab(ax2)

  """
  ax0.ticklabel_format(useOffset=False)
  ax2.ticklabel_format(useOffset=False)

  ax0.yaxis.set_major_locator(MultipleLocator(1))
  ax0.yaxis.set_minor_locator(MultipleLocator(.2))
  ax2.yaxis.set_major_locator(MultipleLocator(1))
  ax2.yaxis.set_minor_locator(MultipleLocator(.2))
  ax0.xaxis.set_major_locator(MultipleLocator(20))
  ax0.xaxis.set_minor_locator(MultipleLocator(4))
  ax2.xaxis.set_major_locator(MultipleLocator(5))
  ax2.xaxis.set_minor_locator(MultipleLocator(1))

  ax0.tick_params(axis='both', which='both', direction='inout')
  ax2.tick_params(axis='both', which='both', direction='inout', labelleft='off', labelright='on')
  ax2.yaxis.set_label_position("right")

  ax0.set_ylabel('Year')
  ax2.set_ylabel('Year')
  ax0_bar.set_xlabel('Flux (% peak)')
  ax2_bar.set_xlabel('Flux (% peak)')

  ax0.set_xlim([-30, 80])
  ax2.set_xlim([0, 25])
  """

  ax0.set_ylim([-20, 50])
  ax0.ticklabel_format(useOffset=False)
  ax0.set_xlabel('Year')
  ax0.tick_params(axis='both', which='both', direction='inout')
  ax2.tick_params(axis='both', which='both', direction='inout', labelleft='off', labelright='on')
  ax2.yaxis.set_label_position("right")
  ax2.xaxis.set_major_locator(MultipleLocator(.5))
  ax2.yaxis.set_major_locator(MultipleLocator(.03))

  def label(ax, number):
    at = AnchoredText(number, prop=dict(size=15), loc=2, frameon=True, pad=.1, borderpad=0.)
    ax.add_artist(at)
    return
  label(ax0, "(a)")
  label(ax2, "(b)")

  fig.subplots_adjust(left=0.1,right=.9,bottom=.15,top=.95)
  return


def colorbar(ax, saturate=[0,1], cmap='hot', log=False):
  saturate[0] *= 100
  saturate[1] *= 100
  if log: norm = mpl.colors.LogNorm(vmin=saturate[0], vmax=saturate[1])
  else: norm = mpl.colors.Normalize(vmin=saturate[0], vmax=saturate[1])
  cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical', format='%d', ticks=np.linspace(saturate[0],saturate[1],9))
  cbar.ax.minorticks_on()
  cbar.ax.xaxis.set_label_position("top")
  cbar.ax.tick_params(axis='both', labelleft='off', labelright='on', labeltop='off', labelbottom='off', right='on', left='off', bottom='off', top='off', which='both')
  return

def image(ax, saturate=[-np.inf,np.inf], fit=False, cmap='hot', line=False, log=False, cbar=False, interpolate=False):
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
  s = ax.imshow(img.T,cmap=cmap,origin="lower",aspect='auto',interpolation='nearest',extent=[year(ref_date), year(dates[-1]), phase_min, phase_max])
  ax.set_ylabel('Phase (ms)')

  if line: ax.axvline(year(convert(55859.)), c='m')
  ax.set_xlim([year(ref_date), year(dates[-1])])

  if fit:
    #idx = np.where(np.std(observations[:,:200], axis=1) < 0.003145)[0]
    #fit_parabola(ax,dates[idx],observations[idx])
    #fit_parabola(ax,dates,observations)
    obs_uniq -= np.median(obs_uniq, axis=1, keepdims=True)
    obs_uniq /= np.max(obs_uniq, axis=1, keepdims=True)
    fit_parabola(ax,date_uniq,obs_uniq)

  if cbar: colorbar(cbar, saturate=saturate, cmap=cmap, log=log)
  return




def sp_variab(ax):
  ar_name = '/data1/Daniele/B2217+47/Analysis/sp/L32532_sp.F'
  load_archive = psrchive.Archive_load(ar_name)
  load_archive.remove_baseline()
  prof = load_archive.get_data().squeeze()
  w = load_archive.get_weights()
  prof *= w
  prof = prof[:prof.shape[0]-120]
  params = [0,800,887,925,951]

  prof -= np.mean(prof[:, params[0] : params[1]])
  prof /= np.std(prof[:, params[0] : params[1]])
  err_bin = np.std(prof[:, params[0] : params[1]], axis=1)

  mp = prof[:, params[2] : params[3]].sum(axis=1)
  pc = prof[:, params[3] : params[4]].sum(axis=1)
  err_mp = err_bin * np.sqrt(params[3]-params[2])
  err_pc = err_bin * np.sqrt(params[4]-params[3])

  err_mp /= mp.max()
  err_pc /= mp.max()
  pc = pc/mp.max()
  mp = mp/mp.max()

  #mp = np.roll(mp,1)

  #idx = np.argsort(mp)
  #mp = mp[idx]
  #pc = pc[idx]
  #err_mp = err_mp[idx]
  #err_pc = err_pc[idx]

  ax.errorbar(mp,pc,fmt='ko',xerr=err_mp/2.,yerr=err_pc/2., ms=0, elinewidth=0.5, capsize=0.5)
  x = np.linspace(0,1,1000)
  #ax.plot(x, np.poly1d(np.polyfit(mp, pc, 2))(x),'r-')
  ax.plot([0,1],np.poly1d(np.polyfit(mp, pc, 1))([0,1]),'-',c='lightgreen')
  ax.plot([0,1], [0,0], 'r', lw=.5)

  ax.set_xlim([0,1])
  ax.set_ylim([-0.015,0.14])
  ax.set_xlabel('Main peak flux density (rel.)')
  ax.set_ylabel('Transient component flux density (rel.)')

  return

if __name__ == '__main__':
  mpl.rc('font',size=8)
  fig = plt.figure(figsize=(5.33,5.33/2))

  plot()

  plt.savefig('proceeding.pdf', format='pdf', dpi=300)

  plt.show()







