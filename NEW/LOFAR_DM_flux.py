import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import matplotlib.gridspec as gridspec
import numpy as np
import os

from mjd2date import convert
from mjd2date import year


data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"


def plot():
  gs = gridspec.GridSpec(1, 2, wspace=0.1)
  ax0 = plt.subplot(gs[0,0])
  ax1 = plt.subplot(gs[0,1], sharey=ax0)

  DM(ax0)
  flux(ax1)

  ax0.ticklabel_format(useOffset=False)
  ax1.ticklabel_format(useOffset=False)

  ax0.yaxis.set_major_locator(MultipleLocator(1))
  ax0.yaxis.set_minor_locator(MultipleLocator(.2))
  ax1.yaxis.set_major_locator(MultipleLocator(1))
  ax1.yaxis.set_minor_locator(MultipleLocator(.2))
  ax0.xaxis.set_major_locator(MultipleLocator(2))
  ax0.xaxis.set_minor_locator(MultipleLocator(.4))
  ax1.xaxis.set_major_locator(MultipleLocator(.5))
  ax1.xaxis.set_minor_locator(MultipleLocator(.1))

  ax0.tick_params(axis='both', which='both', direction='inout')
  ax1.tick_params(axis='both', which='both', direction='inout', labelleft='off', labelright='off')
  ax1.yaxis.set_label_position("right")

  ax0.set_ylabel('Year')
  #ax1.set_ylabel('Year')

  ax0.set_ylim([2013.3, 2016.58])

  def label(ax, number):
    at = AnchoredText(number, prop=dict(size=15), loc=1, frameon=True, pad=.1, borderpad=0.)
    ax.add_artist(at)
    return
  label(ax0, "(a)")
  label(ax1, "(b)")

  fig.subplots_adjust(left=0.16,right=.96,bottom=.15,top=.98)
  return



def DM(ax):
  DM = np.load(os.path.join(data_folder, 'DM.npy'))[:, 1:]
  DM_tel = np.load(os.path.join(data_folder, 'DM_tel.npy'))[1:]

  idx = np.where(DM_tel == 'LOFAR')[0]
  d = DM[:, idx]
  idx = np.argsort(d[0])
  date = [year(convert(n)) for n in d[0,idx]]
  ax.errorbar((d[1,idx]-43.485)*1000, date, xerr=d[2,idx]*1000, fmt='ko', markersize=2, capsize=0)
  ax.set_xlabel('DM - 43.485\n(10$^{-3}$pc cm$^{-3}$)')

  return


def flux(ax):
  LOFAR_f = np.load(os.path.join(data_folder, 'LOFAR_FLUX.npy'))[:, 1:]

  date = np.array([year(n) for n in LOFAR_f[0]])
  idx = np.argsort(date)
  ax.errorbar(LOFAR_f[1,idx]/1000., date[idx], xerr=LOFAR_f[1,idx]/2./1000., fmt='ko-', markersize=2, capsize=0)
  ax.set_xlabel("Flux density\n(Jy)")

  return



if __name__ == '__main__':
  mpl.rc('font',size=8)
  fig = plt.figure(figsize=(3.5,4))

  plot()
  plt.savefig('LOFAR_DM_flux.eps', papertype='a4', orientation='portrait', format='eps', dpi=200)

  plt.show()




