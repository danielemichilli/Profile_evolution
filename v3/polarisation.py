import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
import datetime 
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from mjd2date import year
from matplotlib.ticker import ScalarFormatter 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import rcParams
rcParams['ps.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages

mpl.rc('font',size=8)
mpl.rcParams['lines.linewidth'] = .5


data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"
ref_date = datetime.date(2010, 7, 25)


def main():
  fig = plt.figure(figsize=(3.5,4))

  gsB = gridspec.GridSpec(4, 1, height_ratios=[0.1,1.,1.,1.], hspace=.1)

  ax_bar_r = plt.subplot(gsB[0])
  ax_top_r = plt.subplot(gsB[1])#, sharex=ax_mid_l)
  ax_mid_r = plt.subplot(gsB[2], sharex=ax_top_r, sharey=ax_top_r)
  ax_bot_r = plt.subplot(gsB[3], sharex=ax_top_r, sharey=ax_top_r)

  right_plot(ax_bar_r, ax_top_r, ax_mid_r, ax_bot_r)

  ax_top_r.set_xlim([(450.-517.)/1024.*538.4688219194, (632.-517.)/1024.*538.4688219194])

  fig.subplots_adjust(left=.17,right=.97,bottom=.1,top=.9)
  plt.savefig('polarisation.pdf', format='pdf', dpi=300)
  plt.show()

  return


def right_plot(ax_bar, ax1, ax2, ax3):
  #Load observations
  date = np.load(os.path.join(data_folder, 'pol_date.npy'))
  I, L, V, PA = np.load(os.path.join(data_folder, 'pol.npy'))

  #Relative intensities
  #I = (I.T / I.max(axis=1)).T
  #L = (L.T / L.max(axis=1)).T
  #V = (V.T / V.max(axis=1)).T
  
  #Absolute intensities
  L = (L.T / I.max(axis=1)).T
  V = (V.T / I.max(axis=1)).T
  I = (I.T / I.max(axis=1)).T

  #Plot image of polarization evolution
  days = (date[-1] - date[0]).days
  imgI = np.zeros((days,1024))
  imgL = np.zeros((days,1024))
  imgV = np.zeros((days,1024))

  for i,n in enumerate(imgI):
    idx = np.abs(date - date[0] - datetime.timedelta(i)).argmin()
    imgI[i] = I[idx]
    imgL[i] = L[idx]
    imgV[i] = V[idx]

  lim=0.04
  extent = [(450.-517.)/1024.*538.4688219194, (650.-517.)/1024.*538.4688219194, year(date[0]), year(date[-1])]
  ax1.imshow(np.clip(imgI[:,450:650],0.,lim),cmap='hot',origin="lower",aspect='auto',interpolation='nearest', extent=extent)
  ax2.imshow(np.clip(imgL[:,450:650],0.,lim),cmap='hot',origin="lower",aspect='auto',interpolation='nearest', extent=extent)
  ax3.imshow(np.clip(imgV[:,450:650],0.,lim),cmap='hot',origin="lower",aspect='auto',interpolation='nearest', extent=extent)

  ax2.set_ylabel("Year")
  ax3.set_xlabel("Phase (ms)") 
  ax1.tick_params(axis='both', which='both', direction='inout', labelbottom='off')
  ax2.tick_params(axis='both', which='both', direction='inout', labelbottom='off')
  ax3.tick_params(axis='both', which='both', direction='inout')

  #Color bar
  norm = mpl.colors.Normalize(vmin=0., vmax=lim*100)
  cbar = mpl.colorbar.ColorbarBase(ax_bar, ticks=np.linspace(0.,4.,9), format='%.1f', cmap='hot', norm=norm, orientation='horizontal')
  cbar.ax.minorticks_on()
  cbar.ax.xaxis.set_label_position("top")
  cbar.set_label('Flux density (% peak)')
  cbar.ax.tick_params(which='both', axis='both', labelleft='off', labelright='off', labeltop='on', labelbottom='off', right='off', left='off', bottom='off', top='on')

  ax1.ticklabel_format(useOffset=False)
  ax2.ticklabel_format(useOffset=False)
  ax3.ticklabel_format(useOffset=False)
  ax1.yaxis.set_major_locator(MultipleLocator(1))
  ax2.yaxis.set_major_locator(MultipleLocator(1))
  ax3.yaxis.set_major_locator(MultipleLocator(1))
  ax1.yaxis.set_minor_locator(MultipleLocator(.2))
  ax2.yaxis.set_minor_locator(MultipleLocator(.2))
  ax3.yaxis.set_minor_locator(MultipleLocator(.2))
  ax1.xaxis.set_major_locator(MultipleLocator(20))
  ax2.xaxis.set_major_locator(MultipleLocator(20))
  ax3.xaxis.set_major_locator(MultipleLocator(20))
  ax1.xaxis.set_minor_locator(MultipleLocator(4))
  ax2.xaxis.set_minor_locator(MultipleLocator(4))
  ax3.xaxis.set_minor_locator(MultipleLocator(4))

  def label(ax, number):
    at = AnchoredText(number, prop=dict(size=15), loc=1, frameon=True, pad=.1, borderpad=0.)
    ax.add_artist(at)
    return
  label(ax1, "(a)")
  label(ax2, "(b)")
  label(ax3, "(c)")

  return

if __name__ == '__main__':
  main()
