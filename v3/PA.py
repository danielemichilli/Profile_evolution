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
  fig = plt.figure(figsize=(3.5,7))

  gsA = gridspec.GridSpec(3, 1, height_ratios=[0.05,.6,1], hspace=.05)

  ax_bar_l = plt.subplot(gsA[0])
  ax_top_l = plt.subplot(gsA[1])
  ax_mid_l = plt.subplot(gsA[2], sharex=ax_top_l)

  left_plot(ax_bar_l, ax_top_l, ax_mid_l)

  ax_top_l.set_xlim([-15,15])

  def label(ax, number):
    at = AnchoredText(number, prop=dict(size=15), loc=1, frameon=True, pad=.1, borderpad=0.)
    ax.add_artist(at)
    return
  label(ax_top_l, "(a)")
  label(ax_mid_l, "(b)")

  fig.subplots_adjust(left=.15, right=.98, bottom=.08, top=.95)
  plt.savefig('PA.pdf', format='pdf', dpi=300)
  plt.show()

  return



def left_plot(ax_bar, ax_top, ax_mid):
  #Load observations
  date = np.load(os.path.join(data_folder, 'pol_date.npy'))
  I, L, V, PA = np.load(os.path.join(data_folder, 'pol.npy'))

  #Relative intensities
  m = np.max(I,axis=1)
  I = (I.T / m).T
  L = (L.T / m).T
  V = (V.T / m).T

  #Middle plot
  d = date[0:] - date[1]
  days = [n.days for n in d]
  c = [float(n)/max(days) for n in days]
  colors = iter(cm.jet(c))
  x = np.linspace(-517./1024.*538.4688219194, (1024.-517.)/1024.*538.4688219194, 1024)
  for i in range(I.shape[0]):
    col = next(colors)
    ax_mid.plot(x, I[i]*100, color=col)
    ax_mid.plot(x, L[i]*100, color=col)#, linestyle='--')
    #ax_mid.plot(x, V[i]*100, color=col)#, linestyle=':')
  ax_mid.set_ylabel('Flux density (% peak)')
  ax_mid.tick_params(axis='both', which='both', direction='inout')
  ax_mid.set_ylim([-0.5, 28])
  ax_mid.set_xlabel("Phase (ms)")
  ax_mid.yaxis.set_major_locator(MultipleLocator(10))
  ax_mid.yaxis.set_minor_locator(MultipleLocator(2))

  start = 479.
  duration = 70.
  PA_n = PA[:,int(start):int(start+duration)]  #Select pahses based on pav output
  x = (np.arange(duration) + start - 517.) / 1024. * 538.4688219194

  #Top plot
  colors = iter(cm.jet(c))
  for i,n in enumerate(PA_n):
    col = next(colors)
    n -= n[int(duration/2)]
    np.mod(n - 50 + 180, 180., out=n)
    ax_top.plot(x, n, 'o',color=col,label=str(days[i]), markersize=2, markeredgewidth=0.)
  ax_top.set_ylabel('PA (deg)')
  #ax_top.axvline((45.5+479.-517.)/1024.*538.4688219194,color='k')
  #ax_top.axvline((56.5+479.-517.)/1024.*538.4688219194,color='k')
  ax_top.set_ylabel('PA (deg)')
  ax_top.tick_params(axis='both', which='both', direction='inout', labelbottom='off')

  ax_top.locator_params(axis='y', nbins=5)
  ax_top.set_ylim([0,180])
 
  #Color bar
  norm = mpl.colors.Normalize(vmin=year(date[1]), vmax=year(date[-1]))
  cbar = mpl.colorbar.ColorbarBase(ax_bar, ticks=[2014.022,2014.5,2015, 2015.5], format='%.1f', cmap='jet', norm=norm, orientation='horizontal')
  cbar.ax.minorticks_on()
  cbar.ax.xaxis.set_label_position("top")
  cbar.set_label('Year')
  cbar.ax.tick_params(axis='both', labelleft='off', labelright='off', labeltop='on', labelbottom='off', right='off', left='off', bottom='off', top='on', which='both')

  #ax_top.get_yaxis().set_label_coords(-0.12,0.5)

  return



if __name__ == '__main__':
  main()
