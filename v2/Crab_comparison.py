import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams['ps.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages

import LOFAR_profiles


data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"



def plot():
  gs = gridspec.GridSpec(1, 2, wspace=.1)
  ax0 = plt.subplot(gs[0,0])
  ax1 = plt.subplot(gs[0,1])

  LOFAR_profiles.image(ax0, saturate=[0.05,0.15], cmap='Greys', log=True, interpolate=True)
  Crab(ax1)

  ax0.ticklabel_format(useOffset=False)
  ax1.ticklabel_format(useOffset=False)

  
  ax0.yaxis.set_major_locator(MultipleLocator(1))
  ax0.yaxis.set_minor_locator(MultipleLocator(.2))
  ax1.yaxis.set_major_locator(MultipleLocator(.1))
  ax1.yaxis.set_minor_locator(MultipleLocator(.02))
  ax0.xaxis.set_major_locator(MultipleLocator(5))
  ax0.xaxis.set_minor_locator(MultipleLocator(1))
  ax1.xaxis.set_major_locator(MultipleLocator(3))
  ax1.xaxis.set_minor_locator(MultipleLocator(3./5))
  
  ax0.tick_params(axis='both', which='both', direction='inout')
  ax1.tick_params(axis='both', which='both', direction='inout', labelleft='off', labelright='on')
  ax1.yaxis.set_label_position("right")

  ax0.set_ylabel('Year')
  ax1.set_ylabel('Year')

  #ax0.set_ylim([2011.5, 2016.5])
  ax0.set_ylim([2011.81, 2015.8])
  ax0.set_xlim([0, 25])

  def label(ax, number):
    at = AnchoredText(number, prop=dict(size=15), loc=1, frameon=True, pad=.1, borderpad=0.)
    ax.add_artist(at)
    return
  label(ax0, "(a)")
  label(ax1, "(b)")

  fig.subplots_adjust(left=0.17,right=.81,bottom=.12,top=.99)
  return


def Crab(ax):
  im = plt.imread(os.path.join(data_folder, 'crab_echo.png'))

  extent = [0, 12, 1997.6247, 1998.1726]

  ax.set_xlim(extent[:2])
  ax.set_ylim(extent[2:])
  ax.imshow(im, extent=extent, aspect='auto')
  ax.set_xlabel('Phase (ms)')
  ax.set_ylabel('Year')
  ax.ticklabel_format(useOffset=False)

  return



if __name__ == '__main__':
  mpl.rc('font',size=8)
  fig = plt.figure(figsize=(3.5,4))

  plot()
  #plt.savefig('Crab_comparison.eps', papertype='a4', orientation='portrait', format='eps', dpi=200)

  
  pp = PdfPages('Crab_comparison.pdf')
  pp.savefig(fig, papertype='a4', orientation='portrait', dpi=200)
  pp.close()


  plt.show()

