import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
import datetime
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

mpl.rc('font',size=8)

data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"
ref_date = datetime.date(2010, 7, 25)


def main():
  fig = plt.figure(figsize=(3.3,4))
  gs = gridspec.GridSpec(1, 2, wspace=.6, left=0.2)
  #ax1 = fig.add_subplot(121)
  #ax2 = fig.add_subplot(122)
  ax1 = plt.subplot(gs[0])
  ax2 = plt.subplot(gs[1])

  LOFAR(ax1)
  Crab(ax2)

  #plt.subplots_adjust(wspace=.3)

  fig.savefig('crab_comparison.eps', papertype='a4', orientation='portrait', format='eps', dpi=200)
  #plt.show()

  return


def Crab(ax):
  im = plt.imread('crab_echo.png')

  extent=[0, 12, 0, 200]

  ax.set_xlim(extent[0:2])
  ax.set_ylim(extent[2:4]) 
  ax.imshow(im, extent=extent, aspect='auto')
  ax.set_xlabel('Phase (ms)')
  ax.set_ylabel('Days after MJD 50677 (1997 August 17)')

  return


def LOFAR(ax):
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
  phase_min = -262. / 512. * 538.4688219194
  phase_max = (512. - 262.) / 512. * 538.4688219194
  s = ax.imshow(np.clip(img,0.03,0.15),cmap='Greys',origin="lower",aspect='auto',interpolation='nearest',extent=[phase_min, phase_max, 0, date_max])
  ax.set_xlabel('Phase (ms)')
  ax.set_ylabel('Days after MJD 55402 (2010 July 25)')
  ax.set_xlim([0, 25])
  ax.set_ylim([(dates[0] - ref_date).days, (dates[-1] - ref_date).days])
  #ax.yaxis.set_ticks(range(0, (dates[-1] - ref_date).days, 200))
  #ax.yaxis.set_ticks(np.arange(0, int((dates[-1] - ref_date).days)+1, 0.5))
  return




if __name__ == '__main__':
  main()

