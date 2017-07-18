import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
import datetime
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import time

mpl.rc('font',size=8)

data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"
ref_date = datetime.date(2010, 7, 25)


def main():
  fig = plt.figure(figsize=(3.3,4))
  gs = gridspec.GridSpec(1, 2, wspace=1., left=0.2)
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

  extent = [0, 12, 1997.6247, 1998.1726]

  ax.set_xlim(extent[:2])
  ax.set_ylim(extent[2:])
  ax.imshow(im, extent=extent, aspect='auto')
  ax.set_xlabel('Phase (ms)')
  ax.set_ylabel('Year')
  ax.ticklabel_format(useOffset=False)

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
  #s = ax.imshow(np.clip(img,0.03,0.15),cmap='Greys',origin="lower",aspect='auto',interpolation='nearest',extent=[phase_min, phase_max, 0, date_max])
  ax.imshow(np.clip(img,0.03,0.15),cmap='Greys',origin="lower",aspect='auto',interpolation='nearest',extent=[phase_min, phase_max, year(dates[0]), year(dates[-1])])
  ax.set_xlabel('Phase (ms)')
  ax.set_ylabel('Year')
  #ax.set_ylabel('Days after MJD 55402 (2010 July 25)')
  ax.set_xlim([0, 25])
  #ax.set_ylim([(dates[0] - ref_date).days, (dates[-1] - ref_date).days])
  #ax.yaxis.set_ticks(range(0, (dates[-1] - ref_date).days, 200))
  #ax.yaxis.set_ticks(np.arange(0, int((dates[-1] - ref_date).days)+1, 0.5))
  ax.set_ylim([year(dates[0]),year(dates[-1])])
  ax.ticklabel_format(useOffset=False)

  idx = np.where(np.std(observations[:,:200], axis=1) < 0.003145)[0]
  fit(ax,dates[idx],observations[idx])
  return


def year(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime.datetime(year=year, month=1, day=1)
    startOfNextYear = datetime.datetime(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

def fit(ax,dates,observations):
  p = np.zeros_like(dates)
  for i,day in enumerate(observations):
    p[i] = (np.where(day>0.05)[0].max())
  x = (p - 262.)/ 512. * 538.4688219194
  y = np.array([year(d) for d in dates])
  idx = np.where((y>2011.3)&(y<2015.9))[0]
  x2 = x[idx]
  y2 = y[idx]
  par_fit = np.polyfit(y2, x2, 2)
  lin_fit = np.polyfit(y2, x2, 1)
  ax.errorbar(x,y,fmt='go',xerr=.5,ms=.5, lw=.5, color='g')
  ax.plot(np.poly1d(par_fit)(y2), y2, 'r', lw=.5)
  return



if __name__ == '__main__':
  main()

