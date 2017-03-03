import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mjd2date import convert


data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"
ref_date = datetime.date(2010, 7, 25)
ref_mjd = 55402

def plot(fig):
  gsA = gridspec.GridSpec(1, 2, wspace=.3, width_ratios=[0.02,1], left=0.05, right=0.495, bottom=0.05, top=.95)
  ax_cb = plt.subplot(gsA[0])

  gs = gridspec.GridSpecFromSubplotSpec(1, 3, gsA[1], wspace=.1, width_ratios=[1,1,.5])
  ax1 = plt.subplot(gs[0])
  ax2 = plt.subplot(gs[1], sharex=ax1, sharey=ax1)
  ax3 = plt.subplot(gs[2], sharey=ax1)

  JB(ax_cb, ax1)
  days_max = LOFAR(ax2)
  flux(ax3)

  return days_max, ax2


def JB(ax1, ax2):
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
  #img /= np.sum(img, axis=1, keepdims=True)  #Set the total area constant

  date_max = (dates[-1] - ref_date).days
  phase_min = -353. / 512. * 538.4688219194
  phase_max = (512. - 353.) / 512. * 538.4688219194
  s = ax2.imshow(np.clip(img,0,0.15*img.max()),cmap='hot',origin="lower",aspect='auto',interpolation='nearest',extent=[phase_min, phase_max, 0, date_max])
  ax2.set_xlabel('Phase (ms)')
  ax2.set_ylabel('Days after MJD 55402 (2010 July 25)')

  cbar = plt.colorbar(s, cax=ax1)
  cbar.ax.set_yticklabels(np.linspace(0,15,11,dtype=str))
  cbar.ax.yaxis.set_label_position("left")
  cbar.set_label('Flux [% peak]')
  cbar.ax.tick_params(axis='y', labelleft='on', labelright='off', right='off', left='off')

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
  phase_min = -258. / 512. * 538.4688219194
  phase_max = (512. - 258.) / 512. * 538.4688219194
  s = ax.imshow(np.clip(img,0,0.15*img.max()),cmap='hot',origin="lower",aspect='auto',interpolation='nearest',extent=[phase_min, phase_max, 0, date_max])
  ax.set_xlabel('Phase (ms)')
  ax.tick_params(axis='y', labelleft='off', labelright='off')

  ax.set_xlim([-30, 80])
  ax.set_ylim([(dates[0] - ref_date).days, (dates[-1] - ref_date).days])
  ax.yaxis.set_ticks(range(0, (dates[-1] - ref_date).days, 200))

  return (dates[-1] - ref_date).days



def flux(ax):
  JB_f = np.load(os.path.join(data_folder, 'JB_FLUX.npy'))

  date = [(n - ref_date).days for n in JB_f[0]]
  ax.errorbar(JB_f[1], date, xerr=JB_f[2], fmt='ko-')
  ax.set_xlabel("Flux density (mJy)")
  ax.tick_params(axis='y', labelleft='off')

  return






if __name__ == '__main__':
  fig = plt.figure(figsize=(10,5))
  plot_grid = gridspec.GridSpec(1, 1)

  plot(fig, plot_grid[0])

  plt.show()

