import psrchive
import numpy as np
import os
import datetime
import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib as mpl
from glob import glob

import mjd2date

mpl.rc('font',size=30)

template_name = '/data1/Daniele/B2217+47/ephemeris/160128_profile_template_512.std'
archives_folder = '/data1/Daniele/B2217+47/Archives_updated'
early_archives_folder = '/data1/Daniele/B2217+47/Archives_updated/early_obs'

def plot(ax, multiplot=True, date_min=datetime.date.min):
  #Load template
  template_load = psrchive.Archive_load(template_name)
  template_load.remove_baseline()
  template = template_load.get_data().flatten()

  #Load LOFAR observations
  dates, observations = load_obs_list(template=template, date_min=date_min)
  print "Dates loaded: {} - {}".format(dates[0], dates[-1])

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
  #img /= np.max(img, axis=1, keepdims=True)
  img /= np.sum(img, axis=1, keepdims=True)  #Set the total area constant
  img = img[:,220:320]

  date_min = matplotlib.dates.date2num(dates[0])
  date_max = matplotlib.dates.date2num(dates[-1])
  if multiplot:
    ax.imshow(np.clip(img.T,0,0.05*img.max()),cmap='hot',origin="lower",aspect='auto',interpolation='nearest',extent=[date_min, date_max, 0, img.shape[1]/512.*100])
    ax.xaxis_date()
    ax.set_ylabel('Phase [%]')
  else:
    s = ax.imshow(np.clip(img,0,0.15*img.max()),cmap='hot',origin="lower",aspect='auto',interpolation='nearest',extent=[0, img.shape[1]/512.*100, date_min, date_max])
    ax.yaxis_date()
    cbar = plt.colorbar(s)
    cbar.ax.set_yticklabels(np.linspace(0,15,11,dtype=str))
    cbar.set_label('Flux [% peak]')
    ax.set_xlabel('Phase [%]')
    ax.set_ylabel('Date')
    ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d:%m:%y'))

  return
  

def load_obs_list(template=False, date_min=datetime.date.min):
  # Load numpy arrays containing LOFAR profiles
  date_list = []
  obs_list = []
  archive_list = glob(os.path.join(archives_folder,"*.FTp512"))  
  archive_list.extend(glob(os.path.join(early_archives_folder,"*.pfd.bestprof")))
  for archive in archive_list:
    date, prof = load_archive(archive, template=template)
    if date > date_min:
      date_list.append(date)
      obs_list.append(prof)
  date_list = np.array(date_list)
  obs_list = np.array(obs_list)
  idx = np.argsort(date_list)
  date_list = date_list[idx]
  obs_list = obs_list[idx]
  return date_list, obs_list


def load_archive(archive,template=False):
  # Load single LOFAR profiles
  if archive.endswith(".bestprof"):
    # Early observations
    with open(archive) as f:
      lines = f.readlines()
      mjd = float(lines[3].split('=')[-1])
      date = mjd2date.convert(mjd)
    prof = np.loadtxt(archive,usecols=[1,])
    prof = np.sum(np.reshape(prof,(512,2)),axis=1)
    prof -= np.median(prof)
  else:
    load_archive = psrchive.Archive_load(archive)
    load_archive.remove_baseline()
    epoch = load_archive.get_Integration(0).get_epoch()
    date = datetime.date(int(epoch.datestr('%Y')), int(epoch.datestr('%m')), int(epoch.datestr('%d')))
    prof = load_archive.get_data().squeeze()
  prof /= np.max(prof)
  if isinstance(template,np.ndarray):
    bins = prof.size
    prof_ext = np.concatenate((prof[-bins/2:],prof,prof[:bins/2]))
    shift = bins/2 - np.correlate(prof_ext,template,mode='valid').argmax()
    prof = np.roll(prof,shift)
  else: prof = np.roll(prof,(len(prof)-np.argmax(prof))+len(prof)/2)
  return date, prof


if __name__ == '__main__':
  plt.figure(figsize=(20,40))
  ax = plt.subplot()
  plot(ax, multiplot=False)
  plt.savefig('LOFAR_profile_evolution.png',bbox_inches='tight')
  plt.show()



