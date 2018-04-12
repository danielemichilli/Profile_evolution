import psrchive
import numpy as np
import os
import datetime
import matplotlib.dates
import matplotlib.pyplot as plt
from glob import glob

import mjd2date

data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"

def plot(ax, multiplot=True, date_min=datetime.date(2013,01,01)):
  #Load LOFAR observations
  dates = np.load(os.path.join(data_folder, 'LOFAR_profiles_dates.npy'))
  observations = np.load(os.path.join(data_folder, 'LOFAR_profiles.npy'))
  observations = observations[dates > date_min]
  dates = dates[dates > date_min]


  '''
  #Average Observations on the same day
  date_uniq, idx_uniq, repeated = np.unique(dates, return_index=True, return_counts=True)
  obs_uniq = observations[idx_uniq]
  idx = np.where(repeated>1)[0]
  for i in idx:
    date_rep = np.where(dates == date_uniq[i])[0]
    obs_rep = np.sum(observations[date_rep], axis=0)
    obs_uniq[i] = obs_rep
  obs_uniq /= obs_uniq.max(axis=1)[:, np.newaxis]
  '''

  '''
  #Average over dt
  avg = []
  avg_date = []
  avg_num = []
  date0 = dates[0]
  dt = 30
  temp = []
  for idx,date in enumerate(dates):
    if date - date0 < datetime.timedelta(dt):
      temp.append(observations[idx])
    else:
      avg.append(np.mean(temp,axis=0))
      avg_date.append(date0+datetime.timedelta(dt/2))
      avg_num.append(len(temp))
      temp = []
      obs = observations[idx]
      date0 += datetime.timedelta(dt)
  avg.append(np.mean(temp,axis=0))
  avg_date.append(date0+datetime.timedelta(dt/2))
  avg = np.array(avg)
  avg_date = np.array(avg_date)
  dates = avg_date
  obs_uniq = avg / avg.max(axis=1)[:, np.newaxis]
  '''


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

  days = days * dt + dt / 2
  distance = 0.0005
  for i, obs in enumerate(obs_uniq):
    y = obs[220:350] / distance + days[i]
    x = (np.arange(y.size) - 258 + 220) / 512. * 538.4688219194
    ax.plot(x, y, 'k')
  ax.set_ylim([0., 0.05 / distance + days[i]])
  


  '''
  #obs_uniq = observations

  base_y = np.min([date.toordinal() for date in dates])
  scale_y = 0.0001 / np.min(np.diff(np.unique(dates))).days
  obs_max = 0
  obs_min = 1
  for idx,obs in enumerate(obs_uniq):
    obs = obs[220:420]
    x = np.arange(0,1,1./len(obs))
    obs = np.clip(obs,0,0.1)
    obs += scale_y * (float(dates[idx].toordinal()) - base_y)
    obs_max = max(obs_max, max(obs))
    obs_min = min(obs_min, min(obs))
    ax.plot(x, obs, 'k')
  #ax.set_ylim((0, .3))
  ax.set_xlabel('Phase')
  ax.set_ylabel('Date')
  #glitch_epoch = datetime.date(2011, 10, 25).toordinal()
  fig = plt.gcf()
  fig.canvas.draw()
  ax.set_yticklabels([datetime.date.fromordinal(int((day/scale_y)+base_y)).strftime('%d-%m-%Y') for day in ax.get_yticks()])
  ax.tick_params(axis='y', labelsize=6)

  '''


  return
  

def load_obs_list(template=False, date_min=datetime.date.min):
  #Load template
  template_name = '/data1/Daniele/B2217+47/ephemeris/160128_profile_template_512.std'
  archives_folder = '/data1/Daniele/B2217+47/Archives_updated'
  early_archives_folder = '/data1/Daniele/B2217+47/Archives_updated/early_obs'
  template_load = psrchive.Archive_load(template_name)
  template_load.remove_baseline()
  template = template_load.get_data().flatten()

  # Load numpy arrays containing LOFAR profiles
  date_list = []
  obs_list = []
  archive_list = glob(os.path.join(archives_folder,"*.dm.paz.clean.FTp512"))  
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
  plt.figure(figsize=(10,40))
  ax = plt.subplot()
  plot(ax, multiplot=False)
  plt.show()
  #plt.savefig('LOFAR_profile_evolution.png',bbox_inches='tight')

