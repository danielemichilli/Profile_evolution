import datetime
import psrchive
import numpy as np
import os
import matplotlib.pyplot as plt

home_folder     = '/data1/Daniele/B2217+47'
profile_template= home_folder + '/ephemeris/160128_profile_template_512.std'

date_list = []
obs_list = []
folder_list = os.listdir(".")
template = psrchive.Archive_load(profile_template).get_data().flatten()

for obs in folder_list:
  load_archive = psrchive.Archive_load(obs+"/"+obs+"_correctDM.clean.TF.b512.ar")
  prof = load_archive.get_data().flatten()
  prof -= np.median(prof)
  prof /= np.max(prof)
  bins = prof.size
  prof_ext = np.concatenate((prof[-bins/2:],prof,prof[:bins/2]))
  shift = prof.size/2 - np.correlate(np.clip(prof_ext,0.3,0.9),np.clip(template,0.3,0.9),mode='valid').argmax()
  prof = np.roll(prof,shift)
  epoch = load_archive.get_Integration(0).get_epoch()
  date = datetime.date(int(epoch.datestr('%Y')), int(epoch.datestr('%m')), int(epoch.datestr('%d')))
  date_list.append(date)
  obs_list.append(prof)

date_list = np.array(date_list)
obs_list = np.array(obs_list)

idx = np.argsort(date_list)
date_list = date_list[idx]
obs_list = obs_list[idx]
###
dt = datetime.timedelta(50)
date0 = datetime.date(2013, 5, 24)
date = [date0 + datetime.timedelta(50) * i for i in range(18)]

profiles = np.zeros((18,512))
idx = np.where(date_list>=date0)[0]
date_list = date_list[idx]
obs_list = obs_list[idx]

for i in range(18):
    s = date0 + i * dt
    e = s + dt
    temp = []
    for j,n in enumerate(date_list):
        if s <= n < e:
            temp.append(obs_list[j])
    temp = np.array(temp)
    temp = np.mean(temp,axis=0)
    profiles[i] = temp


