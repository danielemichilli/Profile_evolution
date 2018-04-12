import psrchive
from glob import glob
import numpy as np
import datetime
import os


archive_files = '/data1/Daniele/B2217+47/Archives_updated/*.FTp512'
template_file = '/data1/Daniele/B2217+47/ephemeris/160128_profile_template_512.std'
out_name = '/data1/Daniele/B2217+47/Analysis/plot_data/LOFAR_profiles'


if os.path.isfile(out_name+'.npy'): 
  print "File exists. To continue, remove ", out_name+'.npy'
  exit()

def load_ar(ar_name, template=False):
  load_archive = psrchive.Archive_load(ar_name)
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

  return date, prof


_, template = load_ar(template_file)
date_list = []
prof_list = []
ar_list = glob(archive_files)
for ar_name in ar_list:
  date, prof = load_ar(ar_name, template)
  date_list.append(date)
  prof_list.append(prof)
date = np.array(date_list)
prof = np.array(prof_list)
idx = np.argsort(date)
date = date[idx]
prof = prof[idx]

np.save(out_name, prof)
np.save(out_name+'_dates', date)


