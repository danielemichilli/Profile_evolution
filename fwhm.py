import numpy as np
import psrchive
import os
import matplotlib.pyplot as plt
import datetime

home_folder     = '/data1/Daniele/B2217+47'
product_folder  = home_folder + '/Products'


def width_fwhm():
  date_list, obs_list, freq_list = plot_lists()
  fwhm_list = []
  for prof in obs_list:
    peak = np.where(prof>=0.10)[0]   #Percentage
    width = peak.max() - peak.min()
    #width = float(width) / prof.size
    fwhm_list.append(width)
  return date_list, fwhm_list, freq_list





def plot_lists():
  date_list = []
  obs_list = []
  freq_list = []

  for obs in os.listdir(product_folder):
    if os.path.isdir(os.path.join(product_folder,obs)):
      archive = '{}/{}/{}_correctDM.clean.TF.b512.ar'.format(product_folder,obs,obs)

      if os.path.isfile(archive):
        date, prof, freq = load_archive(obs)

        date_list.append(date)
        obs_list.append(prof)
        freq_list.append(freq)

  #for obs in obs_list:
  #  obs -= np.median(obs)
  #  obs /= np.max(obs)

  idx = np.argsort(date_list)
  return np.array(date_list)[idx], np.array(obs_list)[idx], np.array(freq_list)[idx]


def load_archive(obs):
  archive = '{}/{}/{}_correctDM.clean.TF.b512.ar'.format(product_folder,obs,obs)

  load_archive = psrchive.Archive_load(archive)
  prof = load_archive.get_data().flatten()
  prof -= np.median(prof)
  prof /= np.max(prof)
  prof = np.roll(prof,(len(prof)-np.argmax(prof))+len(prof)/2)

  epoch = load_archive.get_Integration(0).get_epoch()
  date = datetime.date(int(epoch.datestr('%Y')), int(epoch.datestr('%m')), int(epoch.datestr('%d')))

  freq = load_archive.get_centre_frequency() 
        
  #freq = load_archive.get_filename()         

  return date, prof, freq








if __name__ == "__main__":
  date_list, fwhm_list, freq_list = width_fwhm()
  '''
  fwhm_list = [x for (y,x) in sorted(zip(date_list,fwhm_list), key=lambda pair: pair[0])]
  date_list = sorted(date_list)
  plt.plot(date_list,fwhm_list,'ko-')
  plt.xlim((min(date_list)-datetime.timedelta(days=30),max(date_list)+datetime.timedelta(days=30)))
  plt.show()
  '''

  np.save('date',date_list)
  np.save('fwhm',fwhm_list)
  np.save('file',freq_list)

