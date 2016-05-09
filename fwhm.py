import numpy as np
import psrchive
import os
import matplotlib.pyplot as plt
import datetime

home_folder     = '/data1/Daniele/B2217+47'
product_folder  = home_folder + '/Products'
W_fract = 0.1

def width():
  '''
  Return the list of observation date, main component duration and file name for all the observations
  '''

  date_list, obs_list, file_list = observations_lists()

  #Calculate the duration list
  duration_list = []
  for prof in obs_list:
    peak = np.where(prof>=W_fract)[0]
    width = peak.max() - peak.min()
    #width = float(width) / prof.size  #Convert the duration in phase from bins
    duration_list.append(width)

  return date_list, duration_list, file_list



def observations_lists():
  '''
  Return the list of observation date, pulse profile and file name for all the observations 
  '''

  date_list = []
  obs_list = []
  file_list = []

  for obs in os.listdir(product_folder):
    if os.path.isdir(os.path.join(product_folder,obs)):
      archive = '{}/{}/{}_correctDM.clean.TF.b1024.ar'.format(product_folder,obs,obs)

      if os.path.isfile(archive):
        date, prof, file_name = load_archive(obs,archive)

        date_list.append(date)
        obs_list.append(prof)
        file_list.append(file_name)

  #Sort the lists by date
  idx = np.argsort(date_list)
  return np.array(date_list)[idx], np.array(obs_list)[idx], np.array(file_list)[idx]


def load_archive(obs,archive):
  '''
  Load the single observations and return observation date, pulse profile and file name for each
  '''

  #Load, calibrate and align the profile
  load_archive = psrchive.Archive_load(archive)
  prof = load_archive.get_data().flatten()
  prof -= np.median(prof)  #Profile median is set to 0
  prof /= np.max(prof)  #Profile peak is set to 1 (relative units)
  prof = np.roll(prof,(len(prof)-np.argmax(prof))+len(prof)/2)  #Align the peak to the phase middle

  #Load date and filename of the observation
  epoch = load_archive.get_Integration(0).get_epoch()
  date = datetime.date(int(epoch.datestr('%Y')), int(epoch.datestr('%m')), int(epoch.datestr('%d')))

  file_name = load_archive.get_filename()         

  return date, prof, file_name



if __name__ == "__main__":
  date_list, duration_list, file_list = width()
  
  #Plot the results
  plt.plot(date_list,duration_list,'ko-')
  plt.xlim((min(date_list)-datetime.timedelta(days=30),max(date_list)+datetime.timedelta(days=30)))
  plt.show()

  #Save the arrays
  np.save('date',date_list)
  np.save('duration',duration_list)
  np.save('file',file_list)

