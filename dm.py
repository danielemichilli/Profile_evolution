import numpy as np
import psrchive
import os
import matplotlib.pyplot as plt
import datetime

home_folder     = '/data1/Daniele/B2217+47'
product_folder  = home_folder + '/Products'

def dm_list2(only_core=False):
  date_list = []
  dm_16_list = []
  dm_16_err = []
  dm_full_list = []
  dm_full_err = []
  telescope_list = []
  obs_list = []
  num_list = []

  #Loop all the observations
  for obs in os.listdir(product_folder):
    if os.path.isdir(os.path.join(product_folder,obs)):
      archive = '{}/{}/{}_correctDM.clean.TF.ar'.format(product_folder,obs,obs)
      if not os.path.isfile(archive): continue
      if not os.path.isfile('{}/{}/{}.singleTOA'.format(product_folder,obs,obs)): continue
      if only_core and (obs[0] != 'L'): continue

      if os.path.isfile(archive):
        #Load observation code
        obs_list.append(obs)

        #Load observation epoch
        load_archive = psrchive.Archive_load(archive)
        epoch = load_archive.get_Integration(0).get_epoch()
        date = datetime.date(int(epoch.datestr('%Y')), int(epoch.datestr('%m')), int(epoch.datestr('%d')))
        date_list.append(date)

        #Load telescope name
        telescope = load_archive.get_telescope()
        telescope_list.append(telescope)

        #Load number of TOAs used
        with open('{}/{}/{}_F16.tim'.format(product_folder,obs,obs)) as f:
          rows = f.readlines()
          num_TOA = sum([1 for n in rows if n.find(obs) == 0])
          num_list.append(num_TOA)

        #Load dm from full channels 
        try:
          with open('{}/{}/{}_dm_full_chan.dat'.format(product_folder,obs,obs)) as f:
            lines = f.readlines()
          dm = float(lines[0])
          n = len(lines[0].strip().split('.')[-1])
          dm_err = int(lines[1])*10**-n
          dm_full_list.append(dm)
          dm_full_err.append(dm_err)
        except IOError:
          dm_full_list.append(np.nan)
          dm_full_err.append(np.nan)

        #Load dm from 16 channels 
        try:
          with open('{}/{}/{}_dm_16_chan.dat'.format(product_folder,obs,obs)) as f:
            lines = f.readlines()
          dm = float(lines[0])
          n = len(lines[0].strip().split('.')[-1])
          dm_err = int(lines[1])*10**-n
          dm_16_list.append(dm)
          dm_16_err.append(dm_err)
        except IOError:
          dm_16_list.append(np.nan)
          dm_16_err.append(np.nan)

  idx = np.argsort(date_list)
  return np.array(obs_list)[idx], np.array(date_list)[idx], np.array(telescope_list)[idx], np.array(num_list)[idx],\
    np.array(dm_full_list)[idx], np.array(dm_full_err)[idx], np.array(dm_16_list)[idx], np.array(dm_16_err)[idx]



def dm_list_old():
  date_list = []
  pdmp_list = []
  tempo_list = []
  telescope_list = []
  obs_list = []
  num_list = []

  #Loop all the observations
  for obs in os.listdir(product_folder):
    if os.path.isdir(os.path.join(product_folder,obs)):
      archive = '{}/{}/{}_correctDM.clean.TF.ar'.format(product_folder,obs,obs)
      if not os.path.isfile(archive): continue
      if not os.path.isfile('{}/{}/{}.singleTOA'.format(product_folder,obs,obs)): continue

      if os.path.isfile(archive):
        #Load observation epoch
        load_archive = psrchive.Archive_load(archive)
        epoch = load_archive.get_Integration(0).get_epoch()
        date = datetime.date(int(epoch.datestr('%Y')), int(epoch.datestr('%m')), int(epoch.datestr('%d')))
        date_list.append(date)

        #Load dm from pdmp
        dm_pdmp = load_archive.get_dispersion_measure()
        pdmp_list.append(dm_pdmp)

        #Load telescope name
        telescope = load_archive.get_telescope()
        telescope_list.append(telescope)
  
        #Load dm from TEMPO2
        with open('{}/{}/{}.singleTOA'.format(product_folder,obs,obs)) as f:
          print obs
          dm_tempo = float(f.readline().split()[-1]) 
          tempo_list.append(dm_tempo)

        #Load observation code
        obs_list.append(obs)

        #Load number of TOAs used
        with open('{}/{}/{}_F16.tim'.format(product_folder,obs,obs)) as f:
          rows = f.readlines()
          num_TOA = sum([1 for n in rows if n.find(obs) == 0])
          num_list.append(num_TOA)
          
  idx = np.argsort(date_list)
  return np.array(date_list)[idx], np.array(pdmp_list)[idx], np.array(tempo_list)[idx], np.array(telescope_list)[idx], np.array(obs_list)[idx], np.array(num_list)[idx]



if __name__ == "__main__":
  #Usage of the lists
  import numpy as np
  import matplotlib.pyplot as plt
  from dm import dm_list2

  obs_list, date_list, telescope_list, num_list, dm_full_list, dm_full_err, dm_16_list, dm_16_err = dm_list2()

  telescope_list[telescope_list == 'Effelsberg'] = 'DE601'
  telescope_list[telescope_list == 'Eff'] = 'DE601'

  telescopes = np.unique(telescope_list)

  idx = np.where(dm_full_err<0.001)[0]
  x = date_list[idx]
  y = dm_full_list[idx]
  yerr = dm_full_err[idx]
  tel_list = telescope_list[idx]

  for t in ['DE601', 'DE603', 'DE605']:
    idx = np.where(tel_list == t)[0]
    plt.errorbar(x[idx], y[idx], yerr=yerr[idx], fmt='o', label=t)
  plt.legend()
  plt.show()




  for t in telescopes:
    idx = np.where(telescope_list == t)[0]
    plt.errorbar(date_list[idx], dm_full_list[idx], yerr=dm_full_err[idx], fmt='o', label=t)
  plt.legend()
  plt.show()

  for t in telescopes:
    idx = np.where(telescope_list == t)[0]
    plt.errorbar(date_list[idx], dm_16_list[idx], yerr=dm_16_err[idx], fmt='o', label=t)
  plt.legend()
  plt.show()



