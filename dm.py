import numpy as np
import psrchive
import os
import matplotlib.pyplot as plt
import datetime

home_folder     = '/data1/Daniele/B2217+47'
product_folder  = home_folder + '/Products'


def dm_list():
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
  date_list, pdmp_list, tempo_list, telescope_list, obs_list, num_list = dm_list()

  np.save('date',date_list)
  np.save('dm_pdmp',dm_list)
  np.save('dm_tempo',dm_list)
  np.save('telescope',telescope_list)
  np.save('observation',obs_list)
  np.save('TOAs',num_list)


'''
#Usage of the lists
import numpy as np
import matplotlib.pyplot as plt

telescopes = np.unique(telescope_list)

for t in telescopes:
  idx = np.where(telescope_list == t)[0]
  plt.plot(date_list[idx], tempo_list[idx], label=t)

plt.legend()
plt.show()

'''


