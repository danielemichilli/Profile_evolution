import numpy as np
import psrchive
import os
import matplotlib.pyplot as plt
import datetime

home_folder     = '/data1/Daniele/B2217+47'
product_folder  = home_folder + '/Products'


def dm_list():
  date_list = []
  dm_list = []
  telescope_list = []

  for obs in os.listdir(product_folder):
    if os.path.isdir(os.path.join(product_folder,obs)):
      archive = '{}/{}/{}_correctDM.clean.TF.ar'.format(product_folder,obs,obs)

      if os.path.isfile(archive):
        load_archive = psrchive.Archive_load(archive)
        epoch = load_archive.get_Integration(0).get_epoch()
        date = datetime.date(int(epoch.datestr('%Y')), int(epoch.datestr('%m')), int(epoch.datestr('%d')))
        date_list.append(date)

        dm = load_archive.get_dispersion_measure()
        dm_list.append(dm)

        telescope = load_archive.get_telescope()
        telescope_list.append(telescope)
  
  idx = np.argsort(date_list)
  return np.array(date_list)[idx], np.array(dm_list)[idx], np.array(telescope_list)[idx]



if __name__ == "__main__":
  date_list, dm_list, telescope_list = dm_list()

  np.save('date',date_list)
  np.save('dm',dm_list)
  np.save('telescope',telescope_list)





