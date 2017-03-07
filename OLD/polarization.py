import numpy as np
import psrchive
import os
import matplotlib.pyplot as plt
import datetime

from cum_profile import plot_lists

home_folder     = '/data1/Daniele/B2217+47'
product_folder  = home_folder + '/Products'


def polarization():
  for id
    subprocess.call(['pam','-T','-F','-e','TF_pol.ar','{}_correctDM.clean.ar'.format(obs)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)

    archive = id+'_correctDM.clean.TF_pol.ar'
    prof = psrchive.Archive_load(archive).get_data()
    prof = prof[0,:,0,:]
    np.save(id+'_pol',prof)

 
if __name__ == "__main__":
  date_list, fwhm_list = polarization()
  fwhm_list = [x for (y,x) in sorted(zip(date_list,fwhm_list), key=lambda pair: pair[0])]
  date_list = sorted(date_list)
  plt.plot(date_list,fwhm_list,'ko-')
  plt.xlim((min(date_list)-datetime.timedelta(days=30),max(date_list)+datetime.timedelta(days=30)))
  plt.show()

