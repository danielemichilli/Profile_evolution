import numpy as np
import psrchive
import os
import matplotlib.pyplot as plt
import datetime

from cum_profile import plot_lists

home_folder     = '/data1/Daniele/B2217+47'
product_folder  = home_folder + '/Products'


def width_fwhm():
  date_list, obs_list = plot_lists()
  fwhm_list = []
  for prof in obs_list:
    peak = np.where(prof>=0.5)[0]
    width = peak.max() - peak.min()
    #width = float(width) / prof.size
    fwhm_list.append(width)
  return date_list, fwhm_list

if __name__ == "__main__":
  date_list, fwhm_list = width_fwhm()
  '''
  fwhm_list = [x for (y,x) in sorted(zip(date_list,fwhm_list), key=lambda pair: pair[0])]
  date_list = sorted(date_list)
  plt.plot(date_list,fwhm_list,'ko-')
  plt.xlim((min(date_list)-datetime.timedelta(days=30),max(date_list)+datetime.timedelta(days=30)))
  plt.show()
  '''

  idx = np.argsort(date_list) 
  np.save('date',np.array(date_list)[idx])
  np.save('fwhm',np.array(fwhm_list)[idx])


