import numpy as np
import matplotlib.pyplot as plt
from datetime import date

import mjd2date

timing_file = '/data1/Daniele/B2217+47/Analysis/Timing/residuals_LOFAR.dat'

def plot(ax, date_min=date.min):
  res = np.loadtxt(timing_file, usecols=[0,1,2]).T
  date = np.array([mjd2date.convert(mjd) for mjd in res[0]])
  idx = np.where(date > date_min)[0]
  date = date[idx]
  res = res[:, idx]
  
  tel = np.loadtxt(timing_file, usecols=[3,], dtype=object)
  idx_JB = np.where(tel == 'jbafb')[0]
  idx_LOFAR = np.where(tel != 'jbafb')[0]
  ax.errorbar(date[idx_JB], res[1, idx_JB], yerr=res[2, idx_JB], fmt='ko')
  ax.errorbar(date[idx_LOFAR], res[1, idx_LOFAR], yerr=res[2, idx_LOFAR], fmt='r^')
  ax.set_ylabel('dt (s)')
  return


if __name__ == '__main__':
  #plt.figure(figsize=(10,40))
  ax = plt.subplot()
  plot(ax)
  plt.show()

