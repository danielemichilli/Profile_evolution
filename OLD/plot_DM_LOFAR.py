import numpy as np
import matplotlib.pyplot as plt
from datetime import date

import mjd2date

DM_file = '/data1/Daniele/B2217+47/Analysis/DM/DM_values.dat'

def plot(ax, date_min=date.min):
  DM = np.loadtxt(DM_file, usecols=[0,1,2]).T
  date = np.array([mjd2date.convert(mjd) for mjd in DM[0]])
  idx = np.where(date > date_min)[0]
  date = date[idx]
  DM = DM[:, idx]


  ax.errorbar(date, DM[1], yerr=DM[2], fmt='ko')
  ax.set_ylabel("DM (pc/cc)")
  return


if __name__ == '__main__':
  #plt.figure(figsize=(10,40))
  ax = plt.subplot()
  plot(ax)
  plt.show()

