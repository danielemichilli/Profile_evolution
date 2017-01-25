import numpy as np
import matplotlib.pyplot as plt

import mjd2date

DM_file = '/data1/Daniele/B2217+47/Analysis/DM/DM_values.dat'

def plot_DM(ax):
  DM = np.loadtxt(DM_file, usecols=[0,1,2]).T
  ax.errorbar(DM[0], DM[1], yerr=DM[2], fmt='ko')
  return


if __name__ == '__main__':
  #plt.figure(figsize=(10,40))
  ax = plt.subplot()
  plot_DM(ax)
  plt.show()

