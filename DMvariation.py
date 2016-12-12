import numpy as np
import matplotlib.pyplot as plt
import mjd2date

def DMvariation(horizontal=True, ax=False):
  DM_core = np.loadtxt('/data1/Daniele/B2217+47/Analysis/DM/J2219+4754_DM_core.txt', skiprows=1, usecols=[0,1,2]).T
  telescope_core = np.loadtxt('/data1/Daniele/B2217+47/Analysis/DM/J2219+4754_DM_core.txt', skiprows=1, usecols=[3,], dtype=str)
  DM_glow = np.loadtxt('/data1/Daniele/B2217+47/Analysis/DM/J2219+4754_DM_GLOW.txt', skiprows=1, usecols=[0,1,2]).T
  telescope_glow = np.loadtxt('/data1/Daniele/B2217+47/Analysis/DM/J2219+4754_DM_GLOW.txt', skiprows=1, usecols=[3,], dtype=str)
  DM = np.concatenate((DM_core, DM_glow), axis=1)
  telescope = np.concatenate((telescope_core, telescope_glow))

  dates = np.array([mjd2date.converter(mjd) for mjd in DM[0]])

  if not ax: ax = plt.subplot()
  
  if horizontal:
    for tel in np.unique(telescope):  
      idx = np.where(telescope == tel)[0]
      ax.errorbar(dates[idx], DM[1, idx], yerr=DM[2, idx], fmt='o', label=tel)
    
    ax.set_xlim([734550.0, 736200.0])
    ax.set_ylim([43.479, 43.489])
    ax.set_xlabel('Date')
    ax.set_ylabel('DM (pc/cc)')
  
  else:
    for tel in np.unique(telescope):  
      idx = np.where(telescope == tel)[0]
      ax.errorbar(DM[1, idx], dates[idx], xerr=DM[2, idx], fmt='o', label=tel)
    
    ax.set_ylim([734550.0, 736200.0])
    ax.set_xlim([43.479, 43.489])
    ax.set_ylabel('Date')
    ax.set_xlabel('DM (pc/cc)')
    
  ax.legend()
  #plt.show()
  return


