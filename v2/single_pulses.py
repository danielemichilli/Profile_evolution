import psrchive
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from scipy.optimize import curve_fit
import matplotlib as mpl
from matplotlib import rcParams
rcParams['ps.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages

mpl.rc('font',size=8)
data_folder = '/data1/Daniele/B2217+47/Archives_updated'


def sp_variab():
  fig = plt.figure(figsize=(3.3,3.3))
  ar_name = '/data1/Daniele/B2217+47/Analysis/sp/L32532_sp.F'
  load_archive = psrchive.Archive_load(ar_name)
  load_archive.remove_baseline()
  prof = load_archive.get_data().squeeze()
  w = load_archive.get_weights()
  prof *= w
  prof = prof[:prof.shape[0]-120]
  params = [0,800,887,925,951]

  prof -= np.mean(prof[:, params[0] : params[1]])
  prof /= np.std(prof[:, params[0] : params[1]])
  err_bin = np.std(prof[:, params[0] : params[1]], axis=1)

  mp = prof[:, params[2] : params[3]].sum(axis=1)
  pc = prof[:, params[3] : params[4]].sum(axis=1)
  err_mp = err_bin * np.sqrt(params[3]-params[2])
  err_pc = err_bin * np.sqrt(params[4]-params[3])

  err_mp /= mp.max()
  err_pc /= mp.max()
  pc = pc/mp.max()
  mp = mp/mp.max()

  #mp = np.roll(mp,1)

  #idx = np.argsort(mp)
  #mp = mp[idx]
  #pc = pc[idx]
  #err_mp = err_mp[idx]
  #err_pc = err_pc[idx]

  plt.errorbar(mp,pc,fmt='ko',xerr=err_mp/2.,yerr=err_pc/2., ms=0, elinewidth=0.5, capsize=0.5)
  x = np.linspace(0,1,1000)
  plt.plot(x, np.poly1d(np.polyfit(mp, pc, 2))(x),'r-')
  plt.plot([0,1],np.poly1d(np.polyfit(mp, pc, 1))([0,1]),'-',c='lightgreen')
  plt.plot([0,1], [0,0], 'r', lw=.5)

  plt.xlim([0,1])
  plt.ylim([-0.015,0.14])
  plt.xlabel('Main peak flux density (rel.)')
  plt.ylabel('Transient component flux density (rel.)')

  fig.tight_layout()
  #fig.savefig('single_pulses.eps', papertype='a4', orientation='portrait', format='eps', dpi=200)


  pp = PdfPages('single_pulses.pdf')
  pp.savefig(fig, papertype='a4', orientation='portrait', dpi=200)
  pp.close()





  plt.show()


if __name__ == '__main__':
  sp_variab()




