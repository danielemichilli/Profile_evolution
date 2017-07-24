import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import psrchive
from scipy.optimize import curve_fit

mpl.rc('font',size=8)


def plot_pulses(start=50, end=100, step=0.1):
  ar_name = '/data1/Daniele/B2217+47/Archives_updated/L32532_fromfil.paz.clean.dm.F'
  load_archive = psrchive.Archive_load(ar_name)
  load_archive.remove_baseline()
  load_archive.dedisperse()
  prof = load_archive.get_data().squeeze()
  prof = prof[100:150]
  prof /= prof.max()
  prof = prof[:, 50:150]

  d = (end-start)*step
  x = range(100)
  for n in prof:
    plt.fill_between(x, 0, n+d, facecolor='w', edgecolor='k')
    d -= 0.1
  plt.ylim((0.1,5))
  plt.xlim((5,95))
  plt.tick_params(which='both', top='off', bottom='off', right='off', left='off', labelbottom='off', labelleft='off')
  


def ratio():
  ar_name = '/data1/Daniele/B2217+47/Archives_updated/L32532_fromfil.paz.clean.dm.F'
  params = [160,511,80,100,116]
  load_archive = psrchive.Archive_load(ar_name)
  load_archive.remove_baseline()
  load_archive.dedisperse()
  prof = load_archive.get_data().squeeze()[:1850]
  prof /= prof.max()

  #idx = np.where(prof.max(axis=1)>prof.std(axis=1))[0]
  idx = np.where(prof.max(axis=1)>0.1)[0]
  mp = prof[idx, params[2] : params[3]].sum(axis=1)
  pc = prof[idx, params[3] : params[4]].sum(axis=1)
  y = pc / mp
  prof_int = np.sum(prof, axis=0)
  y_int = prof_int[params[3] : params[4]].sum() / prof_int[params[2] : params[3]].sum()



  #Fit histogram
  hist, bin_edges = np.histogram(y, bins=50)
  #hist, bin_edges = np.histogram(y/y_int, bins=50)
  hist = hist / float(hist.max())
  bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
  def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))
  p0 = [1., 0., 1.]
  coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
  x = np.linspace(bin_centres.min(),bin_centres.max(),10000)
  hist_fit = gauss(x, *coeff)
  width = bin_centres[1] - bin_centres[0]
  plt.bar(bin_centres, hist, align='center', width=width, color='w', lw=2.)
  plt.plot(x, hist_fit, 'r--', lw=2.)
  plt.xlabel('Normalized ratio')
  plt.ylabel('Normalized counts')
  print 'Fitted mean = ', coeff[1]   # 0.985
  print 'Fitted standard deviation = ', coeff[2]   # 0.191
  #print 'Average error on measurements = ', err_ratio_scaled.mean()  # 0.222
  textstr = '$\mu=%.2f$\n$\sigma=%.2f$'%(coeff[1], coeff[2])
  plt.text(2.5, 1, textstr, verticalalignment='top', fontsize=14)
  #plt.xlim([-1.5,3.5])
  plt.ylim([0,1.03])





if __name__ == '__main__':
  fig = plt.figure(figsize=(7,6))
  plot_pulses()
  #ratio()  

  fig.tight_layout()
  fig.savefig('single_pulses.eps', papertype='a4', orientation='portrait', format='eps', dpi=200)

  plt.show()

    
