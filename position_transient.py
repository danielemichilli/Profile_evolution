import numpy as np
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt

data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"



def gauss(x, *p):
  A1, mu1, A2, mu2, sigma2 = p
  sigma1 = 3.0629962426616393
  return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))


def main_peak(observations):
  prof = observations[12]
  def fit(x, *p):
    A1, mu1, sigma1 = p
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2))
  x = np.arange(512.)
  p0 = [1., prof.argmax(), 3.]
  coeff, var_matrix = curve_fit(fit, x, prof, p0=p0, maxfev=10000)
  return coeff


def main():
  #Load LOFAR observations
  dates = np.load(os.path.join(data_folder, 'LOFAR_profiles_dates.npy'))
  observations = np.load(os.path.join(data_folder, 'LOFAR_profiles.npy'))

  x = np.arange(512.)
  delay = np.zeros(observations.shape[0])
  transient_peak = np.zeros(observations.shape[0])
  main_peak = np.zeros(observations.shape[0])
  p0 = [1., 260., 0.025, 275., 5.]
  for i, prof in enumerate(observations):
    coeff, var_matrix = curve_fit(gauss, x, prof, p0=p0, maxfev=10000, bounds=([0.9,257.,0.,258.,2.],[1.1,259.,0.3,300.,8.]))
    delay[i] = (coeff[3] - coeff[1]) / 512. * 538.4688219194
    transient_peak[i] = coeff[2]
    main_peak[i] = coeff[0]
    p0 = coeff

    #plt.plot(x, prof, 'k-', x, gauss(x,*coeff), 'r--')
    #plt.show()

  idx = np.where(transient_peak > 0.02)[0]
  

  return delay



