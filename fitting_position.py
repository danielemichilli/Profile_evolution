import cum_profile
import numpy as np
import matplotlib.pyplot as plt
import psrchive
from scipy.optimize import curve_fit
import matplotlib.dates as mdates

product_folder = '/data1/Daniele/B2217+47/Products'
template = '/data1/Daniele/B2217+47/ephemeris/151109_profile_template.std'
template = psrchive.Archive_load(template).get_data().flatten()

dates = []
obs_list = []

def load_selected_obs():
  #obs_good = ['L32532','L33177','L77908','D20130524T055004','L403348','L461516']
  #obs_good = ['L403348','L461516']
  obs_good = ['L32532','L33177','L77908','D20130524T055004']
  for obs in obs_good:
    archive = '{}/{}/{}_correctDM.clean.TF.b1024.ar'.format(product_folder,obs,obs)
    date, prof = cum_profile.load_archive(archive,template)
    dates.append(date)
    obs_list.append(prof)

  obs_good = ['L22946','L24408','L24505','L25192']
  for obs in obs_good:
    archive = '{}/{}/{}_PSR_2217+47.pfd.bestprof'.format(product_folder,obs,obs)
    date, prof = cum_profile.load_early_single_obs(archive,template)
    dates.append(date)
    obs_list.append(prof)

  observations = np.array(obs_list)
  date_list = np.array(dates)
  idx = np.argsort(date_list)
  date_list = date_list[idx]
  observations = observations[idx]
  observations = observations[:,450:600]


'''
#def gauss(x, *p):
#    mu, sigma = p
#    return np.exp(-(x-mu)**2/(2.*sigma**2))
#p0 = [67., 15.]

#def gauss(x, *p):
#    mu1, sigma1, A2, mu2, sigma2 = p
#    return np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))
#p0 = [66., 6., 0.1, 76., 6.]

#def gauss(x, *p):
#    A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3 = p
#    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2)) + A3*np.exp(-(x-mu3)**2/(2.*sigma3**2))
#p0 = [0.7, 65., 6., 0.08, 100., 7., 0.4, 69., 5.]
'''

def fitting(observations):
  p0 = [66., 6., 0.05, 105., 6.]
  def gauss(x, *p):
    mu1, sigma1, A2, mu2, sigma2 = p
    return np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))

  x = np.arange(observations.shape[1])
  params = []
  params_err = []
  for i,prof in enumerate(observations):
    #coeff, var_matrix = curve_fit(gauss, x, prof, p0=p0, maxfev=10000, bounds=[(63.,5.,0.,70.,2.),(67.,8.,0.3,150.,20)])
    coeff, var_matrix = curve_fit(gauss, x, prof, p0=p0, maxfev=10000)
    params.append(coeff)
    params_err.append(np.sqrt(np.diag(var_matrix)))

    #fit_curve = gauss(x, *coeff)
    #plt.plot(x, prof, 'k--', label='Profile')
    #plt.plot(x, fit_curve, 'ro', label='Fit')
    #plt.show()

    #print i, coeff, (((gauss(x, *coeff)-prof)**2)/prof).sum()/(len(prof)-len(p0)-1)
    #print date_list[i], (coeff[3]-coeff[0])/1024.*0.5384688219194
    #if len(p0)>6:
    #  plt.plot(x, coeff[0]*np.exp(-(x-coeff[1])**2/(2.*coeff[2]**2)), label='gauss1')
    #  plt.plot(x, coeff[3]*np.exp(-(x-coeff[4])**2/(2.*coeff[5]**2)), label='gauss2')
    #  plt.plot(x, coeff[6]*np.exp(-(x-coeff[7])**2/(2.*coeff[8]**2)), label='gauss3')
    #  plt.legend()

  return np.array(params), np.array(params_err)

#DELAY
def delay():
  params, params_err = fitting(observations) 
  delay = (params[:,3]-params[:,0])/1024.*0.5384688219194*1000
  delay_err = np.sqrt( params_err[:,3]**2 + params_err[:,0]**2 )/1024.*0.5384688219194*1000

  x = mdates.date2num(date_list)
  xx = np.linspace(x.min()-30, x.max()+30, 100)
  dd = mdates.num2date(xx)
  linear = np.poly1d( np.polyfit(x, delay, 1, w=delay_err))
  parabola = np.poly1d( np.polyfit(x, delay, 2, w=delay_err) )

  plt.errorbar(date_list,delay,yerr=delay_err,fmt='ko')
  plt.plot(dd,linear(xx),'r--')
  plt.plot(dd,parabola(xx),'b-.')
  plt.show()



#FIT ALL
def fit_all_good():
  from plot_Profile_evolution import all_obs
  import fitting_position 
  import numpy as np
  import matplotlib.pyplot as plt

  date_list,observations = all_obs()
  idx = np.array([3,6,7,8,9,10,11,12,14,25,26,28,84,94,97,98,99])
  idx = np.hstack([idx,range(100,155)])
  date_list = date_list[idx]
  observations = observations[idx]
  observations = observations[:,450:700]

  params, params_err = fitting_position.fitting(observations)
  delay = (params[:,3]-params[:,0])/1024.*0.5384688219194*1000
  delay_err = np.sqrt( params_err[:,3]**2 + params_err[:,0]**2 )/1024.*0.5384688219194*1000

  f, (ax1, ax2) = plt.subplots(2, sharex=True)
  ax1.errorbar(date_list,delay,yerr=delay_err,fmt='ko')
  ax2.errorbar(date_list,params[:,2],yerr=params_err[:,2],fmt='ko')
  f.subplots_adjust(hspace=0)
  plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
  plt.show()


def fit_all():
  from plot_Profile_evolution import all_obs
  import fitting_position
  import numpy as np
  import matplotlib.pyplot as plt

  date_list,observations = all_obs()

  def gauss(x, *p): 
    mu1, mu2 = p
    return np.exp(-(x-mu1)**2/(2.*6.2**2)) + 0.09*np.exp(-(x-mu2)**2/(2.*8.**2))

  def fit(prof):
    x = np.arange(prof.size)
    coeff, var_matrix = curve_fit(gauss, x, prof, p0=p0, maxfev=10000)
    return coeff[1]-coeff[0]
  
  delay = []
  for n in observations:
    d = fit(n)
    delay.append(d)

  plt.plot(date_list, n, 'ko')
  plt.show()



def convolve_template():
  from plot_Profile_evolution import all_obs
  import numpy as np

  date_list,observations = all_obs()
  x = np.arange(512-25-50,512+45+51)

  delay = []
  for n in observations:
    best_mu = 0
    best_corr = 0

    for mu in np.linspace(512-25, 512+45, 701):
      template = np.exp(-(x-512)**2/(2.*6.2**2)) + 0.09*np.exp(-(x-mu)**2/(2.*8.**2))
      template /= template.max()
      corr = np.correlate(n,template,mode='valid').max()

      if corr > best_corr:
        best_corr = corr
        best_mu = mu

    delay.append(best_mu - 512)

  plt.plot(date_list, delay, 'ko')
  plt.show()




















