import psrchive
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from scipy.optimize import curve_fit

data_folder = '/data1/Daniele/B2217+47/Archives_updated'


#Bright postcursor: 'L32532_fromfil.paz.clean.dm.pT'

def load_ar(ar_name):
  load_archive = psrchive.Archive_load(ar_name)
  load_archive.remove_baseline()
  load_archive.dedisperse()
  prof = load_archive.get_data().squeeze()
  freq_range = [load_archive.get_centre_frequency() - load_archive.get_bandwidth()/2., load_archive.get_centre_frequency() + load_archive.get_bandwidth()/2.]
  return prof, freq_range


def calculate_alpha(ar_name, params):
  prof, freq_range = load_ar(os.path.join(data_folder, ar_name))

  err_bin = np.std(prof[params[0] : params[1]])

  mp = prof[:, params[2] : params[3]].sum(axis=1)
  pc = prof[:, params[3] : params[4]].sum(axis=1)

  mp = mp[pc!=0]
  err_mp = mp * err_bin
  pc = pc[pc!=0]
  err_pc = pc * err_bin

  y = pc / mp
  err_y = np.sqrt( (err_pc / mp)**2 + (pc * err_mp / mp**2)**2 )
  idx = np.where(y<1)[0]
  y = y[idx]
  err_y = err_y[idx]
  x = np.linspace(freq_range[0], freq_range[1], y.size)

  #[m,q], V = np.polyfit(x, y, 1, w=1/err_y, cov=True)
  #err_m = np.sqrt(V[0][0])
  #err_q = np.sqrt(V[1][1])
  #alpha = np.log( (m*x[0]+q) / (m*x[-1]+q) ) / np.log( x[0] / x[-1] )
  #err_alpha = np.sqrt( err_m**2 * ( (x[0]/(m*x[0]+q) - x[-1]/(m*x[-1]+q)) / np.log(x[0]/x[-1]) )**2 + err_q**2 * ( (1./(m*x[0]+q) - 1./(m*x[-1]+q)) / np.log(x[0]/x[-1]) )**2 )

  def fit(x, *p):
    return p[0]*x**p[1]
  p0 = [1.,1.]
  coeff, pcov = curve_fit(fit, x, y, p0=p0)
  alpha = coeff[1]
  err_alpha = np.sqrt(np.diag(pcov))[1]

  alpha_tot = alpha - 1.98  
  err_alpha_tot = np.sqrt( (err_alpha * alpha)**2 + (1.98 * 0.09)**2 )

  return alpha_tot, err_alpha_tot


def DM_var():
  idx = np.where((prof.sum(axis=1) > 0) & (prof.sum(axis=1) < 1800))[0]
  prof = prof[idx]
  idx = np.where(prof[:, 100:116].max(axis=1) > 3*prof[:, 160:511].std())[0]
  prof = prof[idx]

  x = np.linspace(142.865-46.875/2., 142.865+46.875/2., prof.shape[0])

  def DM_fit(x, *p):
    DM, t0 = p
    return 4148.8*DM/x**2 + t0

  y_m = prof.argmax(axis=1)
  #z_m = np.polyfit(x, y_m, 1)
  #pol_m = np.poly1d(z_m)
  z_m, var_m = curve_fit(DM_fit, x, y_m, p0=p0)

  y_p = prof[:, 100:110].argmax(axis=1) + 100
  #z_p = np.polyfit(x, y_p, 1)
  #pol_p = np.poly1d(z_p)
  z_p, var_p = curve_fit(DM_fit, x, y_p, p0=p0)

  z_m_err = np.sqrt(np.diag(var_m))[0]
  z_p_err = np.sqrt(np.diag(var_p))[0]
  DM = np.abs(z_p[0]-z_m[0])
  #DM_err = np.sqrt(z_m_err**2*z_m[0]**2 + z_p_err**2*z_p[0]**2)
  f1 = 142.865+46.875/2.
  f2 = 142.865-46.875/2.
  DM_err = 1./4.15e6/(f2**-2-f1**-2)

#Profile parameters: off-pulse start, end, main start, end, post end
ar_parameters = {
'L32532_fromfil.paz.clean.dm.pT': [160,511,80,100,116],
'B2217+47_L461516_SAP0_BEAM0.dm.paz.clean.pT': [  0, 500, 535, 580, 622],
'B2217+47_L403348_SAP0_BEAM0.dm.paz.clean.pT': [  0, 150, 185, 231, 287],
'B2217+47_L370634_SAP0_BEAM0.dm.paz.clean.pT': [  0, 170, 203, 248, 310],
'B2217+47_L352456_SAP0_BEAM0.dm.paz.clean.pT': [  0, 180, 211, 259, 318],
'B2217+47_L348342_SAP0_BEAM0.dm.paz.clean.pT': [  0, 180, 219, 266, 327],
'B2217+47_L345486_SAP0_BEAM0.dm.paz.clean.pT': [  0, 190, 226, 272, 334],
'B2217+47_L337740_SAP0_BEAM0.dm.paz.clean.pT': [  0, 200, 238, 285, 349],
'B2217+47_L261995_SAP0_BEAM0.dm.paz.clean.pT': [  0, 200, 246, 293, 360],
'B2217+47_L261697_SAP0_BEAM0.dm.paz.clean.pT': [  0, 200, 233, 280, 340],
'B2217+47_L259469_SAP0_BEAM0.dm.paz.clean.pT': [  0, 230, 255, 303, 360],
'B2217+47_L256929_SAP0_BEAM0.dm.paz.clean.pT': [  0, 230, 262, 313, 376],
'B2217+47_L253707_SAP0_BEAM0.dm.paz.clean.pT': [  0, 240, 272, 323, 385],
'B2217+47_L249740_SAP0_BEAM0.dm.paz.clean.pT': [  0, 240, 282, 332, 396],
'B2217+47_L248701_SAP0_BEAM0.dm.paz.clean.pT': [  0, 250, 287, 337, 400],
'B2217+47_L246189_SAP0_BEAM0.dm.paz.clean.pT': [  0, 250, 291, 342, 410],
}


bad_obs = ['B2217+47_L352456_SAP0_BEAM0.dm.paz.clean.pT','B2217+47_L261995_SAP0_BEAM0.dm.paz.clean.pT','B2217+47_L261697_SAP0_BEAM0.dm.paz.clean.pT','B2217+47_L259469_SAP0_BEAM0.dm.paz.clean.pT','B2217+47_L256929_SAP0_BEAM0.dm.paz.clean.pT','B2217+47_L253707_SAP0_BEAM0.dm.paz.clean.pT','B2217+47_L249740_SAP0_BEAM0.dm.paz.clean.pT','B2217+47_L248701_SAP0_BEAM0.dm.paz.clean.pT','B2217+47_L246189_SAP0_BEAM0.dm.paz.clean.pT']


if __name__ == '__main__':
  
  #ar_name = sys.argv[1]
  #if ar_name in bad_obs:
  #  print "This observation does not have enough quality. Exiting..."
  #  exit()
  #if ar_name in ar_parameters:
  for ar_name in ar_parameters.iterkeys():
    a, a_err = calculate_alpha(ar_name, ar_parameters[ar_name])
    print ar_name, " :", a, a_err

  '''
  else:
    print "Obs. not known, please specify parameters"
    prof,_ = load_ar(os.path.join(data_folder, ar_name))
    prof = prof.sum(axis=0)
    fig = plt.figure(figsize=(20,10))
    plt.plot(prof/prof.max(),'k')
    plt.ylim([0,0.05])
    plt.xlim([prof.argmax()-50,prof.argmax()+100])
    plt.show()
  '''




