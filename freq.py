import psrchive
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

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

  [m,q], V = np.polyfit(x, y, 1, w=1/err_y, cov=True)
  err_m = np.sqrt(V[0][0])
  err_q = np.sqrt(V[1][1])

  alpha = np.log( (m*x[0]+q) / (m*x[-1]+q) ) / np.log( x[0] / x[-1] )
  err_alpha = np.sqrt( err_m**2 * ( (x[0]/(m*x[0]+q) - x[-1]/(m*x[-1]+q)) / np.log(x[0]/x[-1]) )**2 + err_q**2 * ( (1./(m*x[0]+q) - 1./(m*x[-1]+q)) / np.log(x[0]/x[-1]) )**2 )

  alpha_tot = alpha - 1.98  
  err_alpha_tot = np.sqrt( (err_alpha * alpha)**2 + (1.98 * 0.09)**2 )

  return alpha_tot, err_alpha_tot


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
  ar_name = sys.argv[1]
  #if ar_name in bad_obs:
  #  print "This observation does not have enough quality. Exiting..."
  #  exit()

  if ar_name in ar_parameters:
    a, a_err = calculate_alpha(ar_name, ar_parameters[ar_name])
    print a, a_err
  else:
    print "Obs. not known, please specify parameters"
    prof,_ = load_ar(os.path.join(data_folder, ar_name))
    prof = prof.sum(axis=0)
    fig = plt.figure(figsize=(20,10))
    plt.plot(prof/prof.max(),'k')
    plt.ylim([0,0.05])
    plt.xlim([prof.argmax()-50,prof.argmax()+100])
    plt.show()





