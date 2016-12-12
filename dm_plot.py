import numpy as np
import matplotlib.pyplot as plt
import datetime
import cal_obs
import jdutil

dm_folder = '/data1/Daniele/B2217+47/Analysis/DM/'


def DM_evolution():
  #Plot DM evolution with archival data
  core = np.load(dm_folder+'CORE_DM.npy')
  inter = np.load(dm_folder+'dm_INTER.npy')
  GMRT = np.load(dm_folder+'dm_GMRT.npy')
  LWA = np.array([ [datetime.date(2014,7,4),], [43.4975,], [0.0005,] ])
  ATNF = np.array([ [datetime.date(1986,6,18),], [43.519,], [0.012,] ])
  JB = np.array([ [datetime.date(2007,9,29),datetime.date(2004,1,26),datetime.date(1999,10,4),datetime.date(1998,12,18),datetime.date(1998,3,22),datetime.date(1997,12,29),datetime.date(1997,10,15),datetime.date(1997,7,5),datetime.date(1996,9,16),datetime.date(1995,4,30),datetime.date(1993,12,29),datetime.date(1992,10,29),datetime.date(1991,12,10),datetime.date(1990,11,23),datetime.date(1989,8,5),datetime.date(1988,8,14),datetime.date(1984,9,3)], [43.4862,43.5052,43.5038,43.4838,43.506,43.4963,43.503,43.5138,43.5130,43.5157,43.5185,43.5287,43.5139,43.5183,43.520,43.5112,43.5277], [0.0098,0.0075,0.0028,0.0041,0.015,0.0073,0.017,0.0028,0.0034,0.0028,0.0017,0.0050,0.0022,0.0052,0.015,0.0014,0.0035] ])
  
  plt.errorbar(GMRT[0], GMRT[1], yerr=GMRT[2], fmt='go', label='GMRT')
  plt.errorbar(inter[0], inter[1], yerr=inter[2], fmt='co', label='LOFAR international')
  plt.errorbar(core[0],core[1],yerr=core[2],fmt='ko',label='LOFAR core')
  plt.errorbar(LWA[0], LWA[1], yerr=LWA[2], fmt='ro', label='LWA1 (ATNF)')
  plt.errorbar(ATNF[0], ATNF[1], yerr=ATNF[2], fmt='bo', label='JB (ATNF)')
  plt.errorbar(JB[0], JB[1], yerr=JB[2], fmt='mo', label='JB new')

  all_dm = np.hstack((core,GMRT,ATNF,JB,inter))
  idx = np.argsort(all_dm[0])
  all_dm = all_dm[:,idx]
  plt.plot(all_dm[0], all_dm[1], 'k--')

  #Hobbs04 relation
  ordinals = np.array( [n.toordinal() for n in all_dm[0]] )
  y = all_dm[1,-1] - 0.0002*np.sqrt(np.mean(all_dm[1]))*(ordinals-ordinals[-1])/365.
  plt.plot(ordinals,y,'r')

  plt.legend()
  plt.show()


  #Plot LOFAR data
  plt.errorbar(core[0],core[1],yerr=core[2],fmt='ko',label='Core')
  for label in np.unique(inter[3]):
    idx = np.where(inter[3]==label)[0]
    plt.errorbar(inter[0,idx], inter[1,idx], yerr=inter[2,idx], fmt='o', label=label)

  LOFAR_dm = np.hstack((core,inter[:3]))
  idx = np.argsort(LOFAR_dm[0])
  LOFAR_dm = LOFAR_dm[:,idx]
  plt.plot(LOFAR_dm[0], LOFAR_dm[1], 'k--')
  plt.legend()
  plt.show()



def fluxes():
  #Plot DM vs Fluxes
  JB_flux = np.load(dm_folder+'JB_FLUX.npy')
  inter = np.load(dm_folder+'dm_INTER.npy')
  core = np.load(dm_folder+'CORE_DM.npy')
  date,I,L,V,PA = cal_obs.pol_analysis()
  flux = np.mean(I,axis=1)
  LOFAR_flux = np.vstack((date,flux,flux/2.))

  f, axarr = plt.subplots(3, sharex=True)
  #axarr[0].errorbar(core[0],core[1],yerr=core[2],fmt='ko--')

  axarr[0].errorbar(core[0],core[1],yerr=core[2],fmt='ko',label='Core')
  for label in np.unique(inter[3]):
    idx = np.where(inter[3]==label)[0]
    axarr[0].errorbar(inter[0,idx], inter[1,idx], yerr=inter[2,idx], fmt='o', label=label)

  LOFAR_dm = np.hstack((core,inter[:3]))
  idx = np.argsort(LOFAR_dm[0])
  LOFAR_dm = LOFAR_dm[:,idx]
  axarr[0].plot(LOFAR_dm[0], LOFAR_dm[1], 'k--')
  axarr[0].set_ylabel("LOFAR DM (pc/cm3)")
  axarr[0].legend()

  axarr[1].errorbar(LOFAR_flux[0],LOFAR_flux[1],yerr=LOFAR_flux[2],fmt='ko--')
  axarr[1].set_ylabel("LOFAR MEAN FLUX (mJ)")
  axarr[2].errorbar(JB_flux[0],JB_flux[1],yerr=JB_flux[2],fmt='ko--')
  axarr[2].set_ylabel("JODRELL BANK MEAN FLUX (mJ)")
  axarr[2].set_xlabel("Date")
  plt.show()  


def DM_crab():
  crab = np.load(dm_folder+'DM_crab.npy')
  x = np.array([jdutil.mjd2date(n) for n in crab[0]])
  y = crab[1]
  dm_var = y.min() - all_dm[1].max()
  plt.plot(x,y-dm_var,'ko--', label = 'Crab')
  plt.legend()
  plt.show()
  


