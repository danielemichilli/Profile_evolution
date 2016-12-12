import psrchive
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import cum_profile

obs_folder = '/data1/Daniele/B2217+47/cal_Obs/'


def load_archive(archive,template):
  load_archive = psrchive.Archive_load(archive)
  load_archive.remove_baseline()
  epoch = load_archive.get_Integration(0).get_epoch()
  date = datetime.date(int(epoch.datestr('%Y')), int(epoch.datestr('%m')), int(epoch.datestr('%d')))
  prof = load_archive.get_data().squeeze()
  #prof /= np.max(prof)
  I0 = prof[0]
  bins = prof.size
  prof_ext = np.concatenate((I0[-bins/2:],I0,I0[:bins/2]))
  shift = bins/2 - np.correlate(prof_ext,template,mode='valid').argmax()
  prof = np.roll(prof,shift,axis=1)
  return date, prof


def pol_parameters(prof):
  I = prof[0]
  L = np.sqrt(prof[1]**2+prof[2]**2)
  V = prof[3]
  PA = np.rad2deg(np.arctan2(prof[2] , prof[1])) / 2.
  #PA[PA<90] += 90.
  #PA[PA>90] -= 90.
  return I,L,V,PA  

   
def pol_analysis(obs_folder=obs_folder):
  observations = os.listdir(obs_folder)
  date = []
  I = []
  L = []
  V = []
  PA = []

  template = '/data1/Daniele/B2217+47/ephemeris/151109_profile_template.std'
  template = psrchive.Archive_load(template).get_data().flatten()

  for obs in observations:
    if not obs.endswith('.DFT'): continue
    day, prof = load_archive(obs_folder + obs,template)
    params = pol_parameters(prof)
    date.append(day)
    I.append(params[0])
    L.append(params[1])
    V.append(params[2])
    PA.append(params[3])

  date = np.array(date)
  idx = np.argsort(date)
  date = date[idx]
  I = np.array(I)[idx]
  L = np.array(L)[idx]
  V = np.array(V)[idx]
  PA = np.array(PA)[idx]

  return date,I,L,V,PA

  #prof = prof[:,1024*.8:1024*0.9]

if __name__ == '__main__':
  I,L,V,PA = pol_analysis()

'''
#Usage in ipython
import cal_obs
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.cm as cm

date,I,L,V,PA = cal_obs.pol_analysis()

#Plot absolute fluxe peak
plt.errorbar(date,np.max(I,axis=1),yerr=np.max(I,axis=1)/2.,fmt='go',label='I')
plt.errorbar(date,np.max(L,axis=1),yerr=np.max(L,axis=1)/2.,fmt='ro',label='L')
plt.errorbar(date,np.max(V,axis=1),yerr=np.max(V,axis=1)/2.,fmt='bo',label='V')
plt.legend()
plt.show()

#Relative intensities
m = np.max(I,axis=1)
I = (I.T / m).T
L = (L.T / m).T
V = (V.T / m).T

#Plot relative flux of peak postcursor
frac_peak = np.max(L,axis=1) / np.max(I,axis=1)
frac_post = np.max(L[:,530:],axis=1) / np.max(I[:,530:],axis=1)
plt.plot(date,frac_peak,'ko',label='Peaks')
plt.plot(date,frac_post,'ro',label='Postcursors')
plt.legend()
plt.show()

#Plot relative flux of areas
In = I[1:,450:623]
Ln = L[1:,450:623]

#In = np.array([ np.sum(In[:,25:42],axis=1), np.sum(In[:,42:90],axis=1), np.sum(In[:,90:],axis=1) ])
#Ln = np.array([ np.sum(Ln[:,25:42],axis=1), np.sum(Ln[:,42:90],axis=1), np.sum(Ln[:,90:],axis=1) ])
In = np.array([ np.sum(In[:,25:42],axis=1), np.sum(In[:,42:74],axis=1), np.sum(In[:,74:90],axis=1), np.sum(In[:,90:],axis=1) ])
Ln = np.array([ np.sum(Ln[:,25:42],axis=1), np.sum(Ln[:,42:74],axis=1), np.sum(Ln[:,74:90],axis=1), np.sum(Ln[:,90:],axis=1) ])
#In = np.array([ np.sum(In[:,25:42],axis=1), np.sum(In[:,42:74],axis=1), np.sum(In[:,74:83],axis=1), np.sum(In[:,83:],axis=1) ])
#Ln = np.array([ np.sum(Ln[:,25:42],axis=1), np.sum(Ln[:,42:74],axis=1), np.sum(Ln[:,74:83],axis=1), np.sum(Ln[:,83:],axis=1) ])

y = Ln / In

f, axarr = plt.subplots(3, sharex=True)
d = date[1:] - date[1]
days = [n.days for n in d]
c = [float(n)/max(days) for n in days]
colors = iter(cm.cool(c))
for i,n in enumerate(y.T):
  col = next(colors)
  axarr[0].plot([8,33,57,106],n,'o',color=col,label=str(days[i]))
  #axarr[0].plot([8,41,106],n,'o',color=col,label=str(days[i]))
  axarr[1].plot(I[i+1,475:623],color=col)
  axarr[2].plot(L[i+1,475:623],color=col)
axarr[0].set_xlim([0,145])
axarr[0].set_ylabel('L/I')
axarr[1].set_ylabel('I (rel.)')
axarr[2].set_ylabel('L (rel.)')
axarr[2].set_xlabel('Bins')
axarr[0].axvline(17,color='k',linestyle='dashed')
axarr[0].axvline(49,color='k',linestyle='dashed')
axarr[0].axvline(65,color='k',linestyle='dashed')
axarr[1].axvline(17,color='k',linestyle='dashed')
axarr[1].axvline(49,color='k',linestyle='dashed')
axarr[1].axvline(65,color='k',linestyle='dashed')
axarr[2].axvline(17,color='k',linestyle='dashed')
axarr[2].axvline(49,color='k',linestyle='dashed')
axarr[2].axvline(65,color='k',linestyle='dashed')
leg = axarr[0].legend()
for i, text in enumerate(leg.get_texts()):
  if (i>=7) & (i<14): text.set_color("red")
axarr[0].set_zorder(1)
plt.tight_layout()
plt.show()



#Plot image of polarization evolution
days = (date[-1] - date[0]).days
imgI = np.zeros((days,1024))
imgL = np.zeros((days,1024))
imgV = np.zeros((days,1024))

for i,n in enumerate(imgI):
  idx = np.abs(date - date[0] - datetime.timedelta(i)).argmin()
  imgI[i] = I[idx]
  imgL[i] = L[idx]
  imgV[i] = V[idx]

lim=0.03
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
ax1.imshow(np.clip(imgI[:,450:650],0.,lim),cmap='Greys_r',origin="lower",aspect='auto',interpolation='nearest')
ax2.imshow(np.clip(imgL[:,450:650],0.,lim),cmap='Greys_r',origin="lower",aspect='auto',interpolation='nearest')
ax3.imshow(np.clip(imgV[:,450:650],0.,lim),cmap='Greys_r',origin="lower",aspect='auto',interpolation='nearest')
f.subplots_adjust(wspace=0.1)
plt.show()

#Scale intensities independentely
I = (I.T / np.max(I,axis=1)).T
L = (L.T / np.max(L,axis=1)).T
V = (V.T / np.max(V,axis=1)).T

starting_obs = 1 
start = 479
duration = 70

#PA
PA_n = PA[starting_obs:,start:start+duration]  #Select pahses based on pav output

for n in PA_n:  #Shift PA curves in the same quadrant
  n -= n[duration/2]
  np.mod(n - 50 + 180, 180., out=n)

#Plot PA evolution
f, axarr = plt.subplots(2, sharex=True)
d = date[starting_obs:] - date[starting_obs]
days = [n.days for n in d]
c = [float(n)/max(days) for n in days]
colors = iter(cm.cool(c))
for i,n in enumerate(PA_n):
  col = next(colors)
  axarr[0].plot(n,'o',color=col,label=str(days[i]))
  axarr[1].plot(I[i+starting_obs,start:start+duration],color=col)
leg = axarr[0].legend(loc='upper left')
for i, text in enumerate(leg.get_texts()):
  if (i>=7) & (i<14): text.set_color("red")
axarr[0].set_ylabel('PA (deg)')
axarr[1].set_ylabel('Flux (rel.)')
axarr[1].set_xlabel('Bins')
axarr[0].axvline(45.5,color='k',linestyle='dashed')
axarr[0].axvline(56.5,color='k',linestyle='dashed')
axarr[1].axvline(45.5,color='k',linestyle='dashed')
axarr[1].axvline(56.5,color='k',linestyle='dashed')
axarr[0].set_zorder(1)
plt.tight_layout()
plt.show()



'''


