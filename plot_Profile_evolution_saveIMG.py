import datetime
import numpy as np
import matplotlib.pyplot as plt
import psrchive
import matplotlib as mpl
import cum_profile
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc('font',size=15,weight='bold')
bins512 = False

if bins512: template = '/data1/Daniele/B2217+47/ephemeris/160128_profile_template_512.std'
else: template = '/data1/Daniele/B2217+47/ephemeris/151109_profile_template.std'
template = psrchive.Archive_load(template).get_data().flatten()

dates,obs_list = cum_profile.plot_lists(template=template,bin_reduc=bins512)
observations = np.array(obs_list)
date_list = np.array(dates)

#Remove bad observation
if bins512:
  idx = np.where(date_list == datetime.date(2011,10,30))[0][0]
  date_list = np.hstack((date_list[:idx],date_list[idx+1:]))
  observations = np.vstack((observations[:idx],observations[idx+1:]))

#Add earliest LOFAR observations
old_date, old_obs = cum_profile.load_early_obs(template=template,bin_reduc=bins512)
old_date = np.array(old_date)
old_obs = np.array(old_obs)

observations = np.vstack((old_obs,observations))
date_list = np.hstack((old_date,date_list))
idx = np.argsort(date_list)
date_list = date_list[idx]
observations = observations[idx]

#Average Observations on the same day
date_uniq, idx_uniq, repeated = np.unique(date_list,return_index=True,return_counts=True)
obs_uniq = observations[idx_uniq]
idx = np.where(repeated>1)[0]
for i in idx:
  date_rep = np.where(date_list == date_uniq[i])[0]
  obs_rep = np.mean(observations[date_rep],axis=0)
  obs_rep -= np.median(obs_rep)
  obs_rep /= np.max(obs_rep)
  obs_uniq[i] = obs_rep

#Create an image of the profile evolution calibrated in time
days = (date_list[-1] - date_list[0]).days
if bins512: img = np.zeros((days,512))
else: img = np.zeros((days,1024))
for i,n in enumerate(img):
  idx = np.abs(date_uniq - date_uniq[0] - datetime.timedelta(i)).argmin()
  img[i] = obs_uniq[idx]
if bins512: img = img[:,220:320]
else: img = img[:,450:600]

plt.figure(figsize=(10,30))
im = plt.imshow(np.clip(img,0,0.15),cmap='hot',origin="lower",aspect='auto',interpolation='nearest',extent=[0,img.shape[1]/1024.*100,0,img.shape[0]])
plt.xlabel('Phase [%]',fontweight='bold',fontsize=18)
plt.ylabel('Days from MJD 55400 (23/07/2010)',fontweight='bold',fontsize=18) #plt.ylabel('Days from {}'.format(date_list[0].strftime('%d/%m/%Y')),fontweight='bold',fontsize=18)
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="5%")
cbar = plt.colorbar(im, cax=cax)
lab = np.linspace(0,15,11,dtype=str)
lab[-1] = '> 15'
cbar.ax.set_yticklabels(lab)
cbar.set_label('Flux [% peak]',fontsize=18,fontweight='bold')
plt.savefig('test.png',bbox_inches='tight',format='png')


