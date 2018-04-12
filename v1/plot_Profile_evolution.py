import datetime
import numpy as np
import matplotlib.pyplot as plt
import psrchive
import matplotlib as mpl
import cum_profile
import os

def weak_comp():
  date_lim = [datetime.date(2013,01,01),datetime.date(2017,01,01)]
  dates,obs_list = cum_profile.plot_lists(template=template,date_lim=date_lim)
  
  observations = np.array(obs_list)
  date_list = np.array(dates)
  idx = np.argsort(date_list)
  date_list = date_list[idx]
  observations = observations[idx]
  
  #Average over dt
  avg = []
  avg_date = []
  avg_num = []
  date0 = date_list[0]
  dt = 60
  for idx,date in enumerate(date_list):
    if date - date0 < datetime.timedelta(dt):
      temp.append(observations[idx])
    else:
      if len(temp) > 5: avg.append(np.mean(temp,axis=0))
      if len(temp) > 5: avg_date.append(date0+datetime.timedelta(dt/2))
      if len(temp) > 5: avg_num.append(len(temp))
      temp = []
      date0 += datetime.timedelta(dt)
  #avg.append(np.mean(temp,axis=0))
  #avg_date.append(date0+datetime.timedelta(dt/2))
  avg = np.array(avg)
  avg_date = np.array(avg_date)

def normalize_profile(prof,template):
  prof -= np.median(prof)
  prof /= np.max(prof)
  bins = prof.size
  prof_ext = np.concatenate((prof[-bins/2:],prof,prof[:bins/2]))
  shift = prof.size/2 - np.correlate(prof_ext,template,mode='valid').argmax()
  prof = np.roll(prof,shift)
  return prof

def all_obs(bins512 = False, date_lim=False):
  if bins512: template = '/data1/Daniele/B2217+47/ephemeris/160128_profile_template_512.std'
  else: template = '/data1/Daniele/B2217+47/ephemeris/151109_profile_template.std'
  template = psrchive.Archive_load(template).get_data().flatten()

  dates,obs_list = cum_profile.plot_lists(template=template, bin_reduc=bins512, date_lim=date_lim)
  observations = np.array(obs_list)
  date_list = np.array(dates)

  #Remove bad observation
  #if bins512:
  #  idx = np.where(date_list == datetime.date(2011,10,30))[0][0]
  #  date_list = np.hstack((date_list[:idx],date_list[idx+1:]))
  #  observations = np.vstack((observations[:idx],observations[idx+1:]))

  #Add earliest LOFAR observations
  old_date, old_obs = cum_profile.load_early_obs(template=template,bin_reduc=bins512)
  old_date = np.array(old_date)
  old_obs = np.array(old_obs)
  #if bins512:
  #  old_obs = np.mean(np.reshape(old_obs,(old_obs.shape[0],old_obs.shape[1]/2,2)),axis=2)

  observations = np.vstack((old_obs,observations))
  date_list = np.hstack((old_date,date_list))
  idx = np.argsort(date_list)
  date_list = date_list[idx]
  observations = observations[idx]

  return date_list,observations

def plot_image(date_list, observations, bins512 = False, date_lim=False, ax=False):
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

  if not ax: ax = plt.subplot()
  date_min = mpl.dates.date2num(date_list[0])
  date_max = mpl.dates.date2num(date_list[-1])
  s = ax.imshow(np.clip(img,0,0.15),cmap='Greys_r',origin="lower",aspect='auto',interpolation='nearest',extent=[0, img.shape[1]/1024.*100, date_min, date_max])
  ax.yaxis_date()
  cbar = plt.colorbar(s)
  cbar.ax.set_yticklabels(np.linspace(0,15,11,dtype=str))
  cbar.set_label('Flux [% peak]')
  ax.set_xlabel('Phase [%]')
  ax.set_ylabel('Date')
  #ax.yaxis.set_major_formatter(mpl.dates.DateFormatter('%d:%m:%y'))
  #plt.savefig('test',bbox_inches='tight')
  #plt.show()
  return

def make_plot():
  date_list, observations = all_obs()
  plt.figure(figsize=(10,40))
  ax = plt.subplot()
  plot_image(date_list, observations, ax=ax)
  #plt.savefig('test',bbox_inches='tight')
  plt.show()


def shifting_post():
  date_lim = [datetime.date(2000,01,01),datetime.date(2013,01,01)]
  dates,obs_list = cum_profile.plot_lists(template=template,date_lim=date_lim)

  observations = np.array(obs_list)
  date_list = np.array(dates)
  idx = np.argsort(date_list)
  date_list = date_list[idx]
  observations = observations[idx]

  date_list = np.hstack((date_list[0:2],date_list[3],date_list[14:]))
  observations = np.vstack((observations[0:2],np.mean(observations[2:14],axis=0),observations[14:]))
  old_obs = ['L25192','L22946','L08909']
  for obs in old_obs:
    file = '/data1/Daniele/B2217+47/Products/{obs}/{obs}_Profile.b512.dat'.format(obs=obs)
    prof = np.loadtxt(file,skiprows=30,usecols=[1,])
    prof = normalize_profile(prof,template)
    observations = np.vstack((prof,observations))
  date_list = np.hstack(([datetime.date(2010,07,28),datetime.date(2011,01,25),datetime.date(2011,04,13)],date_list))

  observations = observations[:,200:320]
  for n in observations:
    n -= np.median(n)
    n /= n.max()

def image():
  #Shifting:
  plt.imshow(np.clip(observations[:,40:90],0,0.2),cmap='Greys_r',origin="lower",aspect='auto',interpolation='nearest',extent=[-0.5,49.5,-0.5,7.5])
  plt.colorbar()
  plt.plot([-0.5,49.5],[3,3],'r-',linewidth=4)
  plt.xlim((-0.5,49.5))
  plt.ylim((-0.5,7.5))
  plt.show()

  #Weak:
  plt.imshow(np.clip(observations,0,0.03),cmap='Greys_r',origin="lower",aspect='auto',interpolation='nearest')
  plt.colorbar()
  plt.xlim((219.5,320.5))
  #plt.ylim((-0.5,7.5))
  plt.show()

def stacked():
  dates_idx = np.where(date_list > datetime.date(2000,01,01))[0]
  dates = date_list[dates_idx]
  obs_selected = observations[dates_idx]
  fig = plt.figure(figsize=(5,10))
  ax = fig.add_subplot(111)
  base_y = np.min([date.toordinal() for date in dates])
  scale_y = 0.001 / np.min(np.diff(np.unique(dates))).days
  obs_max = 0
  obs_min = 1
  for idx,obs in enumerate(obs_selected):
    obs = obs[450:600]
    x = np.arange(0,1,1./len(obs))
    obs = np.clip(obs,0,0.3)
    obs += scale_y * (float(dates[idx].toordinal()) - base_y)
    obs_max = max(obs_max, max(obs))
    obs_min = min(obs_min, min(obs))
    ax.plot(x, obs, 'k')
  #ax.set_ylim((-0.01,0.9))
  ax.set_xlabel('Phase')
  ax.set_ylabel('Date')
  #glitch_epoch = datetime.date(2011, 10, 25).toordinal()
  fig.canvas.draw()
  ax.set_yticklabels([datetime.date.fromordinal(int((day/scale_y)+base_y)).strftime('%d-%m-%Y') for day in ax.get_yticks()])
  ax.tick_params(axis='y', labelsize=6)
  plt.show()

def threeD():
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  for idx,obs in enumerate(obs_uniq):
    x = np.arange(0,1,1./len(obs))
    # ax.plot(x, [float(date_uniq[idx].toordinal()),]*x.size,obs,color='k')
    #ax.plot(x, [float(date_uniq[idx].toordinal()),]*x.size,np.clip(obs,0.03,0.2),color='k')
    # ax.plot(x[220:320], [float(date_uniq[idx].toordinal()),]*x[220:320].size,np.clip(obs[220:320],-0.01,0.03),color='k')  #Weak
    ax.plot(x[263:300], [float(date_uniq[idx].toordinal()),]*x[263:300].size,np.clip(obs[263:300],-10,0.15),color='k')  #Shifting
  ax.set_xlabel('Phase')
  ax.set_ylabel('Date')
  ax.set_zlabel('Rel. amplitude')
  fig.canvas.draw()
  ax.set_yticklabels([datetime.date.fromordinal(int(day+1)).strftime('%d-%m-%Y') for day in ax.get_yticks()])
  ax.tick_params(axis='y', labelsize=6)
  #ax.plot([0,1],[datetime.date(2011,10,25).toordinal(),datetime.date(2011,10,25).toordinal()],[0,0],'r-',linewidth=4)
  plt.show()

def width():
  #Calculate the duration list
  duration_list = []
  for prof in obs_uniq:
    peak = np.where(prof>=0.2)[0]
    width = peak.max() - peak.min()
    #width = float(width) / prof.size  #Convert the duration in phase from bins
    duration_list.append(width)
  delta = np.array(duration_list)

  plt.plot(date_uniq,delta,'ko')
  plt.show()

  #To calculate the right or left side
  duration_list = []
  for prof in obs_uniq:
    peak = np.where(prof>=0.2)[0] 
    width = prof.size/2 - peak.min()  #Left: prof.size/2 - peak.min()  #Right: peak.max() - prof.size/2 
    #width = float(width) / prof.size  #Convert the duration in phase from bins
    duration_list.append(width)
  delta = np.array(duration_list)

  plt.plot(date_uniq,delta,'ko')
  plt.show()



def cal_Obs():
  import cum_profile
  import numpy as np
  import os
  import datetime
  import psrchive
  import matplotlib.pyplot as plt
  template = '/data1/Daniele/B2217+47/ephemeris/151109_profile_template.std'
  template = psrchive.Archive_load(template).get_data().flatten()
  folder = '/data1/Daniele/B2217+47/cal_Obs/'
  date_list = []
  I0 = []
  I1 = []
  I2 = []
  I3 = []
  for file in os.listdir(folder):
    date, prof = load_archive(folder+file,template)
    date_list.append(date)
    I0.append(prof[0])
    I1.append(prof[1])
    I2.append(prof[2])
    I3.append(prof[3])
  date_list = np.array(date_list)
  I0 = np.array(I0)
  I1 = np.array(I1)
  I2 = np.array(I2)
  I3 = np.array(I3)
  idx = np.argsort(date_list)
  date_list = date_list[idx]
  I0 = I0[idx]
  I1 = I1[idx]
  I2 = I2[idx]
  I3 = I3[idx]

  I = np.stack((I0,I0,I3))
  I[0] = I0
  I[1] = np.sqrt(I[1]**2+I[2]**2) 
  I[2] = I3

  f, axarr = plt.subplots(1, 3, sharey=True, sharex=True)
  for pol,ax in enumerate(axarr):
    days = (date_list[-1] - date_list[0]).days
    img = np.zeros((days,1024))
    for i,n in enumerate(img):
      idx = np.abs(date_list - date_list[0] - datetime.timedelta(i)).argmin()
      img[i] = I[pol,idx]
    img = img[:,450:700]
    ax.imshow(np.clip(img,0.,0.05),cmap='Greys_r',origin="lower",aspect='auto',interpolation='nearest')
  plt.show()
  
  def load_archive(archive,template):
    load_archive = psrchive.Archive_load(archive)
    load_archive.remove_baseline()
    epoch = load_archive.get_Integration(0).get_epoch()
    date = datetime.date(int(epoch.datestr('%Y')), int(epoch.datestr('%m')), int(epoch.datestr('%d')))
    prof = load_archive.get_data().squeeze()
    prof /= np.max(prof)
    I0 = prof[0]
    bins = prof.size
    prof_ext = np.concatenate((I0[-bins/2:],I0,I0[:bins/2]))
    shift = bins/2 - np.correlate(prof_ext,template,mode='valid').argmax()
    prof = np.roll(prof,shift,axis=1)
    return date, prof



