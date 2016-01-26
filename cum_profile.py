#!/usr/bin/env python
import psrchive
import matplotlib.pyplot as plt
import os
import pyfits
import subprocess
import sys
import time
import argparse
import shutil
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from mpl_toolkits.mplot3d import proj3d
import datetime
import matplotlib as mpl
import matplotlib.cm as cm

home_folder     = '/data1/Daniele/B2217+47'
ephemeris_name  = '20111114_JB'
script_folder   = home_folder + '/scripts'
ephemeris_folder= home_folder + '/ephemeris'
ephemeris_file  = ephemeris_folder + '/' + ephemeris_name + '.par'
fits_folder     = home_folder + '/raw'
product_folder  = home_folder + '/Products'
plot_folder     = home_folder + '/Plots'
profile_template= home_folder + '/ephemeris/151109_profile_template.std'


def create_profile(args,verbose,overwrite,loud,parallel,exclude=False):
  """
  Process a list of fits file to produce cleaned archives.
  Plot the profiles of the archives both singularly and together.
  Process all the observations unless specific names are given as args.
  """
  
  if verbose: loud = True
  if parallel: loud = False
  if loud: print "\n#############################################\n"
  
  #Process all the observations if no arguments are given
  if args == None: obs_list = os.listdir(fits_folder)
  else: 
    obs_list = os.listdir(fits_folder)
    obs_list = [file for file in obs_list for obs in args if obs in file]
    if loud: 
      if len(obs_list) != len(args): print "WARNING: Number of submitted and found observations don't match. Pheraps some observation is missing\n"
  if exclude:
    obs_list = [x for x in obs_list if not x in exclude]

  print "The following observations will be processed:\n{}\n".format(obs_list)
  
  #Set the verbosity level
  if verbose: cmd_out = None
  else: cmd_out = open(os.devnull, 'w')
  
  if parallel: parallel_process(obs_list,cmd_out,overwrite,loud)
  else: execute_process(obs_list,cmd_out,overwrite,loud)

  return


def parallel_process(obs_list,cmd_out,overwrite,loud):
  pool = ThreadPool()
  n = len(obs_list) / mp.cpu_count() + 1
  obs_chunks = [obs_list[i:i + n] for i in range(0, len(obs_list), n)]
  for obs_chunk in obs_chunks:
    pool.apply_async(execute_process, args = (obs_chunk,cmd_out,overwrite,loud))
  pool.close()
  pool.join()
  return


def execute_process(obs_list,cmd_out,overwrite,loud):
  #Process all the observations
  if loud: print "\n  Start processing\n"
  for obs in obs_list:
    try:
      filename, file_ext = os.path.splitext(obs)
      if ((file_ext == '.fits') or (file_ext == '.fil')): store_type = 'fits'
      else: store_type = 'ar'
      if os.path.isfile('{}/{}'.format(fits_folder,obs)):
        if loud: print "  Observation {} is being processed".format(obs)
        obs_process(obs,cmd_out,overwrite,loud,store_type)
        if loud: print "  Observation {} processed\n".format(obs)
    except: 
      print "ATTENTION: Problems arised when processing obs {}".format(obs)

  return




def process_fits(fits,obs,output_dir,cmd_out,loud):
  #SCRUNCHED INITIAL ARCHIVE
  zaps = zap_channels('{}/{}'.format(fits_folder,fits))
  zaps = str(zaps)[1:-1].translate(None,',')

  #Create the initial scrunched archive from the fits file
  subprocess.call(['dspsr','-t','16','-E',ephemeris_file,'-j','zap chan',zaps,'-b','1024','-fft-bench',\
                  '-O','{}_{}'.format(obs,ephemeris_name),'-K','-A','-s','-e','ar','{}/{}'.format(fits_folder,fits)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  subprocess.call(['psredit','-c','rcvr:name=HBA','-m','{}_{}.ar'.format(obs,ephemeris_name)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  if loud: print "    Initial full-resolution archive created"



def process_ar(fits,obs,output_dir,cmd_out,loud):
  #SCRUNCHED INITIAL ARCHIVE
  subprocess.call(['pam','-E',ephemeris_file,'-u',output_dir,'{}/{}'.format(fits_folder,fits)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  #print 'pam','-E',ephemeris_file,'-u',output_dir,'{}/{}'.format(fits_folder,fits)
  os.rename('{}/{}'.format(output_dir,fits),'{}/{}_{}.ar'.format(output_dir,obs,ephemeris_name))
  subprocess.call(['psredit','-c','name=B2217+47','-m','{}_{}.ar'.format(obs,ephemeris_name)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  output = subprocess.Popen(['psredit','-c','site','-m','{}_{}.ar'.format(obs,ephemeris_name)],cwd=output_dir,stdout=subprocess.PIPE,stderr=cmd_out)  
  out, err = output.communicate()  
  idx = out.find('site=')
  if out[idx+5] == 't':
    subprocess.call(['psredit','-c','site=LOFAR','-m','{}_{}.ar'.format(obs,ephemeris_name)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  if loud: print "    Initial archive copied"



def obs_process(fits,cmd_out,overwrite,loud,store_type):
  #Extraxt the observation name from the fits file name
  obs = obs_from_fits(fits)
 
  #Create the output directory
  output_dir = '{}/{}'.format(product_folder,obs)
  try: os.makedirs(output_dir)
  except OSError:
    if overwrite:
      shutil.rmtree(output_dir)
      os.makedirs(output_dir)
    else: 
      print "  Directory {} already exists in {}, it will not be overwritten!".format(obs,product_folder)
      return
  if loud: print "    Fits file read and destination directory created"
  
  
  if store_type == 'fits': process_fits(fits,obs,output_dir,cmd_out,loud)
  elif store_type == 'ar': process_ar(fits,obs,output_dir,cmd_out,loud)
  else: 
    print "ATTENTION! Format not supported. Obs {} will not be processed.".format(obs) 
    return


  #Remove RFI and clean the archive
  subprocess.call(['paz','-e','paz.ar','-r','{}_{}.ar'.format(obs,ephemeris_name)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  subprocess.call(['clean.py','-F','surgical','-o','{}_{}.clean.ar'.format(obs,ephemeris_name),'{}_{}.paz.ar'.format(obs,ephemeris_name)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  if loud: print "    Archive cleaned"

  #Scrunch the archive in time and polarization for DM corrections
  subprocess.call(['pam','-T','-p','-e','T.ar','{}_{}.clean.ar'.format(obs,ephemeris_name)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)

  #Correct for DM variations and apply the new DM to the par file
  output = subprocess.Popen(['pdmp','-g','{}_correctDM.ps'.format(obs),'{}_{}.clean.T.ar'.format(obs,ephemeris_name)],cwd=output_dir,stdout=subprocess.PIPE,stderr=cmd_out)
  out, err = output.communicate()
  idx_start = out.find('Best DM')
  idx_end = out.find('\n',idx_start)
  DM = out[idx_start:idx_end].split()[3]
  write_ephemeris(ephemeris_file,DM,output_dir,ephemeris_name) 
  if loud: print "    Corrected DM written in new ephemeris file"

  #Apply new DM
  subprocess.call(['pam','-d','{}'.format(DM),'-e','tmp','{}_{}.clean.ar'.format(obs,ephemeris_name)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  os.rename('{}/{}_{}.clean.tmp'.format(output_dir,obs,ephemeris_name),'{}/{}_correctDM.clean.ar'.format(output_dir,obs))
  if loud: print "    DM updated"
  
  #Create a scrunched archive
  subprocess.call(['pam','-T','-F','-p','-e','TF.ar','{}_correctDM.clean.ar'.format(obs)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  if loud: print "    Scrunched archive created"

  #Plot of the profile of the scrunched archive
  subprocess.Popen(['pav','-CFTpD','-g','{}_profile.ps'.format(obs),'{}_correctDM.clean.TF.ar'.format(obs)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)

  #Crate a scrunched profile with subintegrations of a minute
  output = subprocess.Popen(['vap','-c','nsub length','{}_correctDM.clean.ar'.format(obs)],cwd=output_dir,stdout=subprocess.PIPE,stderr=cmd_out)
  out, err = output.communicate()
  nsub = out.split()[-2]
  length = out.split()[-1]  #Observation duration (s)
  sub_length = float(length) / float(nsub)
  length_parameter = int(60 / sub_length)
  subprocess.call(['pam','-t',str(length_parameter),'-F','-p','-e','t60F.ar','{}_correctDM.clean.ar'.format(obs)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)

  #Generate the TOAs
  output = subprocess.Popen(['pat','-s',profile_template,'{}_correctDM.clean.t60F.ar'.format(obs)],cwd=output_dir,stdout=subprocess.PIPE,stderr=cmd_out)
  out, err = output.communicate()
  idx = out[::-1].find('\n',1)
  out = out[:-idx]
  f = open('{}/{}.tim'.format(output_dir,obs),'w')
  f.write(out)
  f.close()

  return



def obs_from_fits(fits):
  idx_start = fits.find('L')
  #if idx_start < 0: idx_start = fits.find('T')
  if idx_start < 0: idx_start = fits.find('D')
  if idx_start < 0: idx_start = fits.find('_') + 1

  #while not fits[idx_start+1].isdigit():
  #  idx_start = fits.find('L',idx_start+1)
  idx_end = idx_start + 1
  while not fits[idx_end] in ['.','_']:
    idx_end += 1
    if idx_end == len(fits): break  

  return fits[idx_start:idx_end]
    

def read_ephemeris(file):
  with open(file) as f:
    for line in f:
      if len(line) > 1:
        parameter = line.split()
        if len(parameter) > 1:
          if parameter[0] == 'DM': dm = parameter[1]
          elif parameter[0] == 'F0': p = 1000/float(parameter[1])
  return dm, p





def write_ephemeris(ephemeris_file,DM,cwd,ephemeris_name):
  with open(ephemeris_file, 'r') as content_file:
    content = content_file.read()

  start_idx = content.find('DM')
  end_idx = content.find('\n',start_idx)
  
  new = content[:start_idx+3]+str(DM)+content[end_idx:]

  with open('{}/{}_correctDM.par'.format(cwd,ephemeris_name), 'w') as content_file:
    content_file.write(new)


def zap_channels(file):
  fits = pyfits.open(file)
  header = fits['SUBINT'].header

  N_channels = header['NCHAN']
  N_subbands = np.int(np.round(200./1024/header['CHAN_BW']))
  
  #obs_date = fits['PRIMARY'].header['DATE-OBS']
  #obs_date = obs_date[:10].translate(None,'-')

  fits.close()

  zap_channels = range(0,N_channels,N_subbands)

  return zap_channels#, obs_date
  

def plot_lists(exclude=False,date_lim=False,template=False,bin_reduc=False):
  date_list = []
  obs_list = []
  for obs in os.listdir(product_folder):
    if exclude:
      if obs in exclude:
        continue
    if os.path.isdir(os.path.join(product_folder,obs)):
      archive = '{}/{}/{}_correctDM.clean.TF.ar'.format(product_folder,obs,obs)

      if os.path.isfile(archive):
        load_archive = psrchive.Archive_load(archive)
        prof = load_archive.get_data().flatten()
        if bin_reduc:
          prof = prof.reshape(512,-1).mean(axis=1)
        prof -= np.median(prof)
        prof /= np.max(prof)
        if isinstance(template,np.ndarray):
          if bin_reduc:
            template = template.reshape(512,-1).mean(axis=1)
          bins = prof.size
          prof_ext = np.concatenate((prof[-bins/2:],prof,prof[bins/2:]))
          shift = prof.size/2 - np.correlate(prof_ext,template,mode='valid').argmax()
          prof = np.roll(prof,shift)
        else: prof = np.roll(prof,(len(prof)-np.argmax(prof))+len(prof)/2)
        epoch = load_archive.get_Integration(0).get_epoch()
        date = datetime.date(int(epoch.datestr('%Y')), int(epoch.datestr('%m')), int(epoch.datestr('%d')))
        if date_lim:
          if not date_lim[0] <= date <= date_lim[1]:
            continue


        #Add different observations together
        '''
        if date in date_list:
          try: obs_list[date_list.index(date)] += prof
          except ValueError:
            print "Archive {} has a different number of bins!".format(obs)
            date_list.append(date)
            obs_list.append(prof)
        else:
          date_list.append(date)
          obs_list.append(prof)
        '''

        date_list.append(date)
        obs_list.append(prof)
  for obs in obs_list:
    obs -= np.median(obs)
    obs /= np.max(obs)

  return date_list,obs_list


def cum_plot(phase_lim=False,date_lim=False,flux_lim=False,exclude=False,template=False,bin_reduc=False):
  date_list,obs_list = plot_lists(exclude,date_lim,template,bin_reduc)
  fig = plt.figure(figsize=(5,10))
  ax = fig.add_subplot(111)

  norm = mpl.colors.Normalize(vmin=min(date_list).toordinal(), vmax=max(date_list).toordinal())  
  m = cm.ScalarMappable(norm=norm, cmap='copper_r')
  
  base_y = np.min([date.toordinal() for date in date_list])
  scale_y = 0.001 / np.min(np.diff(sorted(date_list))).days
  obs_max = 0
  obs_min = 1
  for idx,obs in enumerate(obs_list):
    x = np.arange(0,1,1./len(obs))
    if phase_lim:
      zoom_idx = np.where((x>=min(phase_lim[0],phase_lim[1]))&(x<=max(phase_lim[0],phase_lim[1])))
      x = x[zoom_idx]
      obs = obs[zoom_idx]
    if flux_lim: obs = np.clip(obs,flux_lim[0],flux_lim[1])
    obs += scale_y * (float(date_list[idx].toordinal()) - base_y)
    obs_max = max(obs_max, max(obs))
    obs_min = min(obs_min, min(obs))
    ax.plot(x, obs, 'k') # color=m.to_rgba(date_list[idx].toordinal())

  if phase_lim: ax.set_xlim([phase_lim[0],phase_lim[1]])
  else: ax.set_xlim([0,1])
  ax.set_ylim([obs_min,obs_max])
  
  ax.set_xlabel('Phase')
  ax.set_ylabel('Date')
  glitch_epoch = datetime.date(2011, 10, 25).toordinal()
  if phase_lim: ax.scatter(phase_lim[1],scale_y * (glitch_epoch - base_y),c='r',marker='o',linewidth=0,s=12)
  else: ax.scatter(1,scale_y * (glitch_epoch - base_y),c='r',marker='o',linewidth=0,s=12)

  #Convert dates from ordinals
  fig.canvas.draw()
  ax.set_yticklabels([datetime.date.fromordinal(int((day/scale_y)+base_y)).strftime('%d-%m-%Y') for day in ax.get_yticks()])
  ax.tick_params(axis='y', labelsize=6)
  #Save the plot 
  plot_name = 'profiles'
  plt.savefig('{}/{}_{}.png'.format(plot_folder,time.strftime("%Y%m%d-%H%M%S"),plot_name),format='png',dpi=200)



def plot3D(phase_lim=False,date_lim=False,flux_lim=False,exclude=False,template=False,bin_reduc=False):
  date_list,obs_list = plot_lists(exclude,date_lim,template,bin_reduc)
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  for idx,obs in enumerate(obs_list):
    x = np.arange(0,1,1./len(obs))
    if phase_lim:
      zoom_idx = np.where((x>=min(phase_lim[0],phase_lim[1]))&(x<=max(phase_lim[0],phase_lim[1])))
      x = x[zoom_idx]
      obs = obs[zoom_idx]
    if flux_lim: obs = np.clip(obs,flux_lim[0],flux_lim[1])
    ax.plot(x, [float(date_list[idx].toordinal()),]*x.size,obs,color='k')
  
  if phase_lim: ax.set_xlim3d([phase_lim[0],phase_lim[1]])
  else: ax.set_xlim3d([0,1])
  if flux_lim: ax.set_zlim3d([flux_lim[0],flux_lim[1]])
  else: ax.set_zlim3d([0,1])
  ax.set_ylim3d([min(date_list).toordinal(),max(date_list).toordinal()])

  ax.set_xlabel('Phase')
  ax.set_ylabel('Date')
  ax.set_zlabel('Rel. amplitude')
  glitch_epoch = datetime.date(2011, 10, 25).toordinal()
  if phase_lim: ax.scatter(phase_lim[1],glitch_epoch,0,c='r',marker='o',linewidth=0,s=12)
  else: ax.scatter(1,glitch_epoch,0,c='r',marker='o',linewidth=0,s=12)
  #Convert dates from ordinals
  fig.canvas.draw()
  ax.set_yticklabels([datetime.date.fromordinal(int(day+1)).strftime('%d-%m-%Y') for day in ax.get_yticks()])
  ax.tick_params(axis='y', labelsize=6)
  #Save the plot
  plot_name = '3D_profiles'
  plt.savefig('{}/{}_{}.png'.format(plot_folder,time.strftime("%Y%m%d-%H%M%S"),plot_name),format='png',dpi=200)




if __name__ == '__main__':
  t0 = time.time()
  
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description="""
    Process a list of fits file to produce cleaned archives.
    Plot the profiles of the archives both singularly and together.
    Process all the observations unless specific names are given as args.
    """)
  parser.add_argument('-f','-folder', nargs='+', help='List of specific observations to process. Format: LXXXXX. All the observations will be processed by default if empty')
  parser.add_argument('-e','-exclude', nargs='+', help='List of specific observations to exclude from the process. Format: LXXXXX')
  parser.add_argument('-p','-plot', action='store_true', help='Produce a cumulative plot of all the observations without process them')
  parser.add_argument('-p3d','-plot3d', action='store_true', help='Produce a tridimensional plot of all the observations without process them')
  parser.add_argument('--phase_lim', nargs=2, type=float, help='Limits on the pulse phase')
  parser.add_argument('--date_lim', nargs=2, help='Limits on the date (DD-MM-YYYY)')
  parser.add_argument('--flux_lim', nargs=2, type=float, help='Limits on the flux (rel. units)')
  parser.add_argument('--bin_reduc', action='store_true', help='Scrunch the profile size to 512 bins')
  parser.add_argument('-ow','-overwrite', action='store_true', help='Reprocess all the observations present. ATTENTION: the content will be overwritten!')  
  parser.add_argument('--parallel', action='store_true', help='Process the observations on multiple CPUs')
  parser.add_argument('-v','-verbose', action='store_true', help='Verbose output')
  parser.add_argument('-q','-quiet', action='store_false', help='Quiet output')
  args = parser.parse_args()
 
  if args.date_lim: date_lim = [datetime.datetime.strptime(x,'%d-%m-%Y').date() for x in args.date_lim]
  else: date_lim = False

  #Produce the cumulative plot and exit if the plot argument is given
  if args.p | args.p3d:
    template = psrchive.Archive_load(profile_template).get_data().flatten() 
    if args.p: cum_plot(exclude=args.e,phase_lim=args.phase_lim,date_lim=date_lim,flux_lim=args.flux_lim,template=template,bin_reduc=args.bin_reduc)
    if args.p3d: plot3D(exclude=args.e,phase_lim=args.phase_lim,date_lim=date_lim,flux_lim=args.flux_lim,template=template,bin_reduc=args.bin_reduc)
    exit()
    
  create_profile(args.f,args.v,args.ow,args.q,args.parallel,exclude=args.e)
  
  if args.q: print """
  
Process complete.
Time spent: {} s
  
#############################################
  
  """.format(time.time()-t0)




