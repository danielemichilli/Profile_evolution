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

home_folder     = '/data1/Daniele/B2217+47'
ephemeris_name  = '20111114_JB'
script_folder   = home_folder + '/scripts'
ephemeris_folder= home_folder + '/ephemeris'
ephemeris_file  = ephemeris_folder + '/' + ephemeris_name + '.par'
fits_folder     = home_folder + '/raw'
product_folder  = home_folder + '/Products'
plot_folder     = home_folder + '/Plots'
profile_template = home_folder + '/ephemeris/151109_profile_template.std'

def create_profile(args,verbose,overwrite,loud,parallel):
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
      if (file_ext == '.fits'): store_type = 'fits'
      else: store_type = 'ar'
      if os.path.isfile('{}/{}'.format(fits_folder,obs)):
        if loud: print "  Observation {} is being processed".format(obs)
        obs_process(obs,cmd_out,overwrite,loud,store_type)
        if loud: print "  Observation {} processed\n".format(obs)
    except: 
      print "ATTENTION: Problems arised when processing obs {}".format(obs)

  return





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
  

  #FULL-RESOLUTION INITIAL ARCHIVE  
  #Create the initial full-resolution archive from the fits file
  #subprocess.call(['dspsr','-t','16','-E',ephemeris_file,'-j','zap chan',zaps,'-b','1024','-fft-bench',\
  #                  '-O','{}_{}'.format(obs,ephemeris_name),'-K','-A','-s','-e','ar','{}/{}'.format(fits_folder,fits)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  #subprocess.call(['psredit','-c','rcvr:name="HBA"','-m','{}_{}.ar'.format(obs,ephemeris_name)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  #if loud: print "    Initial full-resolution archive created"
  #
  ##Remove RFI and clean the archive
  #subprocess.call(['clean.py','-o','{}_{}.clean.ar'.format(obs,ephemeris_name),'{}_{}.ar'.format(obs,ephemeris_name)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  #if loud: print "    Archive cleaned"
  #
  ##Create a time scrunched archive (use for input to pdmp DM refinement)
  #subprocess.call(['pam','-t','64','-e','t64.ar','{}_{}.clean.ar'.format(obs,ephemeris_name)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  #if loud: print "    Scrunched archive created"
  ######################################

  #SCRUNCHED INITIAL ARCHIVE
  ephemeris = ephemeris_name
  if store_type == 'fits':
    #ephemeris = ephemeris_name 
    #Load information from the fits file
    zaps = zap_channels('{}/{}'.format(fits_folder,fits))
    zaps = str(zaps)[1:-1].translate(None,',')

    #Create the initial scrunched archive from the fits file
    subprocess.call(['dspsr','-t','16','-E',ephemeris_file,'-j','zap chan',zaps,'-b','1024','-fft-bench',\
                    '-O','{}_{}'.format(obs,ephemeris),'-K','-A','-L','10','-e','ar','{}/{}'.format(fits_folder,fits)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
    subprocess.call(['psredit','-c','rcvr:name=HBA','-m','{}_{}.ar'.format(obs,ephemeris)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
    if loud: print "    Initial full-resolution archive created"

  elif store_type == 'ar':
    #ephemeris = 'LOFAR_ephemeris'
    subprocess.call(['pam','-E',ephemeris_file,'-u',output_dir,'{}/{}'.format(fits_folder,fits)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
    os.rename('{}/{}'.format(output_dir,fits),'{}/{}_{}.ar'.format(output_dir,obs,ephemeris))    
    #shutil.copyfile('{}/{}'.format(fits_folder,fits),'{}/{}_{}'.format(output_dir,obs,ephemeris))
    subprocess.call(['psredit','-c','site=LOFAR','-m','{}_{}.ar'.format(obs,ephemeris)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)

  #Remove RFI and clean the archive
  #subprocess.call(['clean.py','-o','{}_{}.clean.ar'.format(obs,ephemeris),'{}_{}.ar'.format(obs,ephemeris)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  subprocess.call(['paz','-e','clean.ar','-r','{}_{}.ar'.format(obs,ephemeris)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  if loud: print "    Archive cleaned"

  #Create a time scrunched archive (use for input to pdmp DM refinement)
  subprocess.call(['pam','-t','64','-e','t64.ar','{}_{}.clean.ar'.format(obs,ephemeris_name)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  if loud: print "    Scrunched archive created"

  #################################


  #Correct for DM variations and apply the new DM to the par file
  output = subprocess.Popen(['pdmp','-g','{}_correctDM.ps'.format(obs),'{}_{}.clean.t64.ar'.format(obs,ephemeris)],cwd=output_dir,stdout=subprocess.PIPE,stderr=cmd_out)
  out, err = output.communicate()
  idx_start = out.find('Best DM')
  idx_end = out.find('\n',idx_start)
  DM = out[idx_start:idx_end].split()[3]
  write_ephemeris(ephemeris_file,DM,output_dir,ephemeris) 
  if loud: print "    Corrected DM written in new ephemeris file"

  #Create a new full-resolution archive
  if store_type == 'fits':
    subprocess.call(['dspsr','-t','16','-E','{}_correctDM.par'.format(ephemeris),'-j','zap chan',zaps,'-b','1024','-fft-bench',\
                      '-O','{}_correctDM'.format(obs),'-K','-A','-s','-e','ar','{}/{}'.format(fits_folder,fits)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
    subprocess.call(['psredit','-c','rcvr:name=HBA','-m','{}_correctDM.ar'.format(obs)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
    if loud: print "    Final full-resolution archive created"

  elif store_type == 'ar':
    #Apply new ephemeris
    subprocess.call(['pam','-E','{}_correctDM.par'.format(ephemeris),'-u',output_dir,'{}/{}'.format(fits_folder,fits)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
    os.rename('{}/{}'.format(output_dir,fits),'{}/{}_correctDM.ar'.format(output_dir,obs))
    subprocess.call(['psredit','-c','site=LOFAR','-m','{}_correctDM.ar'.format(obs)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)

  #Remove RFI and clean the archive
  #subprocess.call(['clean.py','-o','{}_correctDM.clean.ar'.format(obs),'{}_correctDM.ar'.format(obs)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  subprocess.call(['paz','-e','clean.ar','-r','{}_correctDM.ar'.format(obs)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  if loud: print "    Archive cleaned"
  
  #Create a scrunched archive and rotate it
  subprocess.call(['pam','-T','-F','-p','-e','TF.ar','{}_correctDM.clean.ar'.format(obs)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
  subprocess.call(['pam','-r','0.5','-m','{}_correctDM.clean.TF.ar'.format(obs)],cwd=output_dir,stdout=cmd_out,stderr=cmd_out)
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
    while not fits[idx_start+1].isdigit():
      idx_start = fits.find('L',idx_start+1)
    idx_end = idx_start + 1
    while fits[idx_end].isdigit():
      idx_end += 1
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


def plot_profiles(archive):
  def norm_shift(prof):
      prof = prof - np.min(prof)
      prof = prof / np.max(prof)
      prof = np.roll(prof,(len(prof)-np.argmax(prof))+len(prof)/2)
      return prof

  load_archive = psrchive.Archive_load(archive)
  prof = load_archive.get_data().flatten()
  prof = norm_shift(prof)
  date = float(load_archive.get_ephemeris().get_value('MJD'))
  plt.plot(np.arange(0,1,1./len(prof)),prof,label=int(date))


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
  

def plot_lists():
  date_list = []
  obs_list = []
  for obs in os.listdir(product_folder):
    if os.path.isdir(os.path.join(product_folder,obs)):
      archive = '{}/{}/{}_correctDM.clean.TF.ar'.format(product_folder,obs,obs)

      if os.path.isfile(archive):
        load_archive = psrchive.Archive_load(archive)
        prof = load_archive.get_data().flatten()
        prof -= np.median(prof)
        prof /= np.max(prof)
        prof = np.roll(prof,(len(prof)-np.argmax(prof))+len(prof)/2)
        epoch = load_archive.get_Integration(0).get_epoch()
        date = datetime.date(int(epoch.datestr('%Y')), int(epoch.datestr('%m')), int(epoch.datestr('%d')))
        if date in date_list:
          obs_list[date_list.index(date)] += prof
        else:
          date_list.append(date)
          obs_list.append(prof)
        #print obs[1:],date
  for obs in obs_list:
    obs -= np.median(obs)
    obs /= np.max(obs)

  return date_list,obs_list


def cum_plot(zoom=False):
  date_list,obs_list = plot_lists()
  for idx,obs in enumerate(obs_list):
    plt.plot(np.arange(0,1,1./len(obs)),obs,label=int(date_list[idx]))

  if zoom:
    plt.xlim([0.51,0.55])
    plt.ylim([-0.01,0.20])
  plt.legend(sorted(date_list))
  plt.xlabel('Phase')
  plt.ylabel('Rel. amplitude')
  #Save the plot
  if zoom: plot_name = 'cum_profile_zoom'
  else: plot_name = 'cum_profile'
  plt.savefig('{}/{}_{}.png'.format(plot_folder,time.strftime("%Y%m%d"),plot_name),format='png',dpi=200)


def plot3D(zoom=False):
  date_list,obs_list = plot_lists()
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  for idx,obs in enumerate(obs_list):
    x = np.arange(0,1,1./len(obs))
    if zoom:
      zoom_idx = np.where((x>=0.51)&(x<=0.6))
      x = x[zoom_idx]
      obs = obs[zoom_idx]
      obs = np.clip(obs,-0.1,0.05)
    ax.plot(x, [float(date_list[idx].toordinal()),]*x.size,obs,color='k')
  
  if zoom:
    ax.set_xlim([0.51,0.6])
    ax.set_ylim([min(date_list),max(date_list)])
    ax.set_zlim([-0.01,0.05])
  ax.set_xlabel('Phase')
  ax.set_ylabel('Date (MJD)')
  ax.set_zlabel('Rel. amplitude')
  glitch_epoch = datetime.date(2011, 10, 25).toordinal()
  ax.plot((x.max(),x.max()),(glitch_epoch,glitch_epoch),(0,0),'ro',linewidth=0,markersize=12)
  #Convert dates from ordinals
  fig.canvas.draw()
  ax.set_yticklabels([datetime.date.fromordinal(int(day+1)).strftime('%d-%m-%Y') for day in ax.get_yticks()])
  ax.tick_params(axis='y', labelsize=6)
  #Save the plot
  if zoom: plot_name = '3D_profiles_zoom'
  else: plot_name = '3D_profiles'
  plt.savefig('{}/{}_{}.png'.format(plot_folder,time.strftime("%Y%m%d"),plot_name),format='png',dpi=200)




if __name__ == '__main__':
  t0 = time.time()
  
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description="""
    Process a list of fits file to produce cleaned archives.
    Plot the profiles of the archives both singularly and together.
    Process all the observations unless specific names are given as args.
    """)
  parser.add_argument('-f','-folder', nargs='+', help='List of specific observations to process. Format: LXXXXX. All the observations will be processed if empty')
  parser.add_argument('-p','-plot', action='store_true', help='Produce a cumulative plot of all the observations without process them')
  parser.add_argument('-p3d','-plot3d', action='store_true', help='Produce a tridimensional plot of all the observations without process them')
  parser.add_argument('--zoom', nargs='+', help='Zoom on pulse parts. The first to values must be the pulse phase limits, the third (optional) is the maximum flux value')
  parser.add_argument('-ow','-overwrite', action='store_true', help='Reprocess all the observations present. ATTENTION: the content will be overwritten!')  
  parser.add_argument('--parallel', action='store_true', help='Process the observations on multiple CPUs')
  parser.add_argument('-v','-verbose', action='store_true', help='Verbose output')
  parser.add_argument('-q','-quiet', action='store_false', help='Quiet output')
  args = parser.parse_args()
  
  #Produce the cumulative plot and exit if the plot argument is given
  if args.p | args.p3d:
    if args.p: cum_plot(zoom = args.zoom)
    if args.p3d: plot3D(zoom = args.zoom)
    exit()
    
  create_profile(args.f,args.v,args.ow,args.q,args.parallel)
  
  if args.q: print """
  
Process complete.
Time spent: {} s
  
#############################################
  
  """.format(time.time()-t0)




