#!/usr/bin/env python
import psrchive
import subprocess
import os
import numpy as np
from io import StringIO

home_folder     = '/data1/Daniele/B2217+47'
product_folder  = home_folder + '/Products/'
profile_template= home_folder + '/ephemeris/151109_profile_template.std'
par_file        = home_folder + '/Timing/LOFAR.par'

print "Process starting..."

obs_list = os.listdir(product_folder)

#Loop all the folders
for obs_name in obs_list:
  if not os.path.isdir(product_folder+obs_name): continue
  if not os.path.isfile('{}{}/{}_correctDM.clean.ar'.format(product_folder,obs_name,obs_name)): continue
  print "Obs. {} is being processed".format(obs_name)

  obs_path = product_folder+obs_name+'/'

  if not os.path.isfile('{}/{}_correctDM.clean.TF16.ar'.format(obs_path,obs_name)):
    #Scrunch the archive to 7 channels
    archive_name = obs_path + obs_name + '_correctDM.clean.ar'
    archive = psrchive.Archive_load(archive_name)
    chans_fact = archive.get_nchan() / 16
    nbin = archive.get_nbin()
    if nbin>1024:
      subprocess.call(['pam','-Tp','-f',str(chans_fact),'--DD','-e','TF16.ar','--setnbin','1024',archive_name],cwd=obs_path)
    else:
      subprocess.call(['pam','-Tp','-f',str(chans_fact),'--DD','-e','TF16.ar',archive_name],cwd=obs_path)

  if not os.path.isfile('{}/{}_F16.tim'.format(obs_path,obs_name)):
    #Calculate the TOAs
    output = subprocess.Popen(['pat','-s',profile_template,'-f','tempo2','{}_correctDM.clean.TF16.ar'.format(obs_name)],cwd=obs_path,stdout=subprocess.PIPE)
    out, err = output.communicate()
    #idx = out[::-1].find('\n',1)  #Uncomment to remove the last TOA
    #out = out[:-idx]              #Uncomment to remove the last TOA
    #Remove bad TOAs having an uncertainty higher than 100
    idx = np.where(np.loadtxt(StringIO(u'{}'.format(out)),skiprows=1,usecols=(3,))>100)[0]
    out_list = out.split('\n')
    for i in idx:
      out_list[i+1] = 'C ' + out_list[i+1]
    out = '\n'.join(out_list)
    with open('{}/{}_F16.tim'.format(obs_path,obs_name),'w') as f:
      f.write(out)
  
  if not os.path.isfile('{}/{}.singleTOA'.format(obs_path,obs_name)):
    #Calculate the DM 
    output = subprocess.Popen(['tempo2','-output','general','-s','startDM {dm_p} endDM\n','-f',par_file,'{}_F16.tim'.format(obs_name)],cwd=obs_path,stdout=subprocess.PIPE)
    out, err = output.communicate()
    start = out.find('startDM') + 8
    if start==7:
      print "ATTENTION: obs. {} not processed correctly".format(obs_name)
      continue
    end = out[start:].find('(')
    DM = out[start:start+end]

    #Write the TOA for the observation
    if os.path.isfile('{}/{}_correctDM.clean.TF.b1024.ar'.format(obs_path,obs_name)):
      output = subprocess.Popen(['pat','-s',profile_template,'-f','tempo2','{}_correctDM.clean.TF.b1024.ar'.format(obs_name)],cwd=obs_path,stdout=subprocess.PIPE)
    else:
      output = subprocess.Popen(['pat','-s',profile_template,'-f','tempo2','{}_correctDM.clean.TF.b512.ar'.format(obs_name)],cwd=obs_path,stdout=subprocess.PIPE)
    out, err = output.communicate()
    with open('{}/{}.singleTOA'.format(obs_path,obs_name),'w') as f:
      f.write(out.split('\n')[1] + '-dm {}\n'.format(DM))

  print '\n\n'
  

