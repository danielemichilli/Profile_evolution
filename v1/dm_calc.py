import subprocess
import psrchive
import os
import datetime

products_folder = '/data1/Daniele/B2217+47/Products'
obs_path = '/data1/Daniele/B2217+47/Products/DM' 
par_file = '/data1/Daniele/B2217+47/Timing/LOFAR.par'

obs_list = os.listdir(obs_path)
for obs_name in obs_list:
  if (not obs_name[0] == 'L') | (not obs_name[-1] == 'a'): continue
  output = subprocess.Popen(['tempo2','-output','general','-s','startDM {dm_p} endDM\n','-f',par_file,obs_name],cwd=obs_path,stdout=subprocess.PIPE)
  out, err = output.communicate()
  start = out.find('startDM') + 8
  if start==7:
    print "ATTENTION: obs. {} not processed correctly".format(obs_name)
    continue
  end = out[start:].find('(')
  DM = out[start:start+end]
  start += end+1
  end = out[start:].find(')')
  DM_err = out[start:start+end]
  with open('{}/DM_values.dat'.format(obs_path),'a') as f:
    f.write(obs_name)
    f.write('\t')
    f.write(DM)
    f.write('\t')
    f.write(DM_err)
    f.write('\n')




data = np.loadtxt('DM_values.dat',dtype=str)

date_list = []
obs_list = []
dm_list = []
dm_err_list = []
for line in data:
  #Save observation name
  obs_name = line[0][:-1]
  obs_list.append(obs_name)

  #Load observation epoch
  load_archive = psrchive.Archive_load('{}/{}/{}_correctDM.clean.TF.ar'.format(products_folder,obs_name,obs_name))
  epoch = load_archive.get_Integration(0).get_epoch()
  date = datetime.date(int(epoch.datestr('%Y')), int(epoch.datestr('%m')), int(epoch.datestr('%d')))
  date_list.append(date)

  #Save dm
  dm = float(line[1])
  dm_list.append(dm)

  #Load dm errors
  n = len(line[1].strip().split('.')[-1])
  dm_err = int(line[2])*10**-n 
  dm_err_list.append(dm_err)





