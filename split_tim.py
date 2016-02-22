import numpy as np
import subprocess
import glob
import os
import matplotlib.pyplot as plt

days = 50
mjd_start = 55838 + days / 4 * 0   + days
mjd_end   = 57376 + days / 4 * 0   + days

timing_dir = '/data1/Daniele/B2217+47/timing_analysis/'
par_file = 'B2217_timing_analysis.par'
tim_file = 'B2217.tim'

mjd_i = 46285.884251715530624
f0_i = 1.8571179660599825782
f1_i = -9.5368641371411035478e-15

#Write out the tim files
tim = np.loadtxt(timing_dir+tim_file,skiprows=1,usecols=(2,))
idx = np.where( (tim >= mjd_start) & (tim <= mjd_end) )[0]
tim = tim[idx]

hist, bin_hedges = np.histogram(tim,bins=(mjd_end-mjd_start)/days,range=(mjd_start,mjd_end))

with open(timing_dir+tim_file) as file:
  TOAs = file.readlines()
TOAs = np.array(TOAs)[idx+1]

count = 0
for i,n in enumerate(hist):
  line = []
  for j in range(n):
    j_start = np.sum( hist[:i] )
    line.append(TOAs[j_start+j])
  if len(line) > 12:
    count += 1
    with open('tim' + str(i), 'w') as f:
      f.write('FORMAT 1\n')
      for j in line:
        f.write(j)

print "{} tim files created\n".format(count)

#exit()

#Calculate periods
os.chdir(timing_dir)

x = []
y = []
y_err = []
z = []
z_err = []

file_list = glob.glob('tim*')
def sort_list(line):                            
  return int(line.split('tim')[-1])
file_list.sort(key=sort_list)

for i in file_list:
  num = int(i.split('tim')[-1])
  date = num * days + mjd_start + days / 2

  with open(timing_dir+par_file,'r') as f:
    par = f.readlines()

  par[2] = 'PEPOCH         {}.0  \r\n'.format(date)
  f0 = f0_i + (date-mjd_i)*24*3600*f1_i
  par[0] = 'F0             {}     1  0.00000000000583536460   \r\n'.format(f0)

  #mjd = date

  with open(timing_dir+par_file,'w') as f:
    for p in par:
      f.write(p)
  print i
  output = subprocess.Popen(['tempo2','-output','general','-s','{F0_p} \n{F1_p} \n','-f',par_file,i],cwd=timing_dir,stdout=subprocess.PIPE)
  out, err = output.communicate()
  out = out.split('\n')
  f0_all = out[-5]
  f0 = float(f0_all.split('(')[0])
  try: f0_err = float(f0_all.split('(')[-1].split(')')[0]) / 10**len(f0_all.split('(')[0].split('.')[-1].split('E')[0])*10**int(f0_all.split('(')[0].split('E')[1])
  except IndexError: f0_err = float(f0_all.split('(')[-1].split(')')[0]) / 10**len(f0_all.split('(')[0].split('.')[-1].split('E')[0])
  f1_all = out[-4]
  f1 = float(f1_all.split('(')[0])
  try: f1_err = float(f1_all.split('(')[-1].split(')')[0]) / 10**len(f1_all.split('(')[0].split('.')[-1].split('E')[0])*10**np.floor(np.log10(np.abs(f1))) 
  except IndexError: f1_err = float(f1_all.split('(')[-1].split(')')[0]) / 10**len(f1_all.split('(')[0].split('.')[-1].split('E')[0])

  print date
  print f0
  print f1
  print ''

  x.append(date)
  y.append(f0)
  y_err.append(f0_err)
  z.append(f1)  
  z_err.append(f1_err)

#y = [i for (j,i) in sorted(zip(x,y), key=lambda pair: pair[0])]
#x = sorted(x)

#plt.plot(x, y, 'kx-')
#plt.show()

np.save('date',x)
np.save('f0',y)
np.save('f0_err',y_err)
np.save('f1',z)
np.save('f1_err',z_err)





