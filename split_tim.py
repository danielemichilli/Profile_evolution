import numpy as np
import subprocess
import glob
import os
import matplotlib.pyplot as plt

mjd_start = 40585-100
mjd_end = 57385-100
days = 200

timing_dir = '/data1/Daniele/B2217+47/timing'

#Write out the tim files
tim = np.loadtxt('B2217_all_t2.tim',skiprows=1,usecols=(2,))
file = open('B2217_all_t2.tim')
hist, bin_hedges = np.histogram(tim,bins=(mjd_end-mjd_start)/days,range=(mjd_start,mjd_end))
for k,i in enumerate(hist):
  line = []
  for j in range(i):
    line.append(file.readline())
  if len(line) > 6:
    f = open('tim' + str(k), 'w')
    f.write('FORMAT 1\n')
    for j in line:
      f.write(j)
    f.close()
file.close()


#Calculate periods
os.chdir(timing_dir)

mjd = 46285.884251715530624
f0 = 1.8571179660262512416
f1 = -9.5374162027855618154e-15
f2 = 2.1835949967530860308e-27

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

  f = open('test.par','r')
  par = f.readlines()
  f.close()

  par[4] = 'PEPOCH         {}.0  \r\n'.format(date)
  f1 = f1 + (date-mjd)*24*3600*f2
  par[3] = 'F1             {} 1  3.9857359207474666112e-20\r\n'.format(f1)
  f0 = f0 + (date-mjd)*24*3600*f1
  par[2] = 'F0             {}     1  0.00000000000583536460   \r\n'.format(f0)

  mjd = date

  with open('test.par','w') as f:
    for p in par:
      f.write(p)
  print i
  output = subprocess.Popen(['tempo2','-output','general','-s','{F0_p} \n{F1_p} \n','-f','test.par',i],cwd=timing_dir,stdout=subprocess.PIPE)
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





