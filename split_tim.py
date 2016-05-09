import numpy as np
import subprocess
import glob
import os
import matplotlib.pyplot as plt


def main():
  precision = 1e-12

  timing_dir = '/data1/Daniele/B2217+47/timing_analysis/'
  par_file = 'B2217.par'
  tim_file = 'B2217.tim'

  x = []
  y = []
  y_err = []
  z = []
  z_err = []
  toas_span = []

  os.chdir(timing_dir)

  TOAs, tim_lines = load_TOAs(tim_file)
  for toa in TOAs:
    #Process each TOA separately
    write_par(toa,par_file)
    f0, f0_err, f1, f1_err, n_toas = timing_toa(toa,tim_lines,TOAs,timing_dir,precision)
    if f0 > 0:
      x.append(toa)
      y.append(f0)
      y_err.append(f0_err)
      z.append(f1)
      z_err.append(f1_err)
      toas_span.append(n_toas)

  results = np.array((x,y,y_err,z,z_err,toas_span))
  np.save('timing_results',results)

  return  


def load_TOAs(tim_file):
  '''
  Load TOAs lists given a .tim file (tempo2 format).
  Return list of TOAs and lines in the .tim file.
  '''
  TOAs = np.loadtxt(tim_file,usecols=(2,)) 
  with open(tim_file) as file:
    tim_lines = np.array(file.readlines())

  return TOAs, tim_lines


def timing_toa(toa,tim_lines,TOAs,timing_dir,precision):
  '''
  Calculate the timing parameters for one toa
  '''
  for n_close in range(3,20):
    idx = np.argpartition(np.abs(TOAs-toa), n_close)[:n_close]
    write_tim(tim_lines[idx])
    f0, f0_err, f1, f1_err = calculate_tim(timing_dir)
    
    if f0_err < precision: return f0, f0_err, f1, f1_err, n_close

  return -1,-1,-1,-1,-1

 
def write_tim(lines):
  '''
  Create a temporary tim file to calculate timing parameters
  '''
  #Put coe at the beginning (TEMPO2 bug)
  lines_new = []
  for n in lines:
    if 'coe' in n:
      lines_new.append(n)
  for n in lines:
    if not 'coe' in n:
      lines_new.append(n)

  #Write the tim file
  with open('tim', 'w') as f:
      f.write('FORMAT 1\n')
      for i in lines_new:
        f.write(i)
  return


def write_par(toa,par_file):
  '''
  Create a temporary par file to calculate timing parameters
  '''
  with open(par_file, 'r') as f:
    par = f.readlines()
  f0_i = float([n.split()[1] for n in par if 'F0' in n][0])
  f1_i = float([n.split()[1] for n in par if 'F1' in n][0])
  mjd_i = int([n.split()[1] for n in par if 'PEPOCH' in n][0])
  
  f0 = f0_i + (toa - mjd_i)* 24*3600* f1_i
  idx = [i for i,n in enumerate(par) if 'F0' in n][0]
  par[idx] = 'F0 {}\n'.format(f0)
  idx = [i for i,n in enumerate(par) if 'PEPOCH' in n][0]
  par[idx] = 'PEPOCH {}\n'.format(toa)

  with open('par', 'w') as f:
    for p in par:
      f.write(p)
    
  return

  
def calculate_tim(timing_dir):
  '''
  Call tempo2 to calculate the timing parameters
  '''
  output = subprocess.Popen(['tempo2','-output','general','-s','{F0_p} \n{F1_p} \n','-f','par','tim'],cwd=timing_dir,stdout=subprocess.PIPE)
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
  
  return f0, f0_err, f1, f1_err


if __name__ == "__main__":
  main()


