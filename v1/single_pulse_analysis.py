import numpy as np
import psrchive
import sys
import matplotlib.pyplot as plt


def timeseries(archive):
  '''
  Load the archive
  '''
  times = psrchive.Archive_load(archive).get_data()
  times = np.sum(times,axis=(1,2))

  med = np.median(times,axis=1)[:, np.newaxis]
  times = times-med
  times /= times.max()
  return times


def pulses_plot(times,start=0,end=100,bin_lim=(0,1024)):
  '''
  Plot the singlepulses
  '''
  if isinstance(times, basestring): times = timeseries(times)
  prof = times.sum(axis=0)
  roll_idx = len(prof)-np.argmax(prof)+len(prof)/2
  times = np.roll(times, roll_idx, axis=1)
 
  n_col = int((end-start)/10)
  if (end-start)%10 != 0: n_col += 1 
  f, axarr = plt.subplots(1, n_col, sharey=True, figsize=(5*n_col,20))

  try:
    for idx, ax_i in enumerate(axarr):
      i = idx * 10
      for j in range(10):
        if start+i+j < times.shape[0]: ax_i.plot(times[start+i+j]+j,'k')
    
      ax_i.set_xlim(bin_lim)
      ax_i.set_ylim((0,10))
      ax_i.set_title("bin0 = {}".format(i))
  
  except TypeError:
    for j in range(start,end):
      axarr.plot(times[j]+j,'k')
    
    axarr.set_xlim(bin_lim)
    axarr.set_ylim((0,10))
    axarr.set_title("bin0 = {}".format(start))

  plt.tight_layout()
  plt.show()


def giant_pulses(times,rel_hight=0.06,left_lim=60,right_lim=980):
  '''
  Detect giant pulses in the data
  '''
  prof = times.sum(axis=0)
  roll_idx = len(prof)-np.argmax(prof)
  times = np.roll(times, roll_idx, axis=1)
  chunk = times[:,left_lim:right_lim]
  print "{} bins above {} times the peak found".format(chunk[chunk>rel_hight].size,rel_hight)
   
  return

if __name__ == "__main__":
  rel_hight = 0.06
  left_lim = 60
  right_lim = 980
  archive = str(sys.argv[1])
  times = timeseries(archive)
  giant_pulses(times,rel_hight,left_lim,right_lim)
  pulses_plot(times=times, start=1790, end=1890)
