import numpy as np
import psrchive
import sys


archive = str(sys.argv[1])

def timeseries(archive):
  times = psrchive.Archive_load(archive).get_data()
  times = np.sum(times,axis=(1,2))

  med = np.median(times,axis=1)[:, np.newaxis]
  times = times-med
  times /= times.max()
  return times


def pulses_plot(times,start=0,end=100):
  prof = times.sum(axis=0)
  roll_idx = len(prof)-np.argmax(prof)+len(prof)/2
  times = np.roll(times, roll_idx, axis=1)
  for i in range(start,end):
    plt.plot(times[i]+i,'k')
  plt.xlim((0,1024))
  plt.show()


def giant_pulses(times,rel_hight=0.06,left_lim=60,right_lim=980):
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
  pulses_plot(times)

