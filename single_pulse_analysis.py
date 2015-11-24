import numpy as np
import psrchive
import sys


archive = str(sys.argv[1])

prof = psrchive.Archive_load(archive).get_data()
times = np.sum(prof,axis=(1,2))
times = (times-times.min(axis=0))/(times.max(axis=0)-times.min(axis=0))

def pulses_plot(times):
  for i in range(100):
    plt.plot(times[i]+i,'k')

  plt.xlim((0,1024))
  plt.show()



def giant_pulses(times,rel_hight,left_lim,right_lim):
  chunk = times[:,left_lim*1024:-right_lim*1024]
  print "{} bins above {} times the peak found".format(chunk[chunk>rel_hight].size,rel_hight)
   
  counts = np.bincount(chunk)
  ind = np.argpartition(counts, -10)[-10:]
  print "First ten profiles for number of single pulses above the limit: {}".format(ind)
   
  return


