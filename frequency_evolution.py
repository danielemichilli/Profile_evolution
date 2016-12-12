import psrchive
import numpy as np
import matplotlib.pyplot as plt
import sys

archive = '/data1/daniele/B2217+47/Products/{}/{}_correctDM.clean.T.ar'.format(sys[1])

load_archive = psrchive.Archive_load(archive)
load_archive.remove_baseline()
prof = load_archive.get_data()

m = np.max(prof,axis=0)
prof = prof / m
prof = prof.T
for i in range(16):
  prof[i] = np.roll(prof[i],(len(prof[i])-np.argmax(prof[i]))+len(prof[i])/2)

peakp = []
peakposp = []
for i in range(16):
  peakp.append(np.max(prof[i,530:]))
  peakposp.append(np.argmax(prof[i,530:])+530)

plt.plot(peakp,'kx')
plt.show()


