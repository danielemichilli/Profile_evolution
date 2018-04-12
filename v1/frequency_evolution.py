import psrchive
import matplotlib.pyplot as plt
import numpy as np


archive = '/data1/Daniele/B2217+47/Analysis/DM/L32532.dm'

#chan 1643 corrupted

ar = psrchive.Archive_load(archive)
ar.remove_baseline()
ar.dedisperse()
#ar.fscrunch_to_nchan(16)
min_freq = ar.get_Profile(0, 0, 0).get_centre_frequency()
max_freq = ar.get_Profile(0, 0, ar.get_nchan()-1).get_centre_frequency()

profs = ar.get_data().squeeze()
profs[1643] = 0

ar.fscrunch()
template = ar.get_data().squeeze()

bins = profs.shape[1]

#for i in range(profs.shape[0]):
#  prof_ext = np.concatenate((profs[i,-bins/2:],profs[i],profs[i,:bins/2]))
#  shift = bins/2 - np.correlate(prof_ext,template,mode='valid').argmax()
#  profs[i] = np.roll(profs[i],shift)


# Relative spectral index

main = profs[:, 80:100]
post = profs[:, 100:115]
freqs = np.linspace(min_freq, max_freq, profs.shape[0])

main_area = main.sum(axis=1)
post_area = post.sum(axis=1)
plt.plot(freqs, post_area/main_area, 'ko')

main_peak = main.max(axis=1)
post_peak = post.max(axis=1)
plt.plot(freqs, post_peak/main_peak, 'r^')


plt.show()


# DM values
main_temp = np.load('/data1/Daniele/B2217+47/Analysis/DM/main_template.npy')
post_temp = np.load('/data1/Daniele/B2217+47/Analysis/DM/post_template.npy')

main_pos = []
for prof in profs:
  prof_ext = np.concatenate((prof[-bins/2:],prof,prof[:bins/2]))
  corr = np.correlate(prof_ext,main_temp,mode='valid')
  main_pos.append(corr.argmax())
main_pos = np.array(main_pos)
main_pos = main_pos[main_pos>0]

post_pos = []
for prof in profs[:, 100:]:
  prof_ext = np.concatenate((prof[-bins/2:],prof,prof[:bins/2]))
  corr = np.correlate(prof_ext,post_temp,mode='valid')
  post_pos.append(corr.argmax())
post_pos = np.array(post_pos)
post_pos = post_pos[post_pos>0]

DM = post_pos + 100 - main_pos
plt.plot(DM, 'ko')
plt.show()

