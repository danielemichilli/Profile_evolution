from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates

import plot_LOFAR_profile
import plot_JB_profile


f, axarr = plt.subplots(2, sharey=True, figsize=(10,40))

plot_LOFAR_profile.plot(axarr[0])
plot_JB_profile.plot(axarr[1])

for ax in axarr:
  ax.set_ylim((datetime(2010, 07, 22),datetime(2016, 05, 05)))

ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d-%m-%y'))
ax.set_ylabel('Date')

plt.show()









