import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates

import plot_LOFAR_profile
import plot_DM_LOFAR
import plot_Timing
import plot_JB_profile

#date_min = datetime.date(2013,07,01)
date_min = datetime.date(2013,06,01)

f, axarr = plt.subplots(4, sharex=True, figsize=(10,40))

plot_LOFAR_profile.plot(axarr[0], date_min=date_min)
plot_DM_LOFAR.plot(axarr[2], date_min=date_min)
plot_Timing.plot(axarr[3])
plot_JB_profile.plot(axarr[1])

for ax in axarr:
  ax.set_xlim((date_min, datetime.date(2016, 05, 05)))

ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d-%m-%y'))
ax.set_xlabel('Date')

plt.show()









