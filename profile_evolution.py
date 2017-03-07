import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import LOFAR_span_all
import LOFAR_new
import long_term

mpl.rc('font',size=5)

fig = plt.figure(figsize=(7,8.5))

days_max, ax = LOFAR_span_all.plot(fig)
ratio = LOFAR_new.plot(fig, days_max=days_max, ax_ref=ax)
long_term.plot(fig, ratio=ratio)


#fig.savefig('evolution.png', papertype='a4', orientation='portrait', format='png')


plt.show()






