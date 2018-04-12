import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
import datetime 
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from mjd2date import year
from matplotlib.ticker import ScalarFormatter 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import rcParams
rcParams['ps.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages

mpl.rc('font',size=8)
mpl.rcParams['lines.linewidth'] = .5


data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"
ref_date = datetime.date(2010, 7, 25)


def main():
  fig = plt.figure(figsize=(7,4))

  gs = gridspec.GridSpec(1, 2, wspace=.3)
  gsA = gridspec.GridSpecFromSubplotSpec(4, 1, gs[0], height_ratios=[0.1,0.5,1.,0.5], hspace=.1)
  gsB = gridspec.GridSpecFromSubplotSpec(4, 1, gs[1], height_ratios=[0.1,1.,1.,1.], hspace=.1)


  ax_bar_l = plt.subplot(gsA[0])
  ax_bar_r = plt.subplot(gsB[0])
  ax_mid_l = plt.subplot(gsA[2])
  ax_top_l = plt.subplot(gsA[1], sharex=ax_mid_l)
  ax_bot_l = plt.subplot(gsA[3], sharex=ax_mid_l)
  ax_top_r = plt.subplot(gsB[1])#, sharex=ax_mid_l)
  ax_mid_r = plt.subplot(gsB[2], sharex=ax_top_r, sharey=ax_top_r)
  ax_bot_r = plt.subplot(gsB[3], sharex=ax_top_r, sharey=ax_top_r)

  left_plot(ax_bar_l, ax_top_l, ax_mid_l, ax_bot_l)
  right_plot(ax_bar_r, ax_top_r, ax_mid_r, ax_bot_r)

  ax_mid_l.set_xlim([-24,60])
  ax_top_r.set_xlim([(450.-517.)/1024.*538.4688219194, (632.-517.)/1024.*538.4688219194])

  fig.subplots_adjust(left=.08,right=.98,bottom=.1,top=.9)
  #fig.savefig('polarisation.eps', papertype='a4', orientation='portrait', format='eps', dpi=200)

  pp = PdfPages('polarisation.pdf')
  pp.savefig(fig, papertype='a4', orientation='portrait', dpi=200)
  pp.close()


  plt.show()

  return



def left_plot(ax_bar, ax_top, ax_mid, ax_bot):
  #Load observations
  date = np.load(os.path.join(data_folder, 'pol_date.npy'))
  I, L, V, PA = np.load(os.path.join(data_folder, 'pol.npy'))

  #Relative intensities
  m = np.max(I,axis=1)
  I = (I.T / m).T
  L = (L.T / m).T
  V = (V.T / m).T

  #Middle plot
  d = date[0:] - date[1]
  days = [n.days for n in d]
  c = [float(n)/max(days) for n in days]
  colors = iter(cm.jet(c))
  x = np.linspace(-517./1024.*538.4688219194, (1024.-517.)/1024.*538.4688219194, 1024)
  for i in range(I.shape[0]):
    col = next(colors)
    ax_mid.plot(x, I[i]*100, color=col)
    ax_mid.plot(x, L[i]*100, color=col)#, linestyle='--')
    ax_mid.plot(x, V[i]*100, color=col)#, linestyle=':')
  ax_mid.set_ylabel('Flux density (% peak)')
  ax_mid.tick_params(axis='x', labelbottom='off')
  ax_mid.set_ylim([-3, 39])

  #PA
  start = 479.
  duration = 70.
  PA_n = PA[:,int(start):int(start+duration)]  #Select pahses based on pav output
  x = (np.arange(duration) + start - 517.) / 1024. * 538.4688219194

  #Top plot
  colors = iter(cm.jet(c))
  for i,n in enumerate(PA_n):
    col = next(colors)
    n -= n[int(duration/2)]
    np.mod(n - 50 + 180, 180., out=n)
    ax_top.plot(x, n, 'o',color=col,label=str(days[i]), markersize=1, markeredgewidth=0.)
  ax_top.set_ylabel('PA (deg)')
  ax_top.axvline((45.5+479.-517.)/1024.*538.4688219194,color='k')
  ax_top.axvline((56.5+479.-517.)/1024.*538.4688219194,color='k')
  ax_top.set_ylabel('PA (deg)')
  ax_top.tick_params(axis='x', labelbottom='off')
  ax_mid.axvline((45.5+479.-517.)/1024.*538.4688219194,color='k')
  ax_mid.axvline((56.5+479.-517.)/1024.*538.4688219194,color='k')
  ax_top.locator_params(axis='y', nbins=5)
  ax_top.set_ylim([0,180])
 
  #Relative flux of areas
  components = [475, 492, 524, 540, 632]
  In = np.array([ np.sum(I[:,components[0]:components[1]],axis=1), np.sum(I[:,components[1]:components[2]],axis=1), np.sum(I[:,components[2]:components[3]],axis=1), np.sum(I[:,components[3]:components[4]],axis=1) ])
  Ln = np.array([ np.sum(L[:,components[0]:components[1]],axis=1), np.sum(L[:,components[1]:components[2]],axis=1), np.sum(L[:,components[2]:components[3]],axis=1), np.sum(L[:,components[3]:components[4]],axis=1) ])
  y = Ln / In

  std = np.std(I[:,:components[0]],axis=1)
  errIn = np.array([std*(components[1]-components[0]), std*(components[2]-components[1]), std*(components[3]-components[2]), std*(components[4]-components[3])])
  std = np.std(L[:,:components[0]],axis=1)
  errLn = np.array([std*(components[1]-components[0]), std*(components[2]-components[1]), std*(components[3]-components[2]), std*(components[4]-components[3])])
  erry = np.sqrt((errLn/In)**2 + (errIn*Ln/In**2)**2)

  #Bottom plot
  colors = iter(cm.jet(c))
  x = (np.array([(475+492)/2., (492+524)/2., (524+540)/2., (540+632)/2.]) - 517) / 1024. * 538.4688219194
  for i,n in enumerate(y.T):
    col = next(colors)
    #ax_bot.plot(x, n, 'o', color=col, label=str(days[i]), markersize=2, markeredgewidth=0.)
    xi = x + (-y.shape[1] + i*2) / 5.
    ax_bot.errorbar(xi, n, fmt='o', capsize=0, color=col, label=str(days[i]), markersize=2, markeredgewidth=0., yerr=erry[:,i]/2.)
  ax_bot.set_ylabel('L/I')
  #for n in components:
  #  ax_bot.axvline((n-517.)/1024.*538.4688219194,color='k',linestyle='dashed')
  c = ['g', 'y']*3
  for i in range(len(components)-1):
    ax_bot.axvspan((components[i]-517.)/1024.*538.4688219194, (components[i+1]-517.)/1024.*538.4688219194, color=c[i], ls='-', ymin=.9)
    ax_mid.axvspan((components[i]-517.)/1024.*538.4688219194, (components[i+1]-517.)/1024.*538.4688219194, color=c[i], ls='-', ymax=.05)
  ax_bot.set_ylim([0.1,.99])
  ax_bot.locator_params(axis='y', nbins=5)
  ax_bot.set_xlabel('Phase (ms)')

  #Color bar
  norm = mpl.colors.Normalize(vmin=year(date[1]), vmax=year(date[-1]))
  cbar = mpl.colorbar.ColorbarBase(ax_bar, ticks=[2014.022,2014.5,2015, 2015.5], format='%.1f', cmap='jet', norm=norm, orientation='horizontal')
  cbar.ax.minorticks_on()
  cbar.ax.xaxis.set_label_position("top")
  cbar.set_label('Year')
  cbar.ax.tick_params(axis='both', labelleft='off', labelright='off', labeltop='on', labelbottom='off', right='off', left='off', bottom='off', top='on', which='both')

  ax_top.get_yaxis().set_label_coords(-0.12,0.5)
  ax_mid.get_yaxis().set_label_coords(-0.12,0.5)
  ax_bot.get_yaxis().set_label_coords(-0.12,0.5)

  def label(ax, number):
    at = AnchoredText(number, prop=dict(size=15), loc=1, frameon=True, pad=.1, borderpad=0.)
    ax.add_artist(at)
    return
  label(ax_top, "(a)")
  label(ax_mid, "(b)")
  label(ax_bot, "(c)")

  return



def right_plot(ax_bar, ax1, ax2, ax3):
  #Load observations
  date = np.load(os.path.join(data_folder, 'pol_date.npy'))
  I, L, V, PA = np.load(os.path.join(data_folder, 'pol.npy'))

  #Relative intensities
  I = (I.T / I.max(axis=1)).T
  L = (L.T / L.max(axis=1)).T
  V = (V.T / V.max(axis=1)).T
  
  #Plot image of polarization evolution
  days = (date[-1] - date[0]).days
  imgI = np.zeros((days,1024))
  imgL = np.zeros((days,1024))
  imgV = np.zeros((days,1024))

  for i,n in enumerate(imgI):
    idx = np.abs(date - date[0] - datetime.timedelta(i)).argmin()
    imgI[i] = I[idx]
    imgL[i] = L[idx]
    imgV[i] = V[idx]

  lim=0.04
  extent = [(450.-517.)/1024.*538.4688219194, (650.-517.)/1024.*538.4688219194, year(date[0]), year(date[-1])]
  ax1.imshow(np.clip(imgI[:,450:650],0.,lim),cmap='hot',origin="lower",aspect='auto',interpolation='nearest', extent=extent)
  ax2.imshow(np.clip(imgL[:,450:650],0.,lim),cmap='hot',origin="lower",aspect='auto',interpolation='nearest', extent=extent)
  ax3.imshow(np.clip(imgV[:,450:650],0.,lim),cmap='hot',origin="lower",aspect='auto',interpolation='nearest', extent=extent)

  ax2.set_ylabel("Year")
  ax3.set_xlabel("Phase (ms)") 
  ax1.tick_params(axis='x', labelbottom='off')
  ax2.tick_params(axis='x', labelbottom='off')

  #Color bar
  norm = mpl.colors.Normalize(vmin=0., vmax=lim*100)
  cbar = mpl.colorbar.ColorbarBase(ax_bar, ticks=np.linspace(0.,4.,9), format='%.1f', cmap='hot', norm=norm, orientation='horizontal')
  cbar.ax.minorticks_on()
  cbar.ax.xaxis.set_label_position("top")
  cbar.set_label('Flux density (% peak)')
  cbar.ax.tick_params(which='both', axis='both', labelleft='off', labelright='off', labeltop='on', labelbottom='off', right='off', left='off', bottom='off', top='on')

  ax1.ticklabel_format(useOffset=False)
  ax2.ticklabel_format(useOffset=False)
  ax3.ticklabel_format(useOffset=False)

  def label(ax, number):
    at = AnchoredText(number, prop=dict(size=15), loc=1, frameon=True, pad=.1, borderpad=0.)
    ax.add_artist(at)
    return
  label(ax1, "(d)")
  label(ax2, "(e)")
  label(ax3, "(f)")

  return

if __name__ == '__main__':
  main()
