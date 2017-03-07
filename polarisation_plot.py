import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
import datetime 

mpl.rc('font',size=5)


data_folder = "/data1/Daniele/B2217+47/Analysis/plot_data"


def main():
  fig = plt.figure(figsize=(7,4))

  gs = gridspec.GridSpec(3, 4, width_ratios=[0.1,1,1,0.1], height_ratios=[0.5,1.,0.5], hspace=.3, wspace=.1)
  ax_bar_l = plt.subplot(gs[:, 0])
  ax_bar_r = plt.subplot(gs[:, -1])
  ax_top = plt.subplot(gs[2, 1])
  ax_mid = plt.subplot(gs[1, 1], sharex=ax_top)
  ax_bot = plt.subplot(gs[0, 1], sharex=ax_top)
  #ax_right = plt.subplot(gs[:, 2])

  left_plot(ax_bar_l, ax_top, ax_mid, ax_bot)
  right_plot(ax_bar_r, gs[:, 2])

  #fig.savefig('polarisation.png', papertype='a4', orientation='portrait', format='png')
  plt.show()



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
  colors = iter(cm.cool(c))
  x = np.linspace(-517./1024.*538.4688219194, (1024.-517.)/1024.*538.4688219194, 1024)
  for i in range(I.shape[0]):
    col = next(colors)
    ax_mid.plot(x, I[i],color=col)
    ax_mid.plot(x, L[i],color=col)  
    ax_mid.plot(x, V[i],color=col)    
  ax_mid.set_xlim([(450.-517.)/1024.*538.4688219194, (450.+145.-517.)/1024.*538.4688219194])
  ax_mid.set_ylabel('Flux (rel.)')
  ax_mid.tick_params(axis='x', labelbottom='off')

  #PA
  start = 479.
  duration = 70.
  PA_n = PA[:,start:start+duration]  #Select pahses based on pav output
  x = (np.arange(duration) + start - 517.) / 1024. * 538.4688219194

  #Top plot
  colors = iter(cm.cool(c))
  for i,n in enumerate(PA_n):
    col = next(colors)
    n -= n[duration/2]
    np.mod(n - 50 + 180, 180., out=n)
    ax_top.plot(x, n, 'o',color=col,label=str(days[i]))
  ax_top.set_ylabel('PA (deg)')
  ax_top.axvline(45.5/1024.*538.4688219194,color='k',linestyle='dashed')
  ax_top.axvline(56.5/1024.*538.4688219194,color='k',linestyle='dashed')
  ax_mid.set_ylabel('P.A. (deg)')
  ax_mid.tick_params(axis='x', labelbottom='off')
 
  #Relative flux of areas
  In = I[1:,450:623]
  Ln = L[1:,450:623]
  In = np.array([ np.sum(In[:,25:42],axis=1), np.sum(In[:,42:74],axis=1), np.sum(In[:,74:90],axis=1), np.sum(In[:,90:],axis=1) ])
  Ln = np.array([ np.sum(Ln[:,25:42],axis=1), np.sum(Ln[:,42:74],axis=1), np.sum(Ln[:,74:90],axis=1), np.sum(Ln[:,90:],axis=1) ])
  y = Ln / In

  #Bottom plot
  colors = iter(cm.cool(c))
  x = (np.array([8,33,57,106]) + 450 - 517) / 1024. * 538.4688219194
  for i,n in enumerate(y.T):
    col = next(colors)
    ax_bot.plot(x, n, 'o', color=col, label=str(days[i]))
  ax_bot.set_ylabel('L/I')
  ax_bot.set_xlabel('Phase (ms)')
  ax_bot.axvline((17.+450.-517.)/1024.*538.4688219194,color='k',linestyle='dashed')
  ax_bot.axvline((49.+450.-517.)/1024.*538.4688219194,color='k',linestyle='dashed')
  ax_bot.axvline((65.+450.-517.)/1024.*538.4688219194,color='k',linestyle='dashed')

  return



def right_plot(ax_bar, gs):
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

  lim=0.03
  plot_grid = gridspec.GridSpecFromSubplotSpec(3, 1, gs, wspace=0., hspace=0.)
  ax1 = plt.subplot(plot_grid[0])
  ax2 = plt.subplot(plot_grid[1])
  ax3 = plt.subplot(plot_grid[2])

  ax1.imshow(np.clip(imgI[:,450:650],0.,lim),cmap='hot',origin="lower",aspect='auto',interpolation='nearest')
  ax2.imshow(np.clip(imgL[:,450:650],0.,lim),cmap='hot',origin="lower",aspect='auto',interpolation='nearest')
  ax3.imshow(np.clip(imgV[:,450:650],0.,lim),cmap='hot',origin="lower",aspect='auto',interpolation='nearest')
 
 
  return

if __name__ == '__main__':
  main()
