import numpy as np
import psrchive
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

archive = '/data1/Daniele/B2217+47/Products/L32532/L32532_correctDM.clean.F.ar'

#Load profiles
load_archive = psrchive.Archive_load(archive)
load_archive.remove_baseline()
times = load_archive.get_data()[:1850]
times = np.sum(times,axis=(1,2))
times = times[1:]
prof = times.sum(axis=0)
roll_idx = len(prof)-np.argmax(prof)
times = np.roll(times, roll_idx, axis=1)
prof = np.roll(prof, roll_idx)
med = np.mean(times[:,200:800],axis=1)[:, np.newaxis]
times = times - med
prof = times.sum(axis=0)

#Estimation of errors
err_mean = np.std(times[:,200:800]) #np.abs(times[:,200:800]).mean()  #Average error on the bins, defined as deviation from null mean
err_main = err_mean * np.sqrt(530 - 491)
err_post = err_mean * np.sqrt(557 - 530)
err_mean_prof = np.std(prof[200:800])
err_main_prof = err_mean_prof * np.sqrt(530 - 491)
err_post_prof = err_mean_prof * np.sqrt(557 - 530)

#Profiles alignment
times = np.roll(times, len(prof)/2, axis=1)
prof = np.roll(prof, len(prof)/2)

#Extract features
main = times[:,491:530].copy()
post = times[:,530:557].copy()
area_main = main.sum(axis=1)
area_post = post.sum(axis=1)
ratio = area_post / area_main
err_ratio = np.sqrt(err_post**2 / area_main**2 + area_main**2 * err_main**2 / area_main**4)

area_main_prof = prof[491:530].sum()
area_post_prof = prof[530:557].sum()
ratio_prof = area_post_prof / area_main_prof
err_ratio_prof = np.sqrt(err_post_prof**2 / area_main_prof**2 + area_main_prof**2 * err_main_prof**2 / area_main_prof**4)

ratio_scaled = ratio / ratio_prof
err_ratio_scaled = np.sqrt(err_ratio**2 / ratio_prof**2 + err_ratio_prof**2 * ratio**2 / ratio_prof**4)

#Fit histogram
hist, bin_edges = np.histogram(ratio_scaled, bins=50)
hist = hist / float(hist.max())
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
def gauss(x, *p):
  A, mu, sigma = p
  return A*np.exp(-(x-mu)**2/(2.*sigma**2))
p0 = [1., 0., 1.]
coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
x = np.linspace(bin_centres.min(),bin_centres.max(),10000)
hist_fit = gauss(x, *coeff)
width = bin_centres[1] - bin_centres[0]
plt.bar(bin_centres, hist, align='center', width=width, color='w', lw=2.)
plt.plot(x, hist_fit, 'r--', lw=2.)
plt.xlabel('Normalized ratio')
plt.ylabel('Normalized counts')
print 'Fitted mean = ', coeff[1]   # 0.985
print 'Fitted standard deviation = ', coeff[2]   # 0.191
print 'Average error on measurements = ', err_ratio_scaled.mean()  # 0.222
textstr = '$\mu=%.2f$\n$\sigma=%.2f$'%(coeff[1], coeff[2])
plt.text(2.5, 1, textstr, verticalalignment='top', fontsize=14)
plt.xlim([-1.5,3.5])
plt.ylim([0,1.03])
plt.show()

#Statistics
mean = np.sum( ratio_scaled / err_ratio_scaled**2 ) / np.sum( 1. / err_ratio_scaled**2 )
mean_err = np.sqrt( 1. / np.sum( 1. / err_ratio_scaled**2 ) )

#Plot distribution in time
plt.errorbar(np.arange(ratio_scaled.size),ratio_scaled, fmt='ok', yerr=err_ratio_scaled)
plt.hline(mean, color='r')


"""
#Analysis excluding areas of postcursor smaller than error
idx = np.where((area_post > err_post) & (area_main > err_main))[0]
ratio_scaled_select = ratio_scaled[idx]
err_ratio_scaled_select = err_ratio_scaled[idx]
plt.hist(ratio_scaled_select,bins=50,color='r')
plt.xlabel('Ratio between areas (units of integrated profile ration)')
plt.ylabel('Counts')
plt.show()
hist, bin_edges = np.histogram(ratio_scaled_select, density=True, bins=50)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
hist_fit = gauss(bin_centres, *coeff)
plt.plot(bin_centres, hist, 'ok')
plt.plot(bin_centres, hist_fit, 'r--')
print 'Fitted mean = ', coeff[1]   # 0.986
print 'Fitted standard deviation = ', coeff[2]   # 0.188
print 'Average error on measurements = ', err_ratio_scaled_select.mean()  # 0.208
plt.show()
"""

