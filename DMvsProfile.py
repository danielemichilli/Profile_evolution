import plot_Profile_evolution
import DMvariation
import matplotlib.pyplot as plt

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

date_list, observations = plot_Profile_evolution.all_obs()  

plot_Profile_evolution.plot_image(date_list, observations, ax=ax1)
DMvariation.DMvariation(ax=ax2, horizontal=False)

plt.show()

