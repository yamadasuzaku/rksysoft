#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

# Data from the table
energy_bins = ['2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '2-8']
q_values = [0.5, 0.2, -0.1, -1.2, -0.3, -2.7, -0.6]
q_errors = [1.2, 0.8, 0.8, 1.0, 1.2, 2.0, 0.5]
u_values = [3.8, -1.4, -2.7, -5.6, -4.1, -9.4, -3.7]
u_errors = [1.2, 0.8, 0.8, 1.0, 1.2, 2.0, 0.5]
pd_values = [3.9, 1.4, 2.7, 5.7, 4.1, 9.7, 3.7]
pd_errors = [1.2, 0.8, 0.8, 1.0, 1.2, 2.0, 0.5]
pa_values = [41.4, -40.7, -46.4, -50.9, -47.3, -53.1, -49.9]
pa_errors = [9.1, 16.1, 8.8, 4.8, 8.3, 6.0, 4.1]

# Convert energy bins to midpoints for plotting
energy_midpoints = [
    (2 + 3) / 2, (3 + 4) / 2, (4 + 5) / 2, (5 + 6) / 2, 
    (6 + 7) / 2, (7 + 8) / 2, (2 + 8) / 2
]

# Create subplots for each parameter
fig, axs = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
plt.suptitle(f"Vela X-1: energy vs. polarization parameters")
plt.subplots_adjust(hspace=0, top=0.9)  # Adjust space between plots and top margin
# Plot for q
axs[0].errorbar(energy_midpoints, q_values, yerr=q_errors, fmt='o', label='$q$ (%)')
axs[0].set_ylabel('$q$ (%)')
# axs[0].set_xticks(energy_midpoints)
# axs[0].set_xticklabels(energy_bins)
axs[0].grid(True,alpha=0.5)

# Plot for u
axs[1].errorbar(energy_midpoints, u_values, yerr=u_errors, fmt='o', label='$u$ (%)', color='orange')
axs[1].set_ylabel('$u$ (%)')
# axs[1].set_xticks(energy_midpoints)
# axs[1].set_xticklabels(energy_bins)
axs[1].grid(True,alpha=0.5)

# Plot for PD
axs[2].errorbar(energy_midpoints, pd_values, yerr=pd_errors, fmt='o', label='PD (%)', color='green')
axs[2].set_ylabel('PD (%)')
# axs[2].set_xticks(energy_midpoints)
# axs[2].set_xticklabels(energy_bins)
axs[2].grid(True,alpha=0.5)

# Plot for PA
axs[3].errorbar(energy_midpoints, pa_values, yerr=pa_errors, fmt='o', label='PA (deg)', color='red')
axs[3].set_ylabel('PA (deg)')
# axs[3].set_xticks(energy_midpoints)
# axs[3].set_xticklabels(energy_bins)
axs[3].grid(True,alpha=0.5)

# Set the labels and title
for ax in axs:
    ax.set_xlabel('Energy (keV)')

#fig.tight_layout()
plt.savefig("ixpe_velax1_ave.png")
plt.show()
