#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

import numpy as np
import matplotlib.pyplot as plt

# Data from the table
phases = ['0.000-0.100', '0.100-0.150', '0.150-0.200', '0.200-0.300', '0.300-0.400', '0.400-0.500',
          '0.500-0.600', '0.600-0.650', '0.650-0.700', '0.700-0.775', '0.775-0.875', '0.875-1.000']

# Define the data with zeros for missing values and -1 for missing errors
q_values = [1.7, 0.8, 0.6, 0.9, -0.2, 0.0, 1.3, -7.1, -7.0, -3.4, -2.0, 3.7]
q_errors = [1.4, 1.8, 1.9, 1.4, 1.3, 1.7, 1.6, 2.2, 2.2, 1.9, 1.6, 1.6]

u_values = [-1.2, -4.4, -8.7, -5.1, -1.5, 0.1, 3.2, -3.5, -8.7, -6.4, -4.3, 1.6]
u_errors = [1.4, 1.8, 1.9, 1.4, 1.3, 1.7, 1.6, 2.2, 2.2, 1.9, 1.6, 1.6]

nh_values = [6.0, 4.4, 4.9, 3.8, 4.0, 4.2, 2.1, 3.8, 2.5, 0.6, 4.0, 3.1]
nh_errors = [0.8, 1.4, 2.1, 0.9, 1.1, 1.2, 1.7, 1.5, 1.8, 0.6, 1.0, 2.1]

nh_tbpcf_values = [23.0, 18.5, 18.4, 19.9, 16.5, 20.5, 17.7, 21.3, 18.2, 15.3, 20.6, 14.9]
nh_tbpcf_errors = [2.2, 2.0, 3.4, 1.9, 1.4, 2.9, 1.7, 3.0, 2.1, 1.2, 2.2, 1.7]

f_cov_values = [0.82, 0.83, 0.77, 0.80, 0.811, 0.78, 0.83, 0.82, 0.84, 0.85, 0.81, 0.811]
f_cov_errors = [0.02, 0.03, 0.04, 0.02, 0.03, 0.03, 0.03, 0.03, 0.04, 0.05, 0.02, 0.05]

photon_index_values = [1.55, 1.26, 0.96, 0.75, 1.11, 1.08, 0.63, 0.98, 0.73, 0.30, 1.05, 0.95]
photon_index_errors = [0.16, 0.16, 0.22, 0.13, 0.11, 0.2, 0.13, 0.22, 0.18, 0.13, 0.15, 0.15]

pd_values = [2.3, 4.2, 7.3, 5.0, 4.6, 4.6, 2.6, 8.3, 9.2, 5.6, 4.4, 4.4]
pd_errors = [1.2, 1.6, 1.7, 1.2,   -1,   -1, 1.5, 2.0, 2.0, 1.6, 1.4, 1.4]

pa_values = [-22.0, -33.0, -42.4, -36.6,   0, 0, 37.7, -77.3, -64.2, -60.6, -58.7, 21.4]
pa_errors = [16.9,   11.3,   6.8,   7.2,  -1, -1, 17.2,   6.9,   6.1,   8.7,   9.2,  9.5]

# Pulse phase midpoints for plotting
phase_midpoints = [0.05, 0.125, 0.175, 0.25, 0.35, 0.45, 0.55, 0.625, 0.675, 0.725, 0.825, 0.95]

# Create subplots for each parameter
fig, axs = plt.subplots(8, 1, figsize=(12, 8))
plt.subplots_adjust(hspace=0, top=0.9)  # Adjust space between plots and top margin
plt.suptitle(f"Vela X-1: pulse phase vs. polarization parameters")

# Function to plot each parameter with errorbar, ignoring data with -1 error
def plot_parameter(ax, x, values, errors, ylabel, label):
    # Filter out data points where errors are -1
    mask = np.array(errors) != -1
    x_filtered = np.array(x)[mask]
    values_filtered = np.array(values)[mask]
    errors_filtered = np.array(errors)[mask]
    
    ax.errorbar(x_filtered, values_filtered, yerr=errors_filtered, fmt='o', label=label)
    ax.set_ylabel(ylabel)
    # ax.set_xticks(x_filtered)
    # ax.set_xticklabels(phases, rotation=45)
    ax.grid(True,alpha=0.5)
    ax.legend()

# Plot for q
plot_parameter(axs[0], phase_midpoints, q_values, q_errors, '$q$ (%)', '$q$')

# Plot for u
plot_parameter(axs[1], phase_midpoints, u_values, u_errors, '$u$ (%)', '$u$')

# Plot for N_H
plot_parameter(axs[2], phase_midpoints, nh_values, nh_errors, '$N_{H}$ \n ($10^{22}  \mathrm{~cm}^{-2}$)', '$N_{H}$')

# Plot for N_H_tbpcf
plot_parameter(axs[3], phase_midpoints, nh_tbpcf_values, nh_tbpcf_errors, '$N_{H, tbpcf}$ \n ($10^{22} \mathrm{~cm}^{-2}$)', '$N_{H, tbpcf}$')

# Plot for f_cov
plot_parameter(axs[4], phase_midpoints, f_cov_values, f_cov_errors, '$f_{cov}$', '$f_{cov}$')

# Plot for Photon Index
plot_parameter(axs[5], phase_midpoints, photon_index_values, photon_index_errors, 'Photon Index', 'Photon Index')

# Plot for PD
plot_parameter(axs[6], phase_midpoints, pd_values, pd_errors, 'PD (%)', 'PD')

# Plot for PA
plot_parameter(axs[7], phase_midpoints, pa_values, pa_errors, 'PA (deg)', 'PA')

# Set the labels and title
for ax in axs:
    ax.set_xlabel('Pulse Phase')

plt.savefig("ixpe_velax1_pulse.png")
plt.show()
