#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
# Set plot parameters for consistent styling
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

def calc_branchingratios(rate, template_length_HR=874, template_length_MR=219, template_length_LR=15, debug=True):
    """
    Compute branching ratios : Hp_bl_cor, Mp_bl, Ms_bl, Lp_bl_cor, and Ls_bl_cor based on the given rate.
    
    Parameters:
    rate (float or numpy array): The rate value(s) for computation.
    template_length_HR (int): Template length for high resolution. Default is 874.
    template_length_MR (int): Template length for medium resolution. Default is 219.
    template_length_LR (int): Template length for low resolution. Default is 15.
    
    Returns:
    tuple: Hp_bl_cor, Mp_bl, Ms_bl, Lp_bl_cor, Ls_bl_cor
    """
    
    ################### setting parameters ###############################################
    # Hardware settings
    # template_length_HR = 1024
    # template_length_MR = 256
    # template_length_LR = 41
    clock = 12500 # 12.5kHz = 80us       
    dtHR = (1.0 / clock) * template_length_HR 
    dtMR = (1.0 / clock) * template_length_MR   
    dtLR = (1.0 / clock) * template_length_LR   

    exp_hr = np.exp(-1.0 * dtHR * rate)
    exp_mr = np.exp(-1.0 * dtMR * rate)             
    exp_lr = np.exp(-1.0 * dtLR * rate)             

    # Calculate the initial values (according to ProcSPIETemplate_A4_takedav6.pdf or https://ui.adsabs.harvard.edu/abs/2014SPIE.9144E..5BT/abstract)
    Hp_bl_cor = exp_hr * (exp_hr + 1 - exp_lr)
    Mp_bl = exp_hr * (exp_mr - exp_hr)
    Ms_bl = exp_mr * (exp_mr - exp_hr)
    Lp_bl_cor = exp_hr * (exp_lr - exp_mr)
    Ls_bl_cor = (1 - exp_mr) * (1 + exp_mr - exp_hr) - exp_hr * (1 - exp_lr)

    # Calculate checksum
    checksum = Hp_bl_cor + Mp_bl + Ms_bl + Lp_bl_cor + Ls_bl_cor

    # Normalize values
    Hp_bl_cor /= checksum
    Mp_bl /= checksum
    Ms_bl /= checksum
    Lp_bl_cor /= checksum
    Ls_bl_cor /= checksum

    # Calculate checksum confirmation
    checksum_confirm = Hp_bl_cor + Mp_bl + Ms_bl + Lp_bl_cor + Ls_bl_cor

    # Debug output
    if debug:
        print("checksum", checksum)
        print("checksum_confirm", checksum_confirm)
    return Hp_bl_cor, Mp_bl, Ms_bl, Lp_bl_cor, Ls_bl_cor

# Define the rate range for plotting
maxrate = 50.0 # Hz
plotrate_x = np.arange(0., maxrate, 0.1)

# Compute parameters for the rate range
Hp_bl_cor, Mp_bl, Ms_bl, Lp_bl_cor, Ls_bl_cor = calc_branchingratios(plotrate_x)

# Plot the results
plt.figure(figsize=(8, 7))

plt.plot(plotrate_x, Hp_bl_cor, label='Hp_bl_cor')
plt.plot(plotrate_x, Mp_bl, label='Mp_bl')
plt.plot(plotrate_x, Ms_bl, label='Ms_bl')
plt.plot(plotrate_x, Lp_bl_cor, label='Lp_bl_cor')
plt.plot(plotrate_x, Ls_bl_cor, label='Ls_bl_cor')

plt.xlabel('Rate (Hz)')
plt.ylabel('Branching Ratios')
plt.title('Branching Ratios vs Input Rate')
plt.legend()
plt.grid(True)
plt.savefig("resolve_bratio.png")
plt.show()
