#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse
from astropy.io import fits
from pathlib import Path

# Set plot parameters for consistent styling
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

# Define a pixel map for the 6x6 grid
pixel_map = np.array([
    [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],  # DETY
    [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6],  # DETX
    [12, 11, 9, 19, 21, 23, 14, 13, 10, 20, 22, 24, 16, 15, 17, 18, 25, 26, 8, 7, 0, 35, 33, 34, 6, 4, 2, 28, 31, 32, 5, 3, 1, 27, 29, 30]  # PIXEL
])

# Maximum number of colors to be used in plots
CMAX = 20
colors = plt.cm.tab20(np.linspace(0, 1, CMAX))

import numpy as np

def get_deriv(ydata):
    # Initialize an empty list to store the derivative values
    ydatalc = []
    
    # Create an array of indices from 0 to the length of ydata
    numarray = np.arange(0, len(ydata), 1)

    # Iterate over each element in ydata
    for onei, oneydata in enumerate(ydata):
        # Get the previous 8 elements
        if onei >= 8:
            prey = ydata[onei-8:onei]
        else:
            prey = []

        # Get the next 8 elements
        if onei + 8 < len(ydata):
            posty = ydata[onei:onei+8]
        else:
            posty = []

        # Calculate the long derivative as the difference of means multiplied by 8
        if len(prey) > 0 and len(posty) > 0:
            derivLong = (np.mean(posty) - np.mean(prey)) * 8
        else:
            derivLong = 0
        
        # Calculate the final derivative, using floor function for adjustment
        derivative = np.floor((derivLong + 2.) / 4.)

        # Append the derivative to the list
        ydatalc.append(derivative)

    # Convert the list to a numpy array
    npydatalc = np.array(ydatalc)

    # Find the maximum and minimum derivative values within the range of 100 to 200 indices
    deriv_max = np.amax(npydatalc[np.where((numarray > 100) & (numarray < 200))])
    deriv_min = np.amin(npydatalc[np.where((numarray > 100) & (numarray < 200))])

    # Find the indices of the maximum and minimum derivative values
    deriv_max_i = numarray[np.where(npydatalc == deriv_max)][0]
    deriv_min_i = numarray[np.where(npydatalc == deriv_min)][0]

    # Determine the peak derivative value
    if np.abs(deriv_max) > np.abs(deriv_min):
        peak = deriv_max
    else:
        peak = deriv_min

    # Find the index of the peak derivative value
    deriv_peak_i = numarray[np.where(npydatalc == peak)][0]

    # Return the computed derivatives and their related values
    return (npydatalc, deriv_max, deriv_max_i, deriv_min, deriv_min_i, peak, deriv_peak_i)


def plot_data_6x6(prevt, itypes, dumptext=False, plotflag=False, usetime=False, prevflag=False, xlims=None, ylims=None, usederiv=False):
    """
    Plots the pulse record data from a FITS file in a 6x6 grid format.

    Args:
    prevt (str): Path to the input FITS file.
    itypes (list): List of itype values to filter the data.
    dumptext (bool): Flag to dump x_time and pulse data to NPZ files.
    plotflag (bool): Flag to display the plot.
    usetime (bool): Flag to use time for the x-axis.
    prevflag (bool): Flag to plot previous interval values as text.
    xlims (tuple): Tuple specifying the x-axis limits (xmin, xmax).
    ylims (tuple): Tuple specifying the y-axis limits (ymin, ymax).
    usederiv (tuple): Flag to use derivative.
    """
    # Open the FITS file and extract relevant data
    data = fits.open(prevt)
    pulse = data[1].data['PULSEREC']
    pix_num_list = data[1].data['PIXEL']
    itype_list = data[1].data['ITYPE']
    prev_list = data[1].data['PREV_INTERVAL']
    data.close()

    # Define the time resolution
    dt = 80.0e-6
    x_time = np.arange(0, pulse.shape[-1], 1) * dt

    for itype in itypes:
        itype = int(itype)

        if pulse[itype_list == itype].size > 0:
            ph_max = pulse[itype_list == itype].max()
        else:
            ph_max = 0

        itype_str = get_itype_str(itype)

        path = Path(prevt)
        ofile = f'pulserecord_{path.stem}_{itype_str}.png'

        print(f'Load file name = {prevt}')
        print(f'ITYPE = {itype}:{itype_str}')

        # Create subplots for a 6x6 grid
        fig, ax = plt.subplots(6, 6, figsize=(16, 9), sharex=True, sharey=True)
        plt.subplots_adjust(right=0.9)
        for e in range(36):
            dety = 6 - pixel_map.T[e][0]
            detx = pixel_map.T[e][1] - 1
            pixel = pixel_map.T[e][2]
            pix_mask = pix_num_list == pixel
            itype_mask = itype_list[pix_mask] == itype

            pulse_p_pix_itype = pulse[pix_mask][itype_mask]
            prev_p_pix_itype  = prev_list[pix_mask][itype_mask]

            num_of_evt = len(pulse_p_pix_itype)
            if num_of_evt > 0:
                print(f'PIXEL={pixel:02d}, N={num_of_evt}')

            ax[dety, detx].text(-0.05, 1.02, f'P{pixel:02d}', fontsize=8, ha='center', va='center', transform=ax[dety, detx].transAxes)
            ax[dety, detx].text(-0.05, 0.92, f'#{num_of_evt}', fontsize=8, ha='center', va='center', transform=ax[dety, detx].transAxes)

            if usederiv: # plot derivative
                for j, (pulserecord, prev) in enumerate(zip(pulse_p_pix_itype, prev_p_pix_itype)):
                    color = colors[j % len(colors)]  # get color 
                    one_deriv, deriv_max, deriv_max_i, deriv_min, deriv_min_i, deriv_peak, deriv_peak_i = get_deriv(pulserecord)                                        

                    if usetime:
                        ax[dety, detx].plot(x_time, one_deriv, color=color)
                    else:
                        ax[dety, detx].plot(one_deriv, color=color)
                    # plot text
                    if prevflag:
                        ax[dety, detx].text(0.5, 0.9 - 0.1 * j, f'{prev}', fontsize=8, color=color, ha='center', va='center', transform=ax[dety, detx].transAxes)

            else: # plot raw pulse
                for j, (pulserecord, prev) in enumerate(zip(pulse_p_pix_itype, prev_p_pix_itype)):
                    color = colors[j % len(colors)]  # get color 
                    # plot pulse
                    if usetime:
                        ax[dety, detx].plot(x_time, pulserecord, color=color)
                    else:
                        ax[dety, detx].plot(pulserecord, color=color)
                    # plot text
                    if prevflag:
                        ax[dety, detx].text(0.5, 0.9 - 0.1 * j, f'{prev}', fontsize=8, color=color, ha='center', va='center', transform=ax[dety, detx].transAxes)

            if dumptext and num_of_evt > 0:
                if usederiv:
                    dump_to_npz(f"deriv_{path.stem}_{itype_str}_dety{dety}_detx{detx}_pixel{pixel}.npz", x_time, pulse[pix_mask][itype_mask])
                else:
                    dump_to_npz(f"{path.stem}_{itype_str}_dety{dety}_detx{detx}_pixel{pixel}.npz", x_time, pulse[pix_mask][itype_mask])

        # Set common labels and limits
        for i in range(6):
            ax[5, i].set_xlabel(r'Time (1 tick = 80us)', fontsize=10)
            if usederiv:
                ax[i, 0].set_ylabel(r'Derivative', fontsize=10)
            else:
                ax[i, 0].set_ylabel(r'PulseRecord', fontsize=10)

            if xlims:
                ax[5, i].set_xlim(xlims)
            if ylims:
                ax[i, 0].set_ylim(ylims)

        if usederiv: # plot derivative
            plt.suptitle(f"Derivative of Pulse Record: {prevt}, ITYPE={itype}")
            ofile = "deriv_" + ofile
        else: # plot raw palse 
            plt.suptitle(f"Pulse Record: {prevt}, ITYPE={itype}")

        plt.tight_layout()
        plt.savefig(ofile)
        print(f'.... {ofile} is saved.')

        if plotflag: 
            plt.show()

def get_itype_str(itype):
    """
    Returns a string representation for a given itype.

    Args:
    itype (int): The itype value.

    Returns:
    str: The string representation of the itype.
    """
    itype_dict = {-1: 'all', 0: 'Hp', 1: 'Mp', 2: 'Ms', 3: 'Lp', 4: 'Ls', 5: 'BL', 6: 'EL', 7: '--'}
    return itype_dict.get(itype, 'unknown')

def dump_to_npz(filename, x_time, pulses):
    """
    Dumps the x_time and pulse data to a NPZ file.

    Args:
    filename (str): The output filename.
    x_time (numpy.ndarray): The time array.
    pulses (numpy.ndarray): The pulse data.
    """
    np.savez(filename, x_time=x_time, pulses=pulses)

def load_from_npz(filename):
    """
    Loads the x_time and pulse data from a NPZ file.

    Args:
    filename (str): The input filename.

    Returns:
    tuple: The time array and pulse data.
    """
    data = np.load(filename)
    return data['x_time'], data['pulses']

def parse_limits(limit_str):
    """
    Parses a comma-separated string into a tuple of floats.

    Args:
    limit_str (str): Comma-separated string of limits.

    Returns:
    tuple: Tuple of float limits.
    """
    try:
        print(limit_str)
        return tuple(map(float, limit_str.split(',')))
    except ValueError:
        print(limit_str)
        raise argparse.ArgumentTypeError("Limits must be comma-separated floats")

def main():
    """
    Main function to parse arguments and call the plot_data_6x6 function.
    """
    parser = argparse.ArgumentParser(
      description='This program is to plot pulserecord',
      epilog='''
        Example 1) just plot 
        resolve_ana_pixel_pr_plot.py xa000114000rsl_a0pxpr_uf_fillprenext.evt
        Example 2) Create a new file:
        resolve_ana_pixel_pr_plot.py xa000114000rsl_a0pxpr_uf_fillprenext_below30.evt --prevflag --xlims 0,500 --ylims=-5000,-2000      
        [Note] --ylims=-5000,-2000 should work but --ylims -5000,-2000 NOT work. 
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('prevt', type=str, help='Input FITS file')
    parser.add_argument('--itypelist', type=str, default='0,1,2,3,4', help='Comma-separated list of itype values (default: 0,1,2,3,4)')
    parser.add_argument('--dumptext', action='store_true', help='Flag to dump x_time and pulse data to NPZ files')
    parser.add_argument('--plot', '-p', action='store_true', help='Flag to plot')
    parser.add_argument('--xlims', type=parse_limits, help='Comma-separated x-axis limits (xmin,xmax)')
    parser.add_argument('--ylims', type=parse_limits, help='Comma-separated y-axis limits (ymin,ymax)')
    parser.add_argument('--prevflag', action='store_true', help='Flag to plot previous interval values as text')
    parser.add_argument('--deriv', action='store_true', help='Flag to plot derivative')
    parser.add_argument('--usetime', action='store_true', help='Flag to usetime')

    args = parser.parse_args()
    itypes = [int(itype) for itype in args.itypelist.split(',')]
    
    plot_data_6x6(args.prevt, itypes, args.dumptext, \
        plotflag=args.plot, usetime=args.usetime, prevflag=args.prevflag, xlims=args.xlims, ylims=args.ylims, usederiv=args.deriv)

if __name__ == "__main__":
    main()