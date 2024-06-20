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

def plot_data_6x6(prevt, itypes, dumptext=False, plotflag=False, usetime=False, prevflag=False, xlims=None, ylims=None):
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
        ofile = f'pulserecode_{path.stem}_{itype_str}.png'

        print(f'Load file name = {prevt}')
        print(f'ITYPE = {itype}:{itype_str}')

        # Create subplots for a 6x6 grid
        fig, ax = plt.subplots(6, 6, figsize=(16, 9), sharex=True, sharey=True)
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
                dump_to_npz(f"{path.stem}_{itype_str}_dety{dety}_detx{detx}_pixel{pixel}.npz", x_time, pulse[pix_mask][itype_mask])

        # Set common labels and limits
        for i in range(6):
            ax[5, i].set_xlabel(r'Time (1 tick = 80us)', fontsize=10)
            ax[i, 0].set_ylabel(r'PulseRecord', fontsize=10)

            if xlims:
                ax[5, i].set_xlim(xlims)
            if ylims:
                ax[i, 0].set_ylim(ylims)

        plt.suptitle(f"Pulse Recode: {prevt}, ITYPE={itype}")
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
    parser = argparse.ArgumentParser(description="Process FITS files and plot data.")
    parser.add_argument('prevt', type=str, help='Input FITS file')
    parser.add_argument('--itypelist', type=str, default='0,1,2,3,4', help='Comma-separated list of itype values (default: 0,1,2,3,4)')
    parser.add_argument('--dumptext', action='store_true', help='Flag to dump x_time and pulse data to NPZ files')
    parser.add_argument('--plot', '-p', action='store_true', help='Flag to plot')
    parser.add_argument('--xlims', type=parse_limits, help='Comma-separated x-axis limits (xmin,xmax)')
    parser.add_argument('--ylims', type=parse_limits, help='Comma-separated y-axis limits (ymin,ymax)')
    parser.add_argument('--prevflag', action='store_true', help='Flag to plot previous interval values as text')

    args = parser.parse_args()
    itypes = [int(itype) for itype in args.itypelist.split(',')]
    
    plot_data_6x6(args.prevt, itypes, args.dumptext, plotflag=args.plot, usetime=args.plot, prevflag=args.prevflag, xlims=args.xlims, ylims=args.ylims)

if __name__ == "__main__":
    main()
