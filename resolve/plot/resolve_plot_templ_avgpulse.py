#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from astropy.time import Time

# Constants
MJD_REFERENCE_DAY = 58484
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')

# Plot style settings for consistency
PLOT_PARAMS = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 7}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(PLOT_PARAMS)

# Color map settings
CMAX = 20
COLORS = plt.cm.hsv(np.linspace(0, 1, CMAX))

# Time step constant
TIME_STEP = 8e-5  # seconds

def plot_hk2(hk2_file, plotflag=False, selected_pixels=None):
    """
    Plot data from HK2 FITS file for selected pixels.

    Args:
        hk2_file (str): Path to the HK2 FITS file.
        plotflag (bool): If True, the plot will be displayed interactively.
        selected_pixels (list): List of pixel numbers to process. Default is all pixels.
    """

    print("Opening FITS file...")
    hdu = fits.open(hk2_file)
    ftag = hk2_file.replace(".hk2", "")
    
    # Extract metadata
    header = hdu[1].header
    date_obs = header["DATE-OBS"]

    # Initialize the figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 1, 1]}, figsize=(12, 7))
    plt.subplots_adjust(right=0.9)

    # Select all pixels if none are specified
    if selected_pixels is None:
        selected_pixels = range(36)

    # Read template data from FITS
    print("Reading template data...")
    data = hdu[1].data
    temphs = data["TEMPLATE_H"]
    tempms = data["TEMPLATE_M"]
    times = data["TIME"]
    pixels = data["PIXEL"]

    # Plot the template data for selected pixels
    for pixel in selected_pixels:
        print(f"Processing pixel {pixel} (Template data)...")
        pixel_indices = np.where(pixels == pixel)[0]
        time_data = times[pixel_indices]
        temph_data = temphs[pixel_indices]
        tempm_data = tempms[pixel_indices]

        for i in range(len(time_data)):
            color = COLORS[pixel % len(COLORS)]
            xvalh = np.arange(len(temph_data[i]))
            xvalm = np.arange(len(tempm_data[i]))
            if i == 0:
                ax1.errorbar(xvalh, temph_data[i], color=color, fmt="-", label=str(pixel), alpha=0.7)
            else:
                ax1.errorbar(xvalh, temph_data[i], color=color, fmt="-", alpha=0.7)                
            ax1.errorbar(xvalm, tempm_data[i], color=color, fmt="-", alpha=0.7)

        ax1.set_ylabel('Template')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., handletextpad=0.5, labelspacing=0.5)

    # Read average pulse and derivative data from FITS
    print("Reading average pulse and derivative data...")
    data = hdu[4].data
    avgs = data["AVGPULSE"]
    derivs = data["AVGDERIV"]
    times = data["TIME"]
    pixels = data["PIXEL"]

    # Plot the average pulse and derivative data for selected pixels
    for pixel in selected_pixels:
        print(f"Processing pixel {pixel} (Pulse and Derivative data)...")
        pixel_indices = np.where(pixels == pixel)[0]
        time_data = times[pixel_indices]
        avg_data = avgs[pixel_indices]
        deriv_data = derivs[pixel_indices]

        for i in range(len(time_data)):
            color = COLORS[pixel % len(COLORS)]
            xval = np.arange(len(avg_data[i]))

            ax2.errorbar(xval, avg_data[i], color=color, fmt="-", alpha=0.7)
            ax3.errorbar(xval, deriv_data[i], color=color, fmt="-", alpha=0.7)

        ax2.set_ylabel('AVGPULSE')
        ax3.set_ylabel('DERIV')

    plt.suptitle(f"{hk2_file} dateobs={date_obs}")
    output_filename = f'plot_hk2_pixel{pixel:02d}_{ftag}.png'
    plt.savefig(output_filename)
    print(f"..... {output_filename} is created.")

    # Show plot interactively if the flag is set
    if plotflag:
        plt.show()

    # Close the figure and FITS file
    plt.close(fig)
    print("Closing FITS file...")
    hdu.close()

def parse_pixel_range(pixel_range_str):
    """
    Parse a comma-separated list of pixel numbers into a list of integers.

    Args:
        pixel_range_str (str): Comma-separated pixel numbers.

    Returns:
        list: List of pixel numbers as integers.
    """
    if pixel_range_str is None:
        return None
    return [int(p) for p in pixel_range_str.split(',')]

def main():
    """
    Main function to parse command-line arguments and call plot_hk2 function.
    """
    parser = argparse.ArgumentParser(
        description='This program is to plot pulse records from HK2 FITS file.',
        epilog='Example: ./plot_hk2.py xa300065010rsl_000_fe55.hk2',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('hk2', type=str, help='Input HK2 file path')
    parser.add_argument('--plot', '-p', action='store_true', help='Display the plot interactively')
    parser.add_argument('--pixels', type=str, help='Comma-separated list of pixel numbers to process')

    args = parser.parse_args()
    selected_pixels = parse_pixel_range(args.pixels)
    
    print("Starting HK2 plotting process...")
    plot_hk2(args.hk2, plotflag=args.plot, selected_pixels=selected_pixels)
    print("Process completed.")

if __name__ == "__main__":
    main()
