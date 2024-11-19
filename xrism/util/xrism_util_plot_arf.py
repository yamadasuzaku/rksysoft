#!/usr/bin/env python 

import os
import glob
import argparse
from astropy.io import fits
import matplotlib.pyplot as plt
from os.path import commonprefix
import numpy as np

# Set matplotlib font to Roman style
plt.rc('font', family='serif')


def read_arf_file(file_path):
    """Read the .arf file and extract ENERG_LO, ENERG_HI, and SPECRESP."""
    with fits.open(file_path) as hdul:
        data = hdul[1].data
        energ_lo = data['ENERG_LO']
        energ_hi = data['ENERG_HI']
        specresp = data['SPECRESP']
        return energ_lo, energ_hi, specresp

def generate_common_prefix(file_list):
    """Determine the common prefix from a list of file names."""
    return commonprefix(file_list).rstrip('_')

def plot_arf_files(show_plot, x_max):
    """Find all .arf files in the current directory and plot their effective area curves."""
    arf_files = glob.glob("*.arf")
    
    if not arf_files:
        print("No .arf files found in the current directory. Exiting.")
        return

    common_prefix = generate_common_prefix([os.path.basename(f) for f in arf_files])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Store data for ratio calculations
    data_dict = {}

    for file_path in arf_files:
        energ_lo, energ_hi, specresp = read_arf_file(file_path)
        x = (energ_lo + energ_hi) / 2  # Compute the average energy
        y = specresp  # Effective area

        # Store data for later use
        data_dict[file_path] = (x, y)

        # Plot the effective area curves on the top subplot
        ax1.plot(x, y, label=file_path)

        # Find and print the maximum effective area and its energy
        max_area = max(y)
        max_energy = x[y.argmax()]
        print(f"File: {file_path}")
        print(f"Maximum Effective Area: {max_area:.2f} cm^2 at {max_energy:.2f} keV")

    # Identify the file with the highest maximum effective area
    max_file = max(data_dict, key=lambda f: max(data_dict[f][1]))
    max_x, max_y = data_dict[max_file]

    print(f"Reference File for Ratio Plot: {max_file}")

    # Plot the ratio curves on the bottom subplot
    for file_path, (x, y) in data_dict.items():
        if str(file_path).strip() == str(max_file).strip():
            print(file_path, max_file)
            continue
        # Avoid division by zero by setting invalid ratios to NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(max_y > 0, y / max_y, np.nan)
        ax2.plot(x, ratio, label=f"{file_path}")

    # Configure the top subplot
    ax1.set_ylabel("Effective Area (cm^2)")
    ax1.set_title("Effective Area Curves")
    ax1.legend()

    # Configure the bottom subplot
    ax2.set_xlabel("Energy (keV)")
    ax2.set_ylabel("Ratio")
    ax2.set_title(f"Ratio to {max_file}")
    ax2.legend()

    # Apply x-axis limit
    ax1.set_xlim(0, x_max)

    # Save the plot to a PNG file
    output_filename = f"{common_prefix}_effective_area.png"
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")

    if show_plot:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot effective area curves from .arf files.")
    parser.add_argument("--show", "-s", action="store_true", help="Show the plot after generation.")
    parser.add_argument("--xmax", type=float, default=20, help="Set the maximum x-axis value (default: 20 keV).")    
    args = parser.parse_args()

    plot_arf_files(args.show, args.xmax)

if __name__ == "__main__":
    main()
