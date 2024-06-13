#!/usr/bin/env python

import argparse
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os

def parse_arguments():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Check if NULLINT in Ms events.',
        epilog='Usage example: python resolve_util_check_nullint.py xa000126000rsl_p0px5000_uf.evt'
    )
    parser.add_argument('filename', type=str, help='FITS file name')
    parser.add_argument('--binnum', '-b', type=int, default=200, help='Number of bins for histogram')
    parser.add_argument('--margin', '-m', type=int, default=10, help='Margin for histogram range')
    return parser.parse_args()

def configure_plot():
    # Configure plot parameters
    params = {'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 7}
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update(params)

def read_fits_data(filename):
    # Open the FITS file and read data
    print(f"Opening FITS file: {filename}")
    hdulist = fits.open(filename)
    data = hdulist[1].data
    pha_org = data["PHA"]
    pha2_org = data["PHA2"]
    itype_org = data["ITYPE"]
    return pha_org, pha2_org, itype_org

def filter_data_by_pha2(pha_org, pha2_org, itype_org):
    # Filter data where PHA2 equals the null value 2147483647
    print("Filtering data with PHA2 == 2147483647 (NULL)")
    null_positions = np.where(pha2_org == 2147483647)
    pha = pha_org[null_positions]
    itype = itype_org[null_positions]

    # Calculate the number of events before and after filtering
    event_numbers = {}
    unique_itypes = np.unique(itype_org)
    for itype_ in unique_itypes:
        original_count = np.sum(itype_org == itype_)
        filtered_count = np.sum(itype == itype_)
        event_numbers[itype_] = (original_count, filtered_count)

    return pha, itype, event_numbers

def save_event_numbers(event_numbers, filename, typename):
    # Save event numbers to a text file and print to standard output
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    print(f"Saving event numbers to {txt_filename}")
    with open(txt_filename, 'w') as f:
        for itype_, counts in event_numbers.items():
            original_count, filtered_count = counts
            if itype_ < len(typename):
                line = f"ITYPE {typename[itype_]}: original={original_count}, filtered={filtered_count}\n"
                print(line.strip())
                f.write(line)

def plot_histogram(pha, itype, event_numbers, binnum, margin, filename, typename, itypename):
    # Create and configure the plot
    plt.figure(figsize=(11, 7))
    plt.subplots_adjust(right=0.8)
    plt.xscale("linear")
    plt.yscale("log")
    plt.ylabel("Counts/bin")
    plt.xlabel("PHA")
    plt.grid(alpha=0.8)
    plt.title(f"{filename} filtered with pha2 == 2147483647 (NULL)")
    
    xmin, xmax = -margin, 65535 + margin

    # Plot histogram for each itype
    for itype_ in itypename:
        if itype_ in event_numbers:
            print(f"Processing itype {typename[itype_]} (ITYPE={itype_})")
            typecut = (itype == itype_)
            subpha = pha[typecut]
            hist, binedges = np.histogram(subpha, bins=binnum, range=(xmin, xmax))
            bincenters = 0.5 * (binedges[1:] + binedges[0:-1])

            # Retrieve and display event counts in the legend
            original_count, filtered_count = event_numbers[itype_]
            label = f"{typename[itype_]} ({filtered_count}/{original_count} events)"
            
            plt.errorbar(bincenters, hist, yerr=np.sqrt(hist), fmt='-', label=label, alpha=0.5)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=6)
    
    # Save the plot as a PNG file
    ofname = f"fig_{filename.split('.')[0]}.png"
    print(f"Saving plot to {ofname}")
    plt.savefig(ofname)
    plt.show()
    print(f"..... {ofname} is created.")

def main():
    # Parse command-line arguments
    args = parse_arguments()
    # Configure plot settings
    configure_plot()

    # Read data from the FITS file
    pha_org, pha2_org, itype_org = read_fits_data(args.filename)
    # Filter data where PHA2 equals the null value
    pha, itype, event_numbers = filter_data_by_pha2(pha_org, pha2_org, itype_org)
    
    # Save event numbers to a text file
    typename = ["Hp", "Mp", "Ms", "Lp", "Ls"]
    save_event_numbers(event_numbers, args.filename, typename)
    
    # Plot histogram of filtered data
    itypename = [0, 1, 2, 3, 4]
    plot_histogram(pha, itype, event_numbers, args.binnum, args.margin, args.filename, typename, itypename)

if __name__ == "__main__":
    main()
