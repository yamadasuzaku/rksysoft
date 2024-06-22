#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.time import Time
import datetime
import argparse
import sys
from matplotlib.colors import LogNorm, Normalize
import matplotlib.cm as cm
import os

# Constants
MJD_REFERENCE_DAY = 58484
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')
TIME_50MK = 150526800.0

# Plotting Configuration
plt.rcParams['font.family'] = 'serif'
usercmap = plt.get_cmap('jet')
cNorm = Normalize(vmin=0, vmax=35)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=usercmap)

# Type Information
itypename = [0, 1, 2, 3, 4]
typename = ["Hp", "Mp", "Ms", "Lp", "Ls"]
icol = ["r", "b", "c", "m", "y"]
ishape = [".", "s", "o", "*", "x"]

def ev_to_pi(ev):
    """Convert energy in eV to PI units."""
    return (ev - 0.5) * 2

def pi_to_ev(pi):
    """Convert PI units to energy in eV."""
    return pi * 0.5 + 0.5

# Define the energy range
emin, emax = 0, 20000  # Energy range in eV
pimin, pimax = ev_to_pi(emin), ev_to_pi(emax)
rebin = 10
binnum = int((pimax - pimin) / rebin)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot gain history from a FITS file.')
    parser.add_argument('filename', help='The name of the FITS file to process.')
    return parser.parse_args()

def open_fits_data(fname):
    try:
        return fits.open(fname)[1].data
    except FileNotFoundError:
        print("ERROR: File not found", fname)
        sys.exit()

def process_data(data):
    time = data["TIME"]
    additional_columns = [data[col] for col in ["PIXEL", "TEMP_FIT", "CHISQ", "NEVENT"]]
    sortid = np.argsort(time)
    sorted_columns = [col[sortid] for col in [time] + additional_columns]
    dtime = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in sorted_columns[0]])
    print(f"data from {dtime[0]} --> {dtime[-1]}")
    return sorted_columns, dtime

def save_pixel_data_to_csv(pixel, time, temp_fit):
    # Create output directory if it doesn't exist
    output_dir = "ghf_check"
    os.makedirs(output_dir, exist_ok=True)

    # Save data for each pixel
    for pixel_ in np.arange(36):
        pixelcut = (pixel == pixel_)
        px_time = time[pixelcut]
        px_temp_fit = temp_fit[pixelcut]
        if len(px_time) == 0:
            print("warning: data is empty for pixel =", pixel_)
            continue

        df = pd.DataFrame({
            'px_time': px_time,
            'px_temp_fit': px_temp_fit
        })
        csv_filename = os.path.join(output_dir, f"pixel_{pixel_:02d}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Data for pixel {pixel_} saved to {csv_filename}")

def plot_ghf(time, dtime, pixel, temp_fit, outfname="mkpi.png", title="test"):
    
    plt.figure(figsize=(11, 7))
    plt.subplots_adjust(right=0.8)  # make the right space bigger
    plt.xscale("linear")
    plt.yscale("linear")
    plt.ylabel("TEMP_FIT : Effective Temperature (mK)")
    plt.xlabel("Time (s) from "  + str(dtime[0]))
    plt.grid(alpha=0.8)
    plt.title(title)

    # Plot histograms for each pixel
    for pixel_ in np.arange(36):
        pixelcut = (pixel == pixel_)
        px_time = time[pixelcut]
        px_temp_fit = temp_fit[pixelcut]        
        if len(px_time) == 0:
            print("warning: data is empty for pixel =", pixel_)
            continue

        color = scalarMap.to_rgba(pixel_)
        event_number = len(px_time)
        plt.errorbar(px_time, px_temp_fit, color=color, alpha=0.8, fmt=ishape[pixel_%5]+'-', label=f"P{pixel_}" + "("+str(event_number)+ "c)")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=6)
    ofname = f"fig_{outfname}"
    plt.savefig(ofname)
    plt.show()
    print(f"..... {ofname} is created.")

def main():
    args = parse_arguments()
    data = open_fits_data(args.filename)
    processed_data, dtime = process_data(data)
    time, pixel, temp_fit, chisq, nevent = processed_data  # data unpack

    # Save pixel data to CSV
    save_pixel_data_to_csv(pixel, time, temp_fit)

    plot_ghf(time, dtime, pixel, temp_fit, outfname=f"ql_plotghf_{args.filename.replace('.ghf', '').replace('ghf.gz', '')}.png", title=f"Gain history of {args.filename}")

if __name__ == "__main__":
    main()
