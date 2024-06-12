#!/usr/bin/env python

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
params = {'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 7}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

from astropy.io import fits
from astropy.time import Time
import datetime
import argparse
import sys

# Define global variables
MJD_REFERENCE_DAY = 58484
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')  # Reference time in Modified Julian Day

# Define constants
TIME_50MK = 150526800.0  # Corresponds to 2023-10-09 05:00:00

# Set up argument parser to allow user input from command line
parser = argparse.ArgumentParser(description='Plot lightcurves from a FITS file.')
parser.add_argument('filelist', help='The name of the FITS file list to process.')
parser.add_argument('--timebinsize', type=float, help='Time bin size for light curves', default=100.0)
parser.add_argument('--itypenames', '-y', type=str, help='Comma-separated list of itype', required=True)
parser.add_argument('--plotpixels','-p',  type=str, help='Comma-separated list of plot pixels', required=True)
parser.add_argument('--output', type=str, help='Output file name', default='mklc_binened.png')
args = parser.parse_args()

# Parse the plotpixels argument to a list of integers
plotpixels = list(map(int, args.plotpixels.split(',')))
itypenames = list(map(int, args.itypenames.split(',')))

# Define the color list based on the number of plotpixels
colors = plt.cm.get_cmap('tab10', len(plotpixels)).colors
ishape = [".", "s", "D", "*", "x"]

# Print the command-line arguments to help with debugging
args_dict = vars(args)
print("Command-line arguments:")
for arg, value in args_dict.items():
    print(f"{arg}: {value}")

# Extract arguments
filelist = args.filelist
# ファイルを読み込み、リストに保存するコード
with open(filelist, 'r') as file:
    event_list = [line.strip() for line in file]
# 読み込んだリストを表示
print(f"Event list: {event_list}")
print(f"plotpixels: {plotpixels}")
print(f"itypenames: {itypenames}")

timebinsize = args.timebinsize

# Define a function to process the data from the FITS file
def process_data(fname, TRIGTIME_FLAG=False, AC_FLAG=False):
    print(f"Processing data from {fname}...")
    # Open the FITS file and extract data
    data = fits.open(fname)[1].data

    # Select appropriate time column based on flag
    time = data["TRIGTIME"] if TRIGTIME_FLAG else data["TIME"]

    if len(time) == 0:
        print("ERROR: data is empty", time)
        sys.exit()

    # Select appropriate type column based on flag
    itype = data["AC_ITYPE"] if AC_FLAG else data["ITYPE"]

    # Extract additional data columns
    pha, rise_time, deriv_max, pixel = data["PHA"], data["RISE_TIME"], data["DERIV_MAX"], data["PIXEL"]

    # Sort data by time
    sortid = np.argsort(time)
    time = time[sortid]
    pha = pha[sortid]
    itype = itype[sortid]
    rise_time = rise_time[sortid]    
    deriv_max = deriv_max[sortid]    
    pixel = pixel[sortid]        
    dtime = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in time])

    # Print the date range of the data
    print(f"Data from {dtime[0]} to {dtime[-1]}")

    # Prepare data for light curve calculation
    dt = np.diff(time) 
    # Remove the last element of each array to match the diff array length
    time, dtime, pha, itype, rise_time, deriv_max, pixel = time[:-1], dtime[:-1], pha[:-1], itype[:-1], rise_time[:-1], deriv_max[:-1], pixel[:-1]

    return dt, time, dtime, pha, itype, rise_time, deriv_max, pixel

# Define a function to compute a fast light curve
def fast_lc(tstart, tstop, binsize, x):
    """Compute a fast light curve using a given time series."""
    times = np.arange(tstart, tstop, binsize)[:-1]
    n_bins = len(times)

    # Initialize arrays for the light curve data
    x_lc, y_lc = np.zeros(n_bins), np.zeros(n_bins)
    x_err = np.full(n_bins, 0.5 * binsize)
    y_err = np.zeros(n_bins)

    # Ensure the time array is sorted
    x = np.sort(x)

    # Compute counts for each bin
    for i, t in enumerate(times):
        start, end = np.searchsorted(x, [t, t + binsize])
        count = end - start

        x_lc[i] = t + 0.5 * binsize
        y_lc[i] = count / binsize
        y_err[i] = np.sqrt(count) / binsize

    return x_lc, x_err, y_lc, y_err

# Define a function to plot the light curve without saving the figure
def plot_lc(ax, time, itype, pixel, obsid, oname):
    print(f"Plotting light curve for {oname} (Obs ID: {obsid})...")
    # Set up the plot
    # Plot each type of observation
    for itype_ in itypenames:
        for j, pix in enumerate(plotpixels):
            print(f"for pixel {pix} and type {itype_} from obsid {obsid}")
            cutid = np.where( (itype == itype_) & (pixel == pix))[0]
            time_ = time[cutid]
            if len(time_) == 0:
                print(f"ERROR: data is empty for pixel {pix} and type {itype_}")
                continue

            x_lc, x_err, y_lc, y_err = fast_lc(time[0], time[-1], timebinsize, time_)

            zcutid = np.where(y_lc > 0)[0]
            x_lc  = x_lc[zcutid]
            x_err = x_err[zcutid]
            y_lc  = y_lc[zcutid]
            y_err = y_err[zcutid]           

            dtime_lc = [REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(date_sec)) for date_sec in x_lc]
            ax.errorbar(dtime_lc, y_lc, yerr=y_err, fmt=ishape[0], label=f"pix={pix}, type={itype_}, id={obsid}")

    # Adjust the plot to make room for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.92, box.height])
    # Place the legend to the right of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=7)

# Main section to execute the functions
# Prepare the output filename

# Outside the function, create the figure and axis, then call the function and save the plot
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xscale("linear")
ax.set_yscale("linear")
ax.set_ylabel(f"Counts/s (binsize = {timebinsize}s)")
ax.set_xlabel("TIME")
ax.grid(alpha=0.8)

for fname in event_list:
    outftag = fname.replace(".evt", "").replace(".gz", "")
    outfname = "ql_mklc_binned_sorted_itype_" + outftag + ".png"
    # Process the data and plot the light curve
    head = fits.open(fname)[1].header
    obsid = head["OBS_ID"]
    oname = head["OBJECT"]
    dt, time, dtime, pha, itype, rise_time, deriv_max, pixel = process_data(fname, TRIGTIME_FLAG=False, AC_FLAG=False)
    plot_lc(ax, time, itype, pixel, obsid, oname)

plt.savefig(args.output)
print(f"Output file {args.output} is created.")
plt.show()
