#!/usr/bin/env python

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import datetime
import argparse

# Define global variables
MJD_REFERENCE_DAY = 58484
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')  # Reference time in Modified Julian Day

# Define constants
TIME_50MK = 150526800.0  # Corresponds to 2023-10-09 05:00:00

# Set up argument parser to allow user input from command line
parser = argparse.ArgumentParser(description='Plot lightcurves from a FITS file.')
parser.add_argument('--timebinsize', type=float, help='Time bin size for light curves', default=100.0)
parser.add_argument('filename', help='The name of the FITS file to process.')
args = parser.parse_args()

# Define arrays for different types of observations
itypename = [0, 1, 2, 3, 4]
typename = ["Hp", "Mp", "Ms", "Lp", "Ls"]
icol = ["r", "b", "c", "m", "y"]
ishape = [".", "s", "D", "*", "x"]

# Print the command-line arguments to help with debugging
args_dict = vars(args)
print("Command-line arguments:")
for arg, value in args_dict.items():
    print(f"{arg}: {value}")

# Extract arguments
fname = args.filename
timebinsize = args.timebinsize

# Define a function to process the data from the FITS file
def process_data(fname, TRIGTIME_FLAG=False, AC_FLAG=False):
    # Open the FITS file and extract data
    data = fits.open(fname)[1].data

    # Select appropriate time column based on flag
    time = data["TRIGTIME"] if TRIGTIME_FLAG else data["TIME"]

    # Select appropriate type column based on flag
    itype = data["AC_ITYPE"] if AC_FLAG else data["ITYPE"]

    # Extract additional data columns
    pha, rise_time, deriv_max = data["PHA"], data["RISE_TIME"], data["DERIV_MAX"]

    # Sort data by time
    sortid = np.argsort(time)
    time = time[sortid]
    pha = pha[sortid]
    itype = itype[sortid]
    rise_time = rise_time[sortid]    
    deriv_max = deriv_max[sortid]    
    dtime = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in time])

    # Print the date range of the data
    print(f"data from {dtime[0]} --> {dtime[-1]}")

    # Prepare data for light curve calculation
    dt = np.diff(time) 
    # Remove the last element of each array to match the diff array length
    time, dtime, pha, itype, rise_time, deriv_max = time[:-1], dtime[:-1], pha[:-1], itype[:-1], rise_time[:-1], deriv_max[:-1]

    return dt, time, dtime, pha, itype, rise_time, deriv_max

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

# Define a function to plot the light curve
def plot_lc(time, itype, outfname="mklc_binened.png", title="test"):
    # Set up the plot
    plt.figure(figsize=(10, 7))
    plt.xscale("linear")
    plt.yscale("linear")
    plt.ylabel(f"Counts/s (binsize = {timebinsize}s)")
    plt.xlabel("TIME")
    plt.grid(alpha=0.8)
    plt.figtext(0.1, 0.95, title)

    # Plot each type of observation
    for itype_ in itypename:
        typecut = itype == itype_
        time_ = time[typecut]
        x_lc, x_err, y_lc, y_err = fast_lc(time[0], time[-1], timebinsize, time_)
        dtime_lc = [REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(date_sec)) for date_sec in x_lc]

        plt.errorbar(dtime_lc, y_lc, yerr=y_err, fmt=icol[itype_]+ishape[itype_], label=typename[itype_] + "(N=" + str(len(time_)) + ")")
        plt.legend(bbox_to_anchor=(0., 1.01, 1., .101), loc='lower left', ncol=10, borderaxespad=0., fontsize=8)

    # Save and show the plot
    plt.savefig(outfname)
    print("..... " + outfname + " is created.")
    plt.show()

# Main section to execute the functions
# Prepare the output filename
outftag = fname.replace(".evt", "").replace(".gz", "")
outfname = "ql_mklc_binned_sorted_itype_" + outftag + ".png"

# Process the data and plot the light curve
dt, time, dtime, pha, itype, rise_time, deriv_max = process_data(fname, TRIGTIME_FLAG=False, AC_FLAG=False)
plot_lc(time, itype, outfname=outfname, title="Light curves of pixels from " + fname)
