#!/usr/bin/env python

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
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
parser.add_argument('--timebinsize', type=float, help='Time bin size for light curves', default=100.0)
parser.add_argument('--xmin', type=float, help='Minimum X-axis value')
parser.add_argument('--xmax', type=float, help='Maximum X-axis value')
parser.add_argument('--ymin', type=float, help='Minimum Y-axis value')
parser.add_argument('--ymax', type=float, help='Maximum Y-axis value')
parser.add_argument('--debug', '-d', action='store_true', help='デバッグモード')
parser.add_argument('filename', help='The name of the FITS file to process.')
args = parser.parse_args()

# Define arrays for different types of observations
itypename = [0, 1, 2, 3, 4]
typename = ["Hp", "Mp", "Ms", "Lp", "Ls"]
icol = ["r", "b", "c", "m", "y"]
ishape = [".", "s", "D", "*", "x"]

# Extract arguments
fname = args.filename
timebinsize = args.timebinsize

# Define a function to process the data from the FITS file
def process_data(fname, TRIGTIME_FLAG=False, AC_FLAG=False):
    data = fits.open(fname)[1].data
    time = data["TRIGTIME"] if TRIGTIME_FLAG else data["TIME"]
    if len(time) == 0:
        print("ERROR: data is empty", time)
        sys.exit()
    itype = data["AC_ITYPE"] if AC_FLAG else data["ITYPE"]
    pha, rise_time, deriv_max = data["PHA"], data["RISE_TIME"], data["DERIV_MAX"]
    sortid = np.argsort(time)
    time, pha, itype, rise_time, deriv_max = (
        time[sortid], pha[sortid], itype[sortid], rise_time[sortid], deriv_max[sortid]
    )
    dtime = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in time])
    print(f"data from {dtime[0]} --> {dtime[-1]}")
    return np.diff(time), time[:-1], dtime[:-1], pha[:-1], itype[:-1], rise_time[:-1], deriv_max[:-1]

# Define a function to compute a fast light curve
def fast_lc(tstart, tstop, binsize, x):
    times = np.arange(tstart, tstop, binsize)[:-1]
    n_bins = len(times)
    x_lc, y_lc, x_err, y_err = np.zeros(n_bins), np.zeros(n_bins), np.full(n_bins, 0.5 * binsize), np.zeros(n_bins)
    x = np.sort(x)
    for i, t in enumerate(times):
        start, end = np.searchsorted(x, [t, t + binsize])
        count = end - start
        x_lc[i], y_lc[i] = t + 0.5 * binsize, count / binsize
        y_err[i] = np.sqrt(count) / binsize
    return x_lc, x_err, y_lc, y_err

# Define a function to plot the light curve
def plot_lc(time, itype, outfname="mklc_binened.png", title="test", debug=False):
    plt.figure(figsize=(10, 7))
    plt.xscale("linear")
    plt.yscale("linear")
    plt.ylabel(f"Counts/s (binsize = {timebinsize}s)")
    plt.xlabel("TIME")
    plt.grid(alpha=0.8)
    plt.figtext(0.1, 0.95, title)
    
    for itype_ in itypename:
        typecut = itype == itype_
        time_ = time[typecut]
        if len(time_) == 0:
            print("ERROR: data is empty", time_)
            continue
        x_lc, x_err, y_lc, y_err = fast_lc(time[0], time[-1], timebinsize, time_)
        dtime_lc = [REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(date_sec)) for date_sec in x_lc]
        plt.errorbar(dtime_lc, y_lc, yerr=y_err, fmt=icol[itype_]+ishape[itype_], 
                     label=typename[itype_] + "(N=" + str(len(time_)) + ")")
        plt.legend(bbox_to_anchor=(0., 1.01, 1., .101), loc='lower left', ncol=10, borderaxespad=0., fontsize=8)
    
    # Set axis limits if provided, otherwise auto-scale
    if args.xmin is not None and args.xmax is not None:
        plt.xlim(args.xmin, args.xmax)
    if args.ymin is not None and args.ymax is not None:
        plt.ylim(args.ymin, args.ymax)
    
    plt.savefig(outfname)
    print("..... " + outfname + " is created.")
    if debug:
        plt.show()

# Main section to execute the functions
outftag = fname.replace(".evt", "").replace(".gz", "")
outfname = "ql_mklc_binned_sorted_itype_" + outftag + ".png"
dt, time, dtime, pha, itype, rise_time, deriv_max = process_data(fname)
plot_lc(time, itype, outfname=outfname, title="Light curves of pixels from " + fname, debug=args.debug)
