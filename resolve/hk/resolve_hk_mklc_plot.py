#!/usr/bin/env python

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
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
parser.add_argument('--hduname', type=str, help='Time bin size for light curves', default="HK_SXS_TEMP")
parser.add_argument('--colname', type=str, help='Time bin size for light curves', default="CAMC_CT0")
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
hduname = args.hduname
colname = args.colname

# Define a function to process the data from the FITS file
def process_data(fname, hduname, colname, TRIGTIME_FLAG=False, AC_FLAG=False):
    # Open the FITS file and extract data
    data = fits.open(fname)[hduname].data

    # Select appropriate time column based on flag
    time = data["TRIGTIME"] if TRIGTIME_FLAG else data["TIME"]

    if len(time) == 0:
        print("ERROR: data is empty", time)
        sys.exit()

    # Extract additional data columns
    hkdata = data[colname]

    # Sort data by time
    sortid = np.argsort(time)
    time = time[sortid]
    hkdata = hkdata[sortid]    
    dtime = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in time])

    # Print the date range of the data
    print(f"data from {dtime[0]} --> {dtime[-1]}")

    return time, dtime, hkdata

# Define a function to plot the light curve
def plot_lc(time, dtime, hkdata, outfname="mklc_hk.png", title="test"):
    # Set up the plot
    plt.figure(figsize=(10, 7))
    plt.xscale("linear")
    plt.yscale("log")
    plt.ylabel("HK data")
    plt.xlabel("TIME since " + str(dtime[0]))
    plt.grid(alpha=0.8)
    plt.figtext(0.1, 0.95, title)

    plt.errorbar(time, hkdata, fmt=".", label=colname)
    plt.legend(bbox_to_anchor=(0., 1.01, 1., .101), loc='lower left', ncol=10, borderaxespad=0., fontsize=8)

    # Save and show the plot
    plt.savefig(outfname)
    print("..... " + outfname + " is created.")
    plt.show()

# Main section to execute the functions
# Prepare the output filename
outftag = fname.replace(".hk1", "").replace(".hk1.gz", "")
outfname = "hk_plot_" + outftag + ".png"

# Process the data and plot the light curve
time, dtime, hkdata = process_data(fname, hduname, colname, TRIGTIME_FLAG=False, AC_FLAG=False)
plot_lc(time, dtime, hkdata, outfname=outfname, title="Light curves of pixels from " + fname)
