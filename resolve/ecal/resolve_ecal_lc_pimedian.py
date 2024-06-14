#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import datetime
import argparse
import sys
from matplotlib.colors import Normalize
import matplotlib.cm as cm

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
typename = ["Hp", "Mp", "Ms", "Lp", "Ls"]
icol = ["r", "b", "c", "m", "y"]
ishape = [".", "s", "D", "*", "x"]

def ev_to_pi(ev):
    """Convert energy in eV to PI units."""
    return (ev - 0.5) * 2

def pi_to_ev(pi):
    """Convert PI units to energy in eV."""
    return pi * 0.5 + 0.5

# Define the energy range
emin, emax = 2000, 10000  # Energy range in eV
pimin, pimax = ev_to_pi(emin), ev_to_pi(emax)
rebin = 10
binnum = int((pimax - pimin) / rebin)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot spectra from a FITS file.')
    parser.add_argument('filename', help='The name of the FITS file to process.')
    return parser.parse_args()

def open_fits_data(fname):
    try:
        return fits.open(fname)[1].data
    except FileNotFoundError:
        print("ERROR: File not found", fname)
        sys.exit()

def process_data(data, TRIGTIME_FLAG=False, AC_FLAG=False):
    time = data["TRIGTIME"] if TRIGTIME_FLAG else data["TIME"]
    itype = data["AC_ITYPE"] if AC_FLAG else data["ITYPE"]
    if len(time) == 0:
        print("ERROR: data is empty", time)
        sys.exit()
    additional_columns = [data[col] for col in ["PI", "RISE_TIME", "DERIV_MAX", "PIXEL"]]
    sortid = np.argsort(time)
    sorted_columns = [col[sortid] for col in [time, itype] + additional_columns]
    dtime = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in sorted_columns[0]])
    print(f"data from {dtime[0]} --> {dtime[-1]}")
    dt = np.diff(sorted_columns[0])
    return [column[:-1] for column in sorted_columns], dt

def plot_pi_time(pi, time, itype, pixel, outfname="mkpi.png", title="test", tbinsize=2000, plotflag=False):
    times = np.arange(time[0], time[-1], tbinsize)[:-1]

    for itype_ in [0]:
        print(f"..... itype={itype_}")
        # Plot histograms for each pixel
        for pixel_ in np.arange(36):
            print(f"..... pixel={pixel_}")
            plt.figure(figsize=(11, 7))
            plt.subplots_adjust(right=0.8)
            plt.xscale("linear")
            plt.yscale("log")
            plt.ylabel("Counts/bin")
            plt.xlabel("PI (eV)")
            plt.grid(alpha=0.8)
            plt.title(title + " TYPE = " + typename[itype_])
            numspec = 0
            specave = []
            spectime = []

            for i, t in enumerate(times):
                idcut = np.where((pixel == pixel_) & (itype == itype_) & (time > t) & (time <= t + tbinsize))
                pi_filtered = pi[idcut]
                if len(pi_filtered) == 0:
                    print("warning: data is empty for pixel =", pixel_)
                    continue
                numspec += 1
                print(f"..... i={i}, t={t}")
                hist, binedges = np.histogram(pi_filtered, bins=binnum, range=(pimin, pimax))
                ene = 0.5 * (binedges[1:] + binedges[:-1]) * 0.5 + 0.5
                event_number = len(pi_filtered)
                print(np.mean(hist))
                specave.append(np.median(pi_filtered))
                spectime.append(t + 0.5 * tbinsize)
                plt.errorbar(ene, hist, fmt='-', label=f"P{pixel_}_time={t}" + "("+str(event_number)+ "c)")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=6)
            plt.xlim(emin, emax)
            ofname = f"fig_{typename[itype_]}_pixel{pixel_:02d}_{outfname}"
            plt.savefig(ofname)
            if plotflag:
                plt.show()
            print(f"..... {ofname} is created.")

            specave = np.array(specave)
            spectime = np.array(spectime)

            plt.figure(figsize=(11, 7))
            plt.xscale("linear")
            plt.yscale("linear")
            plt.xlabel("Time (s)")
            plt.ylabel("each segment / observation average")
            plt.grid(alpha=0.8)
            plt.title(title + " TYPE = " + typename[itype_] + f" Pixel = {pixel_}")
            plt.errorbar(spectime, specave/np.mean(specave), fmt='-o', label=f"P{pixel_} itype={itype_}")
            ofname = f"fig_timemean_{typename[itype_]}_pixel{pixel_:02d}_{outfname}"
            plt.savefig(ofname)
            if plotflag:
                plt.show()
            print(f"..... {ofname} is created.")

def main():
    args = parse_arguments()
    data = open_fits_data(args.filename)
    processed_data, dt = process_data(data)
    time, itype, pi, rise_time, deriv_max, pixel = processed_data
    plot_pi_time(pi, time, itype, pixel, outfname=f"ql_plotspectime_{args.filename.replace('.evt', '').replace('.gz', '')}.png", title=f"Spectra of {args.filename}")

if __name__ == "__main__":
    main()
