#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import datetime
import astropy.io.fits as fits

# MJD reference day 01 Jan 2019 00:00:00
MJD_REFERENCE_DAY = 58484
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')
TYPES = ["Hp", "Mp", "Ms", "Lp", "Ls"]

def plot_histogram(ax, data, col, col_name, col_cut, label, xscale, yscale, bins, range_min, range_max):
    """Helper function to plot histogram on a given axis."""
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_ylabel("Number of events")
    ax.grid(alpha=0.8)
    ax.hist(data[col > col_cut], bins=bins, range=(range_min, range_max), histtype='step', label=label)
    ax.legend()

def plot_dt(dt, col, col_name, col_cut, pixel_cut, target_fname, xscale, yscale, bin_count, range_min, range_max, outfname, xlabel, plot_flag, trigtime_flag, itype_cut):
    """Plot the histogram of delta time and save the figure."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    title = f"created from {target_fname} PIXEL = {pixel_cut}_{TYPES[itype_cut]} COLUMN = {col_name}"
    ax1.set_title(title)
    label = f"{col_name} > {col_cut} (event num = {len(dt[col > col_cut])})"
    plot_histogram(ax1, dt, col, col_name, col_cut, label, xscale, yscale, bin_count, range_min, range_max)
    
    xlabel = "TRIGTIME (sec) from the next event" if trigtime_flag else xlabel
    ax2.set_xlabel(xlabel)
    label = f"{col_name} <= {col_cut} (event num = {len(dt[col <= col_cut])})"
    plot_histogram(ax2, dt, col, col_name, col_cut, label, xscale, yscale, bin_count, range_min, range_max)

    plt.tight_layout()
    plt.savefig(outfname)
    print(f"..... {outfname} is created.")
    if plot_flag:
        plt.show()

def plot_time_data(time, col, col_name, pixel_cut, outfname, plot_flag, trigtime_flag, itype_cut):
    """Plot the time series data and save the figure."""
    dtime = [REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(date_sec)) for date_sec in time]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid(alpha=0.8)
    title = f"{outfname} ITYPE = {TYPES[itype_cut]} PIXEL = {pixel_cut}"
    ax.set_title(title)
    ax.errorbar(dtime, col, fmt=".", ms=1)
    
    xlabel = "TRIGTIME from " + str(dtime[0]) if trigtime_flag else "TIME from " + str(dtime[0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(col_name)
    
    plt.savefig(outfname)
    print(f"..... {outfname} is created.")
    if plot_flag:
        plt.show()

def plot_dt_col(dt, col, col_name, pixel_cut, range_min, range_max, outfname, plot_flag, trigtime_flag, itype_cut):
    """Plot the delta time vs column data and save the figure."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid(alpha=0.8)
    title = f"{outfname} ITYPE = {TYPES[itype_cut]} PIXEL = {pixel_cut}"
    ax.set_title(title)
    ax.errorbar(dt, col, fmt=".", ms=5)
    
    xlabel = "TRIGTIME (sec) from the next event" if trigtime_flag else "TIME (sec) from the next event"
    ax.set_xlabel(xlabel)
    
    ax.set_yscale("log")
    ax.set_xlim(range_min, range_max)
    ax.set_ylabel(col_name)
    
    plt.savefig(outfname)
    print(f"..... {outfname} is created.")
    if plot_flag:
        plt.show()

def process_data(target_fname, trigtime_flag, itype_cut, col_name):
    """Process data from FITS file without pixel cut."""
    with fits.open(target_fname) as hdul:
        data = hdul[1].data
    time_col = "TRIGTIME" if trigtime_flag else "TIME"
    time, col, itype = data[time_col], data[col_name], data["ITYPE"]
    
    sorted_indices = np.argsort(time)
    time = time[sorted_indices]
    col = col[sorted_indices]
    itype = itype[sorted_indices]
    
    cut_indices = np.where(itype == itype_cut)[0]
    time, col, itype = time[cut_indices], col[cut_indices], itype[cut_indices]
    
    print(f"Number of events: {len(sorted_indices)} --> ITYPE == {itype_cut} --->  {len(cut_indices)} ")
    
    dt = np.diff(time)
    return dt, col[:-1], time[:-1], itype[:-1]

def process_data_pixelcut(target_fname, trigtime_flag, itype_cut, col_name, pixel_cut):
    """Process data from FITS file with pixel cut."""
    with fits.open(target_fname) as hdul:
        data = hdul[1].data
    time_col = "TRIGTIME" if trigtime_flag else "TIME"
    time, col, itype, pixel = data[time_col], data[col_name], data["ITYPE"], data["PIXEL"]
    
    sorted_indices = np.argsort(time)
    time, col, itype, pixel = time[sorted_indices], col[sorted_indices], itype[sorted_indices], pixel[sorted_indices]
    
    cut_indices = np.where((itype == itype_cut) & (pixel == pixel_cut))[0]
    time, col, itype, pixel = time[cut_indices], col[cut_indices], itype[cut_indices], pixel[cut_indices]
    
    print(f"Number of events: {len(sorted_indices)} --> ITYPE == {itype_cut} --->  {len(cut_indices)} ")
    
    dt = np.diff(time)
    return dt, col[:-1], time[:-1], itype[:-1], pixel[:-1]

def main(target_fname, xscale, yscale, plot_flag, trigtime_flag, itype_cut, col_name, col_cut, pixel_cut):
    """Main function to process data and generate plots."""
    dt, col, time, itype, pixel = process_data_pixelcut(target_fname, trigtime_flag, itype_cut, col_name, pixel_cut)

    fname_tag = target_fname.replace(".evt", "").replace(".gz", "")
    trigtime_suffix = "_TRIGTIME" if trigtime_flag else "_TIME"
    
    outfname_dt = f"deltaT_histogram_{xscale}_{yscale}_{fname_tag}{trigtime_suffix}_{TYPES[itype_cut]}_pixel{pixel_cut:02d}.png"
    plot_dt(dt, col, col_name, col_cut, pixel_cut, target_fname, xscale, yscale, 100000, 1e-7, 0.03, outfname_dt, "TIME (sec) from the next event", plot_flag, trigtime_flag, itype_cut)
    
    outfname_time_data = f"lightcurve_all_PIXEL_all_types_{fname_tag}{trigtime_suffix}_{TYPES[itype_cut]}_pixel{pixel_cut:02d}.png"
    plot_time_data(time, col, col_name, pixel_cut, outfname_time_data, plot_flag, trigtime_flag, itype_cut)

    outfname_dt_col = f"deltaT_col_{fname_tag}{trigtime_suffix}_{TYPES[itype_cut]}_pixel{pixel_cut:02d}.png"
    plot_dt_col(dt, col, col_name, pixel_cut, 0, 0.01, outfname_dt_col, plot_flag, trigtime_flag, itype_cut)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='This program is used to check deltaT distribution for each pixel',
      epilog='''
        Example:
        resolve_ana_pixel_deltat_distribution_detail.py xa000114000rsl_p0px1000_cl.evt --itype 4 -p
      ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)    
    parser.add_argument('fname', help='The name of the FITS file to process.')
    parser.add_argument('--xscale', type=str, choices=['log', 'linear'], default='linear', help='X-axis scale: log or linear')
    parser.add_argument('--yscale', type=str, choices=['log', 'linear'], default='linear', help='Y-axis scale: log or linear')
    parser.add_argument('--plot', '-p', action='store_true', default=False, help='Flag to display plot')
    parser.add_argument('--trigtime', action='store_true', default=False, help='Flag to use TRIGTIME instead of TIME')
    parser.add_argument('--itypecut', '-i', type=int, default=0, help='ITYPE for cut')
    parser.add_argument('--colname', '-c', type=str, default="PHA", help='Column name')
    parser.add_argument('--colcut', '-t', type=float, default=12250, help='Column cut value')
    parser.add_argument('--pixelcut', '-x', type=int, default=5, help='Pixel cut value')
    args = parser.parse_args()

    # Print the arguments to confirm their values
    print(f"FITS file: {args.fname}")
    print(f"X-axis scale: {args.xscale}")
    print(f"Y-axis scale: {args.yscale}")
    print(f"Plot flag: {args.plot}")
    print(f"TRIGTIME flag: {args.trigtime}")
    print(f"ITYPE cut: {args.itypecut}")
    print(f"Column name: {args.colname}")
    print(f"Column cut value: {args.colcut}")
    print(f"Pixel cut value: {args.pixelcut}")

    main(args.fname, args.xscale, args.yscale, args.plot, args.trigtime, args.itypecut, args.colname, args.colcut, args.pixelcut)
