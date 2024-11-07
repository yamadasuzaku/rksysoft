#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import datetime
import argparse
import sys
import os 
from matplotlib.colors import LogNorm, Normalize
import matplotlib.cm as cm
import csv
import re

# Constants
MJD_REFERENCE_DAY = 58484
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')

# Plotting Configuration
plt.rcParams['font.family'] = 'serif'
usercmap = plt.get_cmap('jet')
cNorm = Normalize(vmin=0, vmax=35)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=usercmap)

# Type Information
itypename = [0, 1, 2, 3, 4]
typename = ["Hp", "Mp", "Ms", "Lp", "Ls"]
icol = ["r", "b", "c", "m", "y"]
ishape = [".", "s", "D", "*", "x"]

def ev_to_pi(ev):
    """Convert energy in eV to PI units."""
    return (ev - 0.5) * 2

def pi_to_ev(pi):
    """Convert PI units to energy in eV."""
    return pi * 0.5 + 0.5


def parse_filter_conditions(conditions):
    filters = []
    # '!=' を含めた正規表現パターンに変更
    condition_pattern = re.compile(r"(.*?)(==|<=|>=|<|>|!=)(.*)")
    for condition in conditions.split(","):
        match = condition_pattern.match(condition.strip())
        if match:
            col, op, value = match.groups()
            filters.append((col.strip(), op, float(value.strip())))
        else:
            raise ValueError(f"Invalid filter condition: {condition}")
    return filters

def apply_filters(data, filters):
    mask = np.ones(len(data), dtype=bool)
    for col, op, value in filters:
        if op == "==":
            mask &= (data[col] == value)
        elif op == "!=":
            mask &= (data[col] != value)
        elif op == "<":
            mask &= (data[col] < value)
        elif op == "<=":
            mask &= (data[col] <= value)
        elif op == ">":
            mask &= (data[col] > value)
        elif op == ">=":
            mask &= (data[col] >= value)
    return data[mask]

def plot_xhist(file_names, x_col, x_hdu, outfname, xmin, xmax, rebin, \
                  plotflag=False, debug=True, filters=False, ylin=False, calout=True, itype_cut=0):

    if x_col == "PI":
        # convert Energy to PI (b/c user specify energy as arguments.) 
        pimin, pimax = ev_to_pi(xmin), ev_to_pi(xmax)

    binnum = int((pimax - pimin) / rebin)

    # Create the figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
#    plt.subplots_adjust(right=0.8)  # make the right space bigger
    # Generate a color array using 'tab10' for up to 10 distinct colors
#    colors = plt.cm.tab10(np.linspace(0, 1, len(file_names)))    
    colors = plt.cm.Set1(np.linspace(0, 1, len(file_names)))    

#    colors = plt.cm.viridis(np.linspace(0, 1, len(file_names)))    

    total_hist = np.zeros(binnum)
    log_data = []
    histograms = []  # To store histograms for each file

    # First loop: Calculate total_hist by summing histograms
    for i, file_name in enumerate(file_names):
        print(f"..... {file_name} is opened.")
        with fits.open(file_name) as hdul:
            print(f"..... {x_col} is opened from HDU={x_hdu}")
            data = hdul[x_hdu].data
            pixel = data["PIXEL"] 
            itype = data["ITYPE"] 
            len_pre = len(pixel)
            if calout:
                print("only cal pixels")
                cutid = np.where( (pixel==12) & (itype==itype_cut))[0]
                data = data[cutid]
                pixel = pixel[cutid]
                cutp="only12"
            else:
                print("except for cal pixels")
                cutid = np.where( ((pixel < 12) | (pixel > 12)) & (itype==itype_cut))[0]
                data = data[cutid]
                pixel = pixel[cutid]
                cutp="cut12"

            len_post = len(pixel)
            print(f".... cut from {len_pre} to {len_post}")

            header = hdul[x_hdu].header
            obsid = header["OBS_ID"]
            target = header["OBJECT"]
            ontime = header["EXPOSURE"]
            print(target, ontime)

            if filters:
                print("..... filters applied")
                data = apply_filters(data, filters)
            xcolval = data[x_col]

            hist, binedges = np.histogram(xcolval, bins=binnum, range=(pimin, pimax))
            total_hist += hist  # Accumulate total histogram
            histograms.append((file_name, hist))  # Store each histogram for later ratio calculation
            if x_col == "PI":
                xval = 0.5 * (binedges[1:] + binedges[:-1]) * 0.5 + 0.5  # convert PI to energy (eV)
            else:
                xval = 0.5 * (binedges[1:] + binedges[:-1])

            realbinsize = xval[1] - xval[0] 
            event_number = len(xcolval)
            print(f".... Event number = {event_number}")

            log_data.append([file_name, obsid, target, ontime, event_number])            
            short_filename=os.path.basename(file_name)
            axs[0].errorbar(xval, hist, yerr=np.sqrt(hist), fmt='.', ms=1, alpha=0.9, color=colors[i], label=f"{short_filename} {target}" + f"({event_number:d} c)")
    
    axs[0].set_ylabel(f"Counts/bin (binsize={realbinsize})")
    if ylin:
        axs[0].set_yscale("linear")
    else:
        axs[0].set_yscale("log")

    axs[0].set_title(f"{outfname}_xmin{xmin}_xmax{xmax}_rebin{rebin}_{cutp}_itype{itype_cut}")

    # axs[0].axvline(1739.98, color='r', linestyle='-', label=r"Si K$\alpha$1", alpha=0.6, lw=0.5)
    # axs[0].axvline(1835.94, color='r', linestyle='-', label=r"Si K$\beta$", alpha=0.6, lw=0.5)
    # axs[0].axvline(1839., color='r', linestyle='--', label="Si Kedge", alpha=0.6, lw=0.5)
    # axs[0].axvline(2122.9, color='c', linestyle='-', label=r"Au M$\alpha$1", alpha=0.6, lw=0.5)
    # axs[0].axvline(2206., color='c', linestyle='--', label=r"Au M5", alpha=0.6, lw=0.5)
    # axs[0].axvline(2129.5, color='y', linestyle='-', label=r"escape(MnKa1-TeLa1)", alpha=0.6, lw=0.5)
    # axs[0].axvline(1869.2, color='y', linestyle='-', label=r"escape(MnKa1-TeLb1)", alpha=0.6, lw=0.5)

    axs[0].grid(alpha=0.1)
    axs[0].legend(loc="best", fontsize=7)

    axs[0].set_xlim(xmin, xmax)

    # Update xval for second panel
    xval = 0.5 * (binedges[1:] + binedges[:-1]) * 0.5 + 0.5 if x_col == "PI" else 0.5 * (binedges[1:] + binedges[:-1])

    # Second loop: Calculate and plot ratio histograms
    for i, (file_name, hist) in enumerate(histograms):
        ratio_hist = hist / total_hist  # Calculate ratio to the total histogram
        ratio_hist[total_hist == 0] = 0  # Avoid division by zero
        axs[1].errorbar(xval, ratio_hist, fmt='.-', ms=1, alpha=0.9, color=colors[i], label=f"Ratio {os.path.basename(file_name)}")

    axs[1].legend(loc="best", fontsize=7)

    if x_col == "PI":
        axs[1].set_xlabel("PI (eV)")
    else:
        axs[1].set_xlabel(x_col)
    axs[1].set_ylabel(f"Counts/bin (binsize={realbinsize})")

    axs[1].set_ylabel("Ratio to Total")
    if ylin:
        axs[1].set_yscale("linear")
    else:
        axs[1].set_yscale("log")
    # axs[1].axvline(1739.98, color='r', linestyle='-', label=r"Si K$\alpha$1", alpha=0.6, lw=0.5)
    # axs[1].axvline(1835.94, color='r', linestyle='-', label=r"Si K$\beta$", alpha=0.6, lw=0.5)
    # axs[1].axvline(1839., color='r', linestyle='--', label="Si Kedge", alpha=0.6, lw=0.5)
    # axs[1].axvline(2122.9, color='c', linestyle='-', label=r"Au M$\alpha$1", alpha=0.6, lw=0.5)
    # axs[1].axvline(2206., color='c', linestyle='--', label=r"Au M5", alpha=0.6, lw=0.5)
    # axs[1].axvline(2129.5, color='y', linestyle='-', label=r"escape(MnKa1-TeLa1)", alpha=0.6, lw=0.5)
    # axs[1].axvline(1869.2, color='y', linestyle='-', label=r"escape(MnKa1-TeLb1)", alpha=0.6, lw=0.5)

    axs[1].grid(alpha=0.1)
    axs[1].set_xlim(xmin, xmax)

    ofname = f"{outfname}_xmin{xmin}_xmax{xmax}_rebin{rebin}_{cutp}_itype{itype_cut}.png"
    fig.tight_layout()  # Adjust layout    
    plt.savefig(ofname)
    if plotflag:
        plt.show()
    print(f"..... {ofname} is created.")

    log_fname = f"{outfname}_xmin{xmin}_xmax{xmax}_rebin{rebin}_{cutp}_itype{itype_cut}_log.csv"
    with open(log_fname, 'w', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(['file_name', 'obsid', 'target', 'ontime', 'event_number'])
        log_writer.writerows(log_data)
    print(f"..... {log_fname} is created.")    


def main():

    parser = argparse.ArgumentParser(
        description='This program is used to check PI hist.',
        epilog='''
        Example:
        >cat eve.list
        xa300049010rsl_p0px3000_cl.evt
        xa300065010rsl_p0px1000_cl.evt
        >resolve_ana_pixel_compspec_pi.py eve.list --x_col PI -p --xmin 0 --xmax 20000 --rebin 250 -i 0 --filters "PIXEL==0"
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)    
    parser.add_argument("file_names", type=str, help="List of Path to the FITS file")
    parser.add_argument("--x_col", type=str, help="Column name for the x-axis", default="PI")
    parser.add_argument('--x_hdu', type=int, help='Number of FITS HDU for X', default=1)
    parser.add_argument("--filters", '-f', type=str, help="Comma-separated filter conditions in the format 'COLUMN==VALUE'", default="")
    parser.add_argument('--plot', '-p', action='store_true', default=False, help='Flag to display plot')
    parser.add_argument('--xmin', type=float, help='xmin', default=1700)
    parser.add_argument('--xmax', type=float, help='xmax', default=2400)
    parser.add_argument('--rebin', '-b', type=int, help='rebin', default=2)
    parser.add_argument('--ylin', action='store_true', default=False, help='Flag to ylinear')
    parser.add_argument('--calout', action='store_true', default=False, help='Flag to cal select')
    parser.add_argument('--itype_cut', '-i', type=int, help='ITYPE CUT (default = 0)', default=0)
    parser.add_argument("--outname", '-o', type=str, help="additional output name", default=None)


    args = parser.parse_args()

    xmin, xmax, rebin = args.xmin, args.xmax, args.rebin

    file_names = [ _.strip() for _ in open(args.file_names)]
    print(f'file_names = {file_names}')

    print(f'x_hdu = {args.x_hdu}')
    print(f'x_col = {args.x_col}')

    filter_conditions = parse_filter_conditions(args.filters) if args.filters else None
    title = f"{args.file_names} : filtered with {args.filters}"
    if args.outname == None:
        outfname = "hist1d_" + args.file_names.replace(",","_").replace(".","_p_") 
    else:
        outfname = "hist1d_" + args.file_names.replace(",","_").replace(".","_p_") + "_" + args.outname

    plot_xhist(file_names, args.x_col, args.x_hdu, outfname, xmin, xmax, rebin, \
              plotflag=args.plot, debug=True, filters=filter_conditions, ylin=args.ylin, calout = args.calout, itype_cut = args.itype_cut)

if __name__ == "__main__":
    main()
