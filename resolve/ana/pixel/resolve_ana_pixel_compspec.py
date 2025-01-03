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
    for condition in conditions.split(","):
        col, value = condition.split("==")
        filters.append((col.strip(), float(value.strip())))
    return filters

def apply_filters(data, filters):
    mask = np.ones(len(data), dtype=bool)
    for col, value in filters:
        mask &= (data[col] == value)
    return data[mask]

def plot_xhist(file_names, x_col, x_hdu, outfname, pimin, pimax, emin, emax, rebin, binnum, \
                  plotflag=False, debug=True, filters=False, ylin=False, calout=True, itype_cut=0):

    # Create the figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(right=0.7)  # make the right space bigger
    colors = plt.cm.viridis(np.linspace(0, 1, len(file_names)))    
    
    total_hist = np.zeros(binnum)

    log_data = []

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
            total_hist += hist
            if x_col == "PI":
                xval = 0.5 * (binedges[1:] + binedges[:-1]) * 0.5 + 0.5
            else:
                xval = 0.5 * (binedges[1:] + binedges[:-1])
            event_number = len(xcolval)
            log_data.append([file_name, obsid, target, ontime, event_number])            
            short_filename=os.path.basename(file_name)
            axs[0].errorbar(xval, hist, yerr=np.sqrt(hist), fmt='.', ms=1, alpha=0.9, color=colors[i], label=f"{short_filename} {target}" + f"({event_number/1e6:0.2f}Mcnt)")
    
    axs[0].set_xlabel("PI (eV)")
    axs[0].set_ylabel("Counts/bin")
    if ylin:
        axs[0].set_yscale("linear")
    else:
        axs[0].set_yscale("log")

    axs[0].set_title(f"{outfname}_emin{emin}_emax{emax}_rebin{rebin}_{cutp}_itype{itype_cut}")
    axs[0].axvline(1739.98, color='r', linestyle='-', label=r"Si K$\alpha$1", alpha=0.6, lw=0.5)
    axs[0].axvline(1835.94, color='r', linestyle='-', label=r"Si K$\beta$", alpha=0.6, lw=0.5)
    axs[0].axvline(1839., color='r', linestyle='--', label="Si Kedge", alpha=0.6, lw=0.5)
#    axs[0].axvline(2145.5, color='g', linestyle='--', label="P Kedge", alpha=0.6, lw=0.5)
#    axs[0].axvline(2013.7, color='b', linestyle='-', label=r"P K$\alpha$1", alpha=0.6, lw=0.5)
    axs[0].axvline(2122.9, color='c', linestyle='-', label=r"Au M$\alpha$1", alpha=0.6, lw=0.5)
#    axs[0].axvline(2195.3, color='m', linestyle='-', label=r"Hg M$\alpha$1", alpha=0.6, lw=0.5)
    axs[0].axvline(2206., color='c', linestyle='--', label=r"Au M5", alpha=0.6, lw=0.5)
    axs[0].axvline(2129.5, color='y', linestyle='-', label=r"escape(MnKa1-TeLa1)", alpha=0.6, lw=0.5) # 5898.8(MnKa1) - 3769.3(TeLa1) = 2129.5
    axs[0].axvline(1869.2, color='y', linestyle='-', label=r"escape(MnKa1-TeLb1)", alpha=0.6, lw=0.5) # 5898.8(MnKa1) - 4029.6(TeLb1) = 1869.2


    axs[0].grid(alpha=0.1)
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=5)
    axs[0].set_xlim(emin, emax)

    xval = 0.5 * (binedges[1:] + binedges[:-1]) * 0.5 + 0.5 if x_col == "PI" else 0.5 * (binedges[1:] + binedges[:-1])
    axs[1].errorbar(xval, total_hist, yerr=np.sqrt(total_hist), fmt='.', color='black', label="Total", ms=1, alpha=1.0)
    axs[1].legend()
    axs[1].set_xlabel("PI (eV)")
    axs[1].set_ylabel("Counts/bin")
    if ylin:
        axs[1].set_yscale("linear")
    else:
        axs[1].set_yscale("log")
    axs[1].axvline(1739.98, color='r', linestyle='-', label=r"Si K$\alpha$1", alpha=0.6, lw=0.5)
    axs[1].axvline(1835.94, color='r', linestyle='-', label=r"Si K$\beta$", alpha=0.6, lw=0.5)
    axs[1].axvline(1839., color='r', linestyle='--', label="Si Kedge", alpha=0.6, lw=0.5)
#    axs[1].axvline(2145.5, color='g', linestyle='--', label="P Kedge", alpha=0.6, lw=0.5)
#    axs[1].axvline(2013.7, color='b', linestyle='-', label=r"P K$\alpha$1", alpha=0.6, lw=0.5)
    axs[1].axvline(2122.9, color='c', linestyle='-', label=r"Au M$\alpha$1", alpha=0.6, lw=0.5)
#    axs[1].axvline(2195.3, color='m', linestyle='-', label=r"Hg M$\alpha$1", alpha=0.6, lw=0.5)
    axs[1].axvline(2206., color='c', linestyle='--', label=r"Au M5", alpha=0.6, lw=0.5)
    axs[1].axvline(2129.5, color='y', linestyle='-', label=r"escape(MnKa1-TeLa1)", alpha=0.6, lw=0.5) # 5898.8(MnKa1) - 3769.3(TeLa1) = 2129.5
    axs[1].axvline(1869.2, color='y', linestyle='-', label=r"escape(MnKa1-TeLb1)", alpha=0.6, lw=0.5) # 5898.8(MnKa1) - 4029.6(TeLb1) = 1869.2

    axs[1].grid(alpha=0.1)
    axs[1].set_xlim(emin, emax)

    ofname = f"{outfname}_emin{emin}_emax{emax}_rebin{rebin}_{cutp}_itype{itype_cut}.png"
    fig.tight_layout()  # Adjust layout    
    plt.savefig(ofname)
    if plotflag:
        plt.show()
    print(f"..... {ofname} is created.")

    log_fname = f"{outfname}_emin{emin}_emax{emax}_rebin{rebin}_{cutp}_itype{itype_cut}_log.csv"
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
        >resolve_ana_pixel_compspec.py eve.list
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)    
    parser.add_argument("file_names", type=str, help="List of Path to the FITS file")
    parser.add_argument("--x_col", type=str, help="Column name for the x-axis", default="PI")
    parser.add_argument('--x_hdu', type=int, help='Number of FITS HDU for X', default=1)
    parser.add_argument("--filters", '-f', type=str, help="Comma-separated filter conditions in the format 'COLUMN==VALUE'", default="")
    parser.add_argument('--plot', '-p', action='store_true', default=False, help='Flag to display plot')
    parser.add_argument('--emin', type=float, help='emin', default=1700)
    parser.add_argument('--emax', type=float, help='emin', default=2400)
    parser.add_argument('--rebin', type=int, help='rebin', default=2)
    parser.add_argument('--ylin', action='store_true', default=False, help='Flag to ylinear')
    parser.add_argument('--calout', action='store_true', default=False, help='Flag to cal select')

    args = parser.parse_args()
    file_names = [ _.strip() for _ in open(args.file_names)]
    print(f'file_names = {file_names}')

    print(f'x_hdu = {args.x_hdu}')
    print(f'x_col = {args.x_col}')

    # Define the energy range
    emin, emax = args.emin, args.emax  # Energy range in eV
    pimin, pimax = ev_to_pi(emin), ev_to_pi(emax)
    rebin = args.rebin
    binnum = int((pimax - pimin) / rebin)

    filter_conditions = parse_filter_conditions(args.filters) if args.filters else None
    title = f"{args.file_names} : filtered with {args.filters}"
    outfname = "compspec_" + args.file_names.replace(",","_").replace(".","_p_")
    plot_xhist(file_names, args.x_col, args.x_hdu, outfname,\
        pimin, pimax, emin, emax, rebin, binnum, \
    plotflag=args.plot, debug=True, filters=filter_conditions, ylin=args.ylin, calout = args.calout)

if __name__ == "__main__":
    main()
