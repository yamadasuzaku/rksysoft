#!/usr/bin/env python

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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot spectra from a FITS file.')
    parser.add_argument('filename', help='The name of the FITS file to process.')
    parser.add_argument('--rebin', type=int, default=10, help='Rebin factor')
    parser.add_argument('--emin', type=float, default=0, help='Minimum energy in eV')
    parser.add_argument('--emax', type=float, default=20000, help='Maximum energy in eV')
    parser.add_argument('--debug', '-d', action='store_true', help='do show')     
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
    print(f"Data from {dtime[0]} --> {dtime[-1]}")
    dt = np.diff(sorted_columns[0])
    return [column[:-1] for column in sorted_columns], dt

def plot_pi(pi, itype, pixel, emin, emax, rebin, outfname="mkpi.png", title="test", debug=False):
    pimin, pimax = ev_to_pi(emin), ev_to_pi(emax)
    binnum = int((pimax - pimin) / rebin)
    
    for itype_ in itypename:
        plt.figure(figsize=(11, 7))
        plt.subplots_adjust(right=0.8) # make the right space bigger
        plt.xscale("linear")
        plt.yscale("log")
        plt.ylabel(f"Counts/bin (bin={rebin/2}eV)")
        plt.xlabel("PI (eV)")
        plt.grid(alpha=0.8)
        plt.title(title + " TYPE = " + typename[itype_])

        # Filter data by itype
        typecut = (itype == itype_)
        pi_filtered = pi[typecut]

        # Compute histogram for all pixels of current itype
        hist, binedges = np.histogram(pi_filtered, bins=binnum, range=(pimin, pimax))
        bincenters = 0.5 * (binedges[1:] + binedges[0:-1])
        ene = bincenters * 0.5 + 0.5
        event_number = len(pi_filtered)
        plt.errorbar(ene, hist, yerr=np.sqrt(hist), color="k", fmt='-', label="all" + "("+str(event_number)+ "c)", alpha=0.5)

        # Plot histograms for each pixel
        for pixel_ in np.arange(36):
            pixelcut = (pixel == pixel_) & typecut
            pi_pixel_filtered = pi[pixelcut]
            
            if len(pi_pixel_filtered) == 0:
                print("Warning: data is empty for pixel =", pixel_)
                continue

            hist, binedges = np.histogram(pi_pixel_filtered, bins=binnum, range=(pimin, pimax))
            ene = 0.5 * (binedges[1:] + binedges[:-1]) * 0.5 + 0.5
            color = scalarMap.to_rgba(pixel_)
            event_number = len(pi_pixel_filtered)
            plt.errorbar(ene, hist, yerr=np.sqrt(hist), color=color, fmt='-', label=f"P{pixel_}" + "("+str(event_number)+ "c)")

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=6)
        plt.xlim(emin, emax)
        ofname = f"fig_{typename[itype_]}_{emin}_{emax}_{rebin}_{outfname}"
        plt.savefig(ofname)
        if debug:
            plt.show()
        print(f"..... {ofname} is created.")

def plot_pi_allpixel(pi, itype, pixel, emin, emax, rebin, outfname="mkpi.png", title="test", debug=False):
    pimin, pimax = ev_to_pi(emin), ev_to_pi(emax)
    binnum = int((pimax - pimin) / rebin)
    
    for itype_ in itypename:
        plt.figure(figsize=(11, 7))
        plt.subplots_adjust(right=0.8) # make the right space bigger
        plt.xscale("linear")
        plt.yscale("log")
        plt.ylabel(f"Counts/bin (bin={rebin/2}eV)")
        plt.xlabel("PI (eV)")
        plt.grid(alpha=0.8)
        plt.title(title + " TYPE = " + typename[itype_])
        # Filter data by itype
        typecut = (itype == itype_)
        pi_filtered = pi[typecut]

        # Compute histogram for all pixels of current itype
        hist, binedges = np.histogram(pi_filtered, bins=binnum, range=(pimin, pimax))
        bincenters = 0.5 * (binedges[1:] + binedges[0:-1])
        ene = bincenters * 0.5 + 0.5
        event_number = len(pi_filtered)
        cutid = np.where( (ene > emin) & (ene < emax) )[0]
        ene = ene[cutid]
        hist = hist[cutid]        
        plt.errorbar(ene, hist, yerr=np.sqrt(hist), color="k", fmt='-', label="all" + "("+str(event_number)+ "c)", alpha=0.9)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)
        plt.xlim(emin, emax)
        plt.ylim(np.amin(hist), np.amax(hist))        
        ofname = f"fig_allpixel_{typename[itype_]}_{emin}_{emax}_{rebin}_{outfname}"
        plt.savefig(ofname)
        if debug:
            plt.show()
        print(f"..... {ofname} is created.")

def main():
    args = parse_arguments()
    print(f"Parsing arguments: {args}")
    data = open_fits_data(args.filename)
    processed_data, dt = process_data(data)
    time, itype, pi, rise_time, deriv_max, pixel = processed_data  # data unpack
    plot_pi(pi, itype, pixel, emin=args.emin, emax=args.emax, rebin=args.rebin, 
            outfname=f"ql_plotspec_{args.filename.replace('.evt', '').replace('.gz', '')}.png", 
            title=f"Spectra of {args.filename}", debug=args.debug)
    plot_pi_allpixel(pi, itype, pixel, emin=args.emin, emax=args.emax, rebin=args.rebin, 
            outfname=f"ql_plotspec_{args.filename.replace('.evt', '').replace('.gz', '')}.png", 
            title=f"Spectra of {args.filename}", debug=args.debug)


if __name__ == "__main__":
    main()
