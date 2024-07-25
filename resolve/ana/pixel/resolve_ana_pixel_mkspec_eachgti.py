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

# Plotting Configuration
plt.rcParams['font.family'] = 'serif'
usercmap = plt.get_cmap('jet')
cNorm = Normalize(vmin=0, vmax=35)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=usercmap)

# Type Information
g_itypename = [0, 1, 2, 3, 4]
g_typename = ["Hp", "Mp", "Ms", "Lp", "Ls"]
g_icol = ["r", "b", "c", "m", "y"]
g_ishape = [".", "s", "D", "*", "x"]

# Maximum number of colors to be used in plots
CMAX = 20
colors = plt.cm.tab20(np.linspace(0, 1, CMAX))

def ev_to_pi(ev):
    """Convert energy in eV to PI units."""
    return (ev - 0.5) * 2

def pi_to_ev(pi):
    """Convert PI units to energy in eV."""
    return pi * 0.5 + 0.5

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot spectra from a FITS file.')
    parser.add_argument('filename', help='The name of the FITS file to process.')
    parser.add_argument('--rebin', '-r', type=int, default=10, help='Rebin factor in PI space (dE = 0.5 dPI)')
    parser.add_argument('--mgti', '-m', type=int, default=1, help='Rebin GTI')
    parser.add_argument('--emin', '-i', type=float, default=0, help='Minimum energy in eV')
    parser.add_argument('--emax', '-x', type=float, default=20000, help='Maximum energy in eV')
    parser.add_argument('--ymin', '-ymin', type=float, help='Minimum y-axis value')
    parser.add_argument('--ymax', '-ymax', type=float, help='Maximum y-axis value')
    parser.add_argument('--itypenames', '-y', type=str, help='Comma-separated list of itype', default='0,1,2,3,4')
    parser.add_argument('--plotpixels', '-p', type=str, help='Comma-separated list of pixels to plot', default=','.join(map(str, range(36))))
    parser.add_argument('--specoffset', '-t', action='store_true', help='Offset for the spectrum')    
    parser.add_argument('--specoffsetval', '-v', type=float, default=0.001, help='Offset value for the spectrum')    

    return parser.parse_args()

def open_fits_data(fname):
    """Open the FITS file and return the data."""
    try:
        print(f"Opening FITS file: {fname}")
        return fits.open(fname)[1].data
    except FileNotFoundError:
        print("ERROR: File not found", fname)
        sys.exit()

def process_data(data, TRIGTIME_FLAG=False, AC_FLAG=False):
    """Process the data from the FITS file."""
    time = data["TRIGTIME"] if TRIGTIME_FLAG else data["TIME"]
    itype = data["AC_ITYPE"] if AC_FLAG else data["ITYPE"]
    if len(time) == 0:
        print("ERROR: data is empty", time)
        sys.exit()
    additional_columns = [data[col] for col in ["PI", "RISE_TIME", "DERIV_MAX", "PIXEL"]]
    sortid = np.argsort(time)
    sorted_columns = [col[sortid] for col in [time, itype] + additional_columns]
    dtime = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in sorted_columns[0]])
    print(f"Data from {dtime[0]} to {dtime[-1]}")
    dt = np.diff(sorted_columns[0])
    return [column[:-1] for column in sorted_columns], dt

def plot_pi(pi, itype, pixel, time, gtistart, gtistop, emin=0, emax=20000, ymin=None, ymax=None, rebin=10,
            outfname="mkpi.png", title="test", brcor=False, itypenames=[0], mgti=3,
            plotpixels=[0], specoffset=True, specoffsetval=0.001):
    """Plot the PI spectrum."""
    pimin, pimax = ev_to_pi(emin), ev_to_pi(emax)
    ene_rebin = 0.5 * rebin  # dE = 0.5 * dPI
    binnum = int((pimax - pimin) / rebin)
    
    for itype_ in itypenames:
        plt.figure(figsize=(11, 9))
        plt.subplots_adjust(right=0.8)  # make the right space bigger
        plt.xscale("linear")
        plt.ylabel(f"Counts/{ene_rebin:.1f}eV/s")
        plt.xlabel("Energy (eV)")
        plt.grid(alpha=0.4)
        plt.title(title + " TYPE = " + g_typename[itype_])
        plt.figtext(0.55, 0.01, "pixel = " + ",".join(map(str, plotpixels)), fontsize=6, alpha=0.4)
        cumulative_hist = np.zeros(binnum)
        cumulative_event_number = 0
        mgti_dt = 0
        nspec = 0
        handles = []

        for i, (t1, t2) in enumerate(zip(gtistart, gtistop)):
            dt = t2 - t1
            # Filter data by itype and specified pixels
            cutid = np.where((itype == itype_) & ((time > t1) & (time <= t2)) & ((pi > pimin) & (pi <= pimax)) & np.isin(pixel, plotpixels))[0]
            pi_filtered = pi[cutid]

            hist, binedges = np.histogram(pi_filtered, bins=binnum, range=(pimin, pimax))
            cumulative_hist += hist
            cumulative_event_number += len(pi_filtered)
            mgti_dt += dt 

            if (i + 1) % mgti == 0:
                nspec += 1
                color = colors[nspec % len(colors)]
                bincenters = 0.5 * (binedges[1:] + binedges[0:-1])
                ene = bincenters * 0.5 + 0.5
                if specoffset:
                    offset = nspec * specoffsetval
                else:
                    offset = 0

                handle = plt.errorbar(ene, cumulative_hist/(ene_rebin * mgti_dt) + offset, \
                                      yerr=np.sqrt(cumulative_hist)/(ene_rebin * mgti_dt), color=color, fmt='-', 
                                      label=f"GTI-{i + 1 - mgti + 1} to {i + 1} ({cumulative_event_number} counts / {mgti_dt:.1f} seconds)", alpha=0.8)
                handles.append(handle)
                # Set error bar transparency
                for bar in handle[2]:
                    bar.set_alpha(0.2)            

                print(f"itype={itype_}, gti_i={i}, de={ene[1]-ene[0]}eV, {ene[0]} to {ene[-1]} eV ({cumulative_event_number} counts / {mgti_dt:.1f} seconds)")
                # initialize cumulative values
                cumulative_hist = np.zeros(binnum)
                cumulative_event_number = 0
                mgti_dt = 0

        # Handle the case where the last cumulative histogram may not have been plotted
        if cumulative_event_number > 0:
            nspec += 1
            if specoffset:
                offset = nspec * specoffsetval
            else:
                offset = 0

            color = colors[nspec % len(colors)]
            bincenters = 0.5 * (binedges[1:] + binedges[0:-1])
            ene = bincenters * 0.5 + 0.5
            handle = plt.errorbar(ene, cumulative_hist/(ene_rebin * mgti_dt) + offset, yerr=np.sqrt(cumulative_hist)/(ene_rebin * mgti_dt), color=color, fmt='-', 
                                  label=f"GTI-{len(gtistart) - cumulative_event_number // binnum + 1} to {len(gtistart)} ({cumulative_event_number} counts / {mgti_dt:.1f} seconds)", alpha=0.8)
            handles.append(handle)
            print(f"itype={itype_}, gti_i={i}, de={ene[1]-ene[0]}eV, {ene[0]} to {ene[-1]} eV ({cumulative_event_number} counts / {mgti_dt:.1f} seconds)")
            # Set error bar transparency
            for bar in handle[2]:
                bar.set_alpha(0.2)            

        plt.legend(handles=handles[::-1], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=8)
        plt.xlim(emin, emax)

        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)
        outfname_with_pixels = f"{outfname.split('.')[0]}_{'-'.join(map(str, plotpixels))}.png"
        ofname = f"fig_{g_typename[itype_]}_{outfname_with_pixels}"
        plt.tight_layout()
        plt.savefig(ofname)
        plt.show()
        print(f"..... {ofname} is created.")

def main():
    """Main function to parse arguments and call the plotting function."""
    args = parse_arguments()
    print(f"Parsing arguments: {args}")
    data = open_fits_data(args.filename)
    processed_data, dt = process_data(data)
    time, itype, pi, rise_time, deriv_max, pixel = processed_data  # Unpack data
    gtistart = fits.open(args.filename)[2].data["START"]
    gtistop = fits.open(args.filename)[2].data["STOP"]
    itypenames = list(map(int, args.itypenames.split(',')))
    plotpixels = list(map(int, args.plotpixels.split(',')))

    outfname = f"mkspec_eachgti_{args.filename.replace('.evt', '').replace('.gz', '')}.png"

    plot_pi(pi, itype, pixel, time, gtistart, gtistop, emin=args.emin, emax=args.emax, ymin=args.ymin, ymax=args.ymax, rebin=args.rebin, 
            outfname=outfname, 
            title=f"Spectra of {args.filename}",
            itypenames=itypenames, plotpixels=plotpixels, mgti=args.mgti, 
            specoffset=args.specoffset, specoffsetval=args.specoffsetval)

if __name__ == "__main__":
    main()
