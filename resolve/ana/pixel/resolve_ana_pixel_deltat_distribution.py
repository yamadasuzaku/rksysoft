#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'

from astropy.time import Time
import datetime
import astropy.io.fits

# MJD reference day 01 Jan 2019 00:00:00
MJD_REFERENCE_DAY = 58484
reference_time = Time(MJD_REFERENCE_DAY, format='mjd')

# Type Information
itypename = [0, 1, 2, 3, 4]
typename = ["Hp", "Mp", "Ms", "Lp", "Ls"]
icol = ["r", "b", "c", "m", "y"]
ishape = [".", "s", "D", "*", "x"]

def plot_dt(dt, pha, target_fname, xscale="linear", yscale="linear", bin_count=100000, range_min=1e-7, range_max=0.03, outfname="output.png", xlabel="TIME (sec) from the next event", PLOT_FLAG=True, TRIGTIME_FLAG=True):
    
    phacut = 1000
    F, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,7.), sharex=True)  # Create two subplots sharing the x-axis
    
    ax1.set_xscale(xscale)
    ax1.set_yscale(yscale)
    ax1.set_ylabel("Number of events")
    ax1.grid(alpha=0.8)
    ax1.set_title("created from " + target_fname + " using all pixels")
    ax1.hist(dt[pha>phacut], bins=bin_count, range=(range_min, range_max), \
                  histtype='step', label="PHA > " + str(phacut) + " (event num = " + str(len(dt[pha>phacut]))+ ")")
    ax1.legend()
#    ax1.legend(bbox_to_anchor=(0., 1.01, 1., 0.01), loc='lower left', ncol=10, borderaxespad=0., fontsize=8)
    
    ax2.set_xscale(xscale)
    ax2.set_yscale(yscale)
    if TRIGTIME_FLAG:
        ax2.set_xlabel("TRIGTIME (sec) from the next event")
    else:
        ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Number of events")
    ax2.grid(alpha=0.8)
    ax2.hist(dt[pha<=phacut], bins=bin_count, range=(range_min, range_max), \
                     histtype='step', label="PHA <= " + str(phacut) + " (event num = " + str(len(dt[pha<=phacut]))+ ")")
    ax2.legend()
#    ax2.legend(bbox_to_anchor=(0., 1.01, 1., 0.01), loc='lower left', ncol=10, borderaxespad=0., fontsize=8)
    
    plt.tight_layout()
    plt.savefig(outfname)
    print("..... "  + outfname + " is created.")
    
    if PLOT_FLAG: plt.show()

def plot_time_data(time, pha, range_min, range_max, outfname="output.png", PLOT_FLAG=True, TRIGTIME_FLAG=True, color="r"):
    dtime = [reference_time.datetime + datetime.timedelta(seconds=float(date_sec)) for date_sec in time]
    fig = plt.figure(figsize=(12, 6))
    plt.grid(alpha=0.8)
    plt.figtext(0.1, 0.95, outfname)
    plt.errorbar(dtime, pha, fmt=".", ms=1, color=color)
    if TRIGTIME_FLAG:
        plt.xlabel("TRIGTIME from " + str(dtime[0]))
    else:
        plt.xlabel("TIME from " + str(dtime[0]))
    plt.ylabel("PHA")
    plt.savefig(outfname)
    print("..... "  + outfname + " is created.")
    if PLOT_FLAG: plt.show()

def process_data(target_fname, TRIGTIME_FLAG):
    data = astropy.io.fits.open(target_fname)[1].data
    if TRIGTIME_FLAG:
        time, pha, itype = data["TRIGTIME"], data["PHA"], data["ITYPE"]
    else:
        time, pha, itype = data["TIME"], data["PHA"], data["ITYPE"]

    sortid = np.argsort(time)
    time = time[sortid]
    pha = pha[sortid]
    itype = itype[sortid]

    cut_type = np.where(itype < 5)[0] 

    time = time[cut_type]
    pha = pha[cut_type]
    itype = itype[cut_type]

    print(f"Number of events are {len(time)} --> ITYPE < 5 --->  {len(cut_type)} ")

    dt = np.diff(time) 

    time = time[:-1]
    pha = pha[:-1]
    itype = itype[:-1]

    return dt, pha, time, itype

def main(target_fname, xscale, yscale, PLOT_FLAG, TRIGTIME_FLAG):
    dt, pha, time, itype = process_data(target_fname, TRIGTIME_FLAG)

    fname_tag = target_fname.replace(".evt","").replace(".gz","")
    trigtime_suffix = "_TRIGTIME" if TRIGTIME_FLAG else "_TIME"
    
    outfname_dt = f"deltaT_histogram_{xscale}_{yscale}_" + fname_tag + trigtime_suffix + ".png"
    plot_dt(dt, pha, target_fname, xscale=xscale, yscale=yscale, outfname=outfname_dt, PLOT_FLAG=PLOT_FLAG, TRIGTIME_FLAG=TRIGTIME_FLAG)

    outfname_dt = f"deltaT_histogram_linear_{yscale}_" + fname_tag + trigtime_suffix + "_nearzero.png"
    plot_dt(dt, pha, target_fname, xscale="linear", yscale=yscale, outfname=outfname_dt, PLOT_FLAG=PLOT_FLAG, TRIGTIME_FLAG=TRIGTIME_FLAG, 
                range_min=0, range_max=0.01)
    
    outfname_time_data = "lightcurve_pha_all_PIXEL_all_types_" + fname_tag + trigtime_suffix + ".png"
    plot_time_data(time, pha, -10, 65536 + 200, outfname=outfname_time_data, PLOT_FLAG=PLOT_FLAG, TRIGTIME_FLAG=TRIGTIME_FLAG)

    for it, tname, ic in zip(itypename, typename, icol):
        idcut = np.where(itype==it)[0]
        print(f"{it} {tname} {ic} : {len(idcut)} events")
        outfname_time_data = "lightcurve_pha_all_PIXEL_" + tname + "_" + fname_tag + trigtime_suffix + ".png"
        plot_time_data(time[idcut], pha[idcut], -10, 65536 + 200, color=ic, \
            outfname=outfname_time_data, PLOT_FLAG=PLOT_FLAG, TRIGTIME_FLAG=TRIGTIME_FLAG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some pixel data')
    parser.add_argument('fname', help='The name of the FITS file to process.')    
#    parser.add_argument('--fname', type=str, required=True, help='Target fits file name')
    parser.add_argument('--xscale', type=str, choices=['log', 'linear'], default='linear', help='X-axis scale: log or linear')
    parser.add_argument('--yscale', type=str, choices=['log', 'linear'], default='linear', help='Y-axis scale: log or linear')
    parser.add_argument('--plot', action='store_true', default=False, help='Flag to display plot')
    parser.add_argument('--trigtime', action='store_true', default=False, help='Flag to use TRIGTIME instead of TIME')
    
    args = parser.parse_args()
    main(args.fname, args.xscale, args.yscale, args.plot, args.trigtime)
