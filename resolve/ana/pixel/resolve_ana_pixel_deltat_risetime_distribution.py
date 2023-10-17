#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import datetime
import astropy.io.fits

# MJD reference day 01 Jan 2019 00:00:00
MJD_REFERENCE_DAY = 58484
reference_time = Time(MJD_REFERENCE_DAY, format='mjd')

itypename = [0, 1, 2, 3, 4]
typename = ["Hp", "Mp", "Ms", "Lp", "Ls"]
icol = ["r","b","c", "m", "y"]
ishape = [".","s","D", "*", "x"]

def plot_dt_risetime(target_fname, dt, pha, rise_time, itype, xscale="linear", yscale="linear", bin_count=100000, range_min=1e-6, range_max=100, risetime_min = 0, risetime_max = 130, outfname="output.png", xlabel="TIME (sec) from the next event", PLOT_FLAG=True, TRIGTIME_FLAG=True):
    phacut = 1000
    mindt = 5e-6
    dt = dt + mindt 
    F = plt.figure(figsize=(10,8.))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)    
    ax = plt.subplot(2,1,1)
    plt.ylim(risetime_min,risetime_max)
    plt.xlim(range_min,range_max)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.ylabel("RISE_TIME")
    plt.grid(alpha=0.8)
    plt.figtext(0.1, 0.95, "created from " + target_fname + " using all pixels")
    for i, one_typename in enumerate(typename):
        cut = (pha>phacut) & (itype == i)
        plt.errorbar(dt[cut], rise_time[cut], alpha = 0.9, ms =3, \
            fmt=ishape[i], color = icol[i], label=one_typename + " PHA>" + str(phacut) + " (N=" + str(len(dt[cut]))+ ")")
        plt.legend(bbox_to_anchor=(0., 1.01, 1., 0.01), loc='lower left',ncol=3, borderaxespad=0.,fontsize=8)

    ax = plt.subplot(2,1,2)
    plt.ylim(risetime_min,risetime_max)
    plt.xlim(range_min,range_max)
    plt.xscale(xscale)
    plt.yscale(yscale)
    if TRIGTIME_FLAG:
        plt.xlabel(r"TRIGTIME + 5 $\mu$s as an offset (sec) from the next event")
    else:
        plt.xlabel(xlabel)
    plt.ylabel("RISE_TIME")
    plt.grid(alpha=0.8)
    for i, one_typename in enumerate(typename):
        cut = (pha<=phacut) & (itype == i)
        plt.errorbar(dt[cut], rise_time[cut], alpha = 0.9, ms =3,\
            fmt=ishape[i], color = icol[i], label=one_typename + " PHA<=" + str(phacut) + "(N=" + str(len(dt[cut]))+ ")")
    plt.legend(bbox_to_anchor=(0., 1.01, 1., 0.01), loc='lower left',ncol=3, borderaxespad=0.,fontsize=8)
    
    plt.savefig(outfname)
    print("..... "  + outfname + " is created.")

    if PLOT_FLAG: plt.show()

def plot_time_data(time, pha, range_min, range_max, outfname="output.png", PLOT_FLAG=True, TRIGTIME_FLAG=True):
    dtime = [reference_time.datetime + datetime.timedelta(seconds=float(date_sec)) for date_sec in time]
    fig = plt.figure(figsize=(12, 6))
    plt.grid(alpha=0.8)
    plt.figtext(0.1, 0.95, outfname)
    plt.errorbar(dtime, pha, fmt=".", ms=1)
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
        time, pha, itype, rise_time = data["TRIGTIME"], data["PHA"], data["ITYPE"], data["RISE_TIME"]
    else:
        time, pha, itype, rise_time = data["TIME"], data["PHA"], data["ITYPE"], data["RISE_TIME"]

    sortid = np.argsort(time)
    time = time[sortid]
    pha = pha[sortid]
    itype = itype[sortid]
    rise_time = rise_time[sortid]    

    cut_type = np.where(itype < 5)[0] 

    time = time[cut_type]
    pha = pha[cut_type]
    itype = itype[cut_type]
    rise_time = rise_time[cut_type]    

    print(f"Number of events are {len(time)} --> ITYPE < 5 --->  {len(cut_type)} ")

    dt = np.diff(time) 

    time = time[:-1]
    pha = pha[:-1]
    itype = itype[:-1]
    rise_time = rise_time[:-1]    

    return dt, pha, time, itype, rise_time

def main(target_fname, xscale, yscale, PLOT_FLAG, TRIGTIME_FLAG):
    dt, pha, time, itype, rise_time = process_data(target_fname, TRIGTIME_FLAG)

    fname_tag = target_fname.replace(".evt","").replace(".gz","")
    trigtime_suffix = "_TRIGTIME" if TRIGTIME_FLAG else "_TIME"
    
    outfname_dt = f"deltaT_risetime_{xscale}_{yscale}_" + fname_tag + trigtime_suffix + ".png"
    plot_dt_risetime(target_fname, dt, pha, rise_time, itype, xscale=xscale, yscale=yscale, outfname=outfname_dt, PLOT_FLAG=PLOT_FLAG, TRIGTIME_FLAG=TRIGTIME_FLAG)
    
    outfname_time_data = "lightcurve_all_PIXEL_all_types_" + fname_tag + trigtime_suffix + ".png"
    plot_time_data(time, pha, 0, 70000, outfname=outfname_time_data, PLOT_FLAG=PLOT_FLAG, TRIGTIME_FLAG=TRIGTIME_FLAG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some pixel data')
    parser.add_argument('--fname', type=str, required=True, help='Target fits file name')
    parser.add_argument('--xscale', type=str, choices=['log', 'linear'], default='linear', help='X-axis scale: log or linear')
    parser.add_argument('--yscale', type=str, choices=['log', 'linear'], default='linear', help='Y-axis scale: log or linear')
    parser.add_argument('--plot', action='store_true', default=False, help='Flag to display plot')
    parser.add_argument('--trigtime', action='store_true', default=False, help='Flag to use TRIGTIME instead of TIME')
    
    args = parser.parse_args()
    main(args.fname, args.xscale, args.yscale, args.plot, args.trigtime)
