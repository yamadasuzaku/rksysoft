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
PLOT_FLAG = True

def plot_dt(dt, pha, target_fname, xscale="linear", yscale="log", bin_count=100000, range_min=1e-7, range_max=0.03, outfname="output.png", xlabel="PHA"):
    phacut = 1000
    F = plt.figure(figsize=(10,7.))
    ax = plt.subplot(2,1,1)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.ylabel("Number of events")
    plt.grid(alpha=0.8)
    plt.figtext(0.1, 0.95, "created from " + target_fname + " using all pixels")
    plt.hist(dt[pha>phacut], bins=bin_count, range=(range_min, range_max), \
                          histtype='step', label="PHA > " + str(phacut) + " (event num = " + str(len(dt[pha>phacut])))
    plt.legend(bbox_to_anchor=(0., 1.01, 1., 0.01), loc='lower left',ncol=10, borderaxespad=0.,fontsize=8)


    ax = plt.subplot(2,1,2)

    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(xlabel)
    plt.ylabel("Number of events")
    plt.grid(alpha=0.8)
    plt.hist(dt[pha<=phacut], bins=bin_count, range=(range_min, range_max), \
                           histtype='step', label="PHA <= " + str(phacut) + " (event num = " + str(len(dt[pha<=phacut])))
    plt.legend(bbox_to_anchor=(0., 1.01, 1., 0.01), loc='lower left',ncol=10, borderaxespad=0.,fontsize=8)
    
    plt.savefig(outfname)
    if PLOT_FLAG: plt.show()

def plot_time_data(time, pha, range_min, range_max, outfname="output.png"):
    dtime = [reference_time.datetime + datetime.timedelta(seconds=float(date_sec)) for date_sec in time]
    fig = plt.figure(figsize=(12, 6))
    plt.grid(alpha=0.8)
    plt.figtext(0.1, 0.95, outfname)
    plt.errorbar(dtime, pha, fmt=".", ms=1)
    plt.xlabel("TIME from " + str(dtime[0]))
    plt.ylabel("PHA")
    plt.savefig(outfname)
    if PLOT_FLAG: plt.show()

def process_data(target_fname):
    data = astropy.io.fits.open(target_fname)[1].data
    time, pha = data["TIME"], data["PHA"]

    sortid = np.argsort(time)
    time = time[sortid]
    pha = pha[sortid]

    dt = np.diff(time[sortid])

    time = time[:-1]
    pha = pha[:-1]

    return dt, pha, time

def main(target_fname):
    dt, pha, time = process_data(target_fname)
    
    plot_dt(dt, pha, target_fname)
    plot_time_data(time, pha, 0, 70000, outfname="lightcurve_all_PIXEL_all_types.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some pixel data')
    parser.add_argument('--fname', type=str, required=True, help='Target fits file name')
    args = parser.parse_args()
    main(args.fname)
