#!/usr/bin/env python

import os
import astropy.io.fits
import matplotlib.pylab as plt 
import argparse
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm

def plot_fits(fname, target_pixel, target_itype, target_pulserec_mode, plot_flag):
    fnametag = fname.replace(".evt", "").replace(".gz", "")

    # Create directory
    output_directory = fnametag + "_plots"
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    hdu = astropy.io.fits.open(fname)[1]
    data = hdu.data

    itypename = [0, 1, 2, 3, 4, 5, 6, 7]
    typename = ["Hp", "Mp", "Ms", "Lp", "Ls", "BL", "EL", "-"]
    time, itype, pixel, pha = data["TIME"], data["ITYPE"], data["PIXEL"], data["PHA"]
    pulserec_mode, pulserec = data["PULSEREC_MODE"], data["PULSEREC"]

    date_obs = hdu.header['DATE-OBS']
    date_end = hdu.header['DATE-END']

    adclength = 1040
    xadc = np.arange(0, adclength, 1)

    F = plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1)
    cutid = np.where( (pixel == target_pixel) & (itype == target_itype) &  (pulserec_mode == target_pulserec_mode))[0]

    Nevent = len(cutid)
    jet = plt.get_cmap('jet')
    cNorm  = colors.Normalize(vmin=0, vmax=Nevent)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

    plt.title(fname + " pixel = " + str(target_pixel) + " ITYPE = " + str(typename[target_itype]) + " pulserec_mode = " + str(target_pulserec_mode))
    plt.figtext(0.1,0.94, date_obs + " to " + date_end + "  # Total Number of events = " + str(Nevent))

    for i, onepulse in enumerate(pulserec[cutid]):
        c = scalarMap.to_rgba(i)
        plt.errorbar(xadc, onepulse, fmt='-', color = c, label="PIXEL " + str(pixel), alpha=0.6)

    plt.ylabel('ADC ' + '(pulse)')
    plt.xlabel('Time in unit of 1 tick (80 us nominal)')
    plt.grid(True)
    plt.savefig(os.path.join(output_directory, fnametag + "_all.png"))
    if plot_flag: plt.show()

    for i, onepulse in enumerate(pulserec[cutid]):
        c = scalarMap.to_rgba(i)
        F = plt.figure(figsize=(8,6))
        ax = plt.subplot(1,1,1)
        plt.title(fname + " pixel = " + str(target_pixel) + " ITYPE = " + str(typename[target_itype]) + " pulserec_mode = " + str(target_pulserec_mode))
        plt.figtext(0.1,0.94, date_obs + " to " + date_end + "  # Total Number of events = " + str(Nevent) + "  No. " + str(i))
        plt.errorbar(xadc, onepulse, fmt='-', color = "r", label="PIXEL " + str(pixel))
        plt.ylabel('ADC ' + '(pulse)')
        plt.xlabel('Time in unit of 1 tick (80 us nominal)')
        plt.grid(True)
        plt.savefig(os.path.join(output_directory, "each_" + fnametag + "_" + str("%04d" % i) + ".png"))
        if plot_flag : plt.show()
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process FITS file and generate plots.")
    parser.add_argument("fname", type=str, help="File name to be processed.")
    parser.add_argument("--target_pixel", type=int, default=22, choices=range(36), help="Target pixel value.")
    parser.add_argument("--target_itype", type=int, default=3, choices=range(8), help="Target itype value.")
    parser.add_argument("--target_pulserec_mode", type=int, default=0, choices=[0, 1], help="Target pulserec_mode value.")
    parser.add_argument("--plot_flag", action="store_true", help="If set, will display plots.")
    args = parser.parse_args()
    plot_fits(args.fname, args.target_pixel, args.target_itype, args.target_pulserec_mode, args.plot_flag)
