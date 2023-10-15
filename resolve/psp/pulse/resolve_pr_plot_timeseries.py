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

    itypename = [ 0,    1,   2,   3,   4,   5,   6,  7]
    icolor =    ["r", "b", "c", "g", "m", "k", "y", "0.8"]
    typename = ["Hp", "Mp", "Ms", "Lp", "Ls", "BL", "EL", "-"]
    time, itype, pixel, pha = data["TIME"], data["ITYPE"], data["PIXEL"], data["PHA"]
    pulserec_mode, pulserec = data["PULSEREC_MODE"], data["PULSEREC"]

    cutid = np.where( (pixel == target_pixel) & (itype >= target_itype) &  (pulserec_mode == target_pulserec_mode))[0]

    pulserec = pulserec[cutid]
    time = time[cutid]
    itype = itype[cutid]
    rtime = time - time[0] 

    date_obs = hdu.header['DATE-OBS']
    date_end = hdu.header['DATE-END']

    adclength = 1040
    xadc = np.arange(0, adclength, 1)
    xadc_time = xadc * 80e-6 # sec

    F = plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1)

    Nevent = len(time)

    plt.title(fname + " pixel = " + str(target_pixel) + " ITYPE = " + str(typename[target_itype]) + " pulserec_mode = " + str(target_pulserec_mode))
    plt.figtext(0.1,0.94, date_obs + " to " + date_end + "  # Total Number of events = " + str(Nevent))

    for i, (onepulse, onertime, oneitype) in enumerate(zip(pulserec,rtime,itype)):
        plt.errorbar(xadc_time + onertime, onepulse, fmt='-', color = icolor[oneitype], label="PIXEL " + str(pixel), alpha=0.6)

    plt.ylabel('ADC ' + '(pulse)')
    plt.xlabel('Time (s) from ' + f'{time[0]}')
    plt.grid(True)
    plt.savefig(os.path.join(output_directory, fnametag + "_all.png"))
    if plot_flag: plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process FITS file and generate plots.")
    parser.add_argument("fname", type=str, help="File name to be processed.")
    parser.add_argument("--target_pixel", type=int, default=22, choices=range(36), help="Target pixel value.")
    parser.add_argument("--target_itype", type=int, default=0, choices=range(8), help="Target itype value.")
    parser.add_argument("--target_pulserec_mode", type=int, default=0, choices=[0, 1], help="Target pulserec_mode value.")
    parser.add_argument("--plot_flag", action="store_true", help="If set, will display plots.")
    args = parser.parse_args()
    plot_fits(args.fname, args.target_pixel, args.target_itype, args.target_pulserec_mode, args.plot_flag)
