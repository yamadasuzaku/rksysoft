#!/usr/bin/env python

import astropy.io.fits
import matplotlib.pylab as plt 
import argparse

def generate_plot(xdata, ydata, xlabel, ylabel, title, savename):
    plt.yticks(itypename, typename)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(xdata, ydata, ".")
    plt.savefig(savename)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process FITS file and generate plots.")
    parser.add_argument("fname", type=str, help="File name to be processed.")
    args = parser.parse_args()
    fname = args.fname

    fnametag = fname.replace(".evt", "").replace(".gz", "")
    
    data = astropy.io.fits.open(fname)[1].data

    itypename = [0, 1, 2, 3, 4, 5, 6, 7]
    typename = ["Hp", "Mp", "Ms", "Lp", "Ls", "BL", "EL", "-"]
    time, itype, pixel, pha = data["TIME"], data["ITYPE"], data["PIXEL"], data["PHA"]

    generate_plot(time - time[0], itype, "TIME (s)", "TYPE", fname, fnametag + "_time_itype.png")
    generate_plot(pixel, itype, "PIXEL", "TYPE", fname, fnametag + "_pixel_itype.png")
    generate_plot(pha, itype, "PHA", "TYPE", fname, fnametag + "_pha_itype.png")
