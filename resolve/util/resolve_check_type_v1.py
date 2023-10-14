#!/usr/bin/env python 

import astropy.io.fits
import matplotlib.pylab as plt 
from collections import Counter
import argparse

def main(fname):
    data = astropy.io.fits.open(fname)[1].data
    time, type, itype, pixel = data["TIME"], data["TYPE"], data["ITYPE"], data["PIXEL"]

    counter_type = Counter(type)
    print("[check type] all")
    for _ in counter_type.most_common():
        print(_)

    pairs = list(zip(type, pixel))
    counter_type_pixel = Counter(pairs)

    print("[check type] each pixel")
    for _ in counter_type_pixel.most_common():
        print(_)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process FITS file.")
    parser.add_argument("fname", type=str, help="File name to be processed.")
    args = parser.parse_args()

    main(args.fname)
