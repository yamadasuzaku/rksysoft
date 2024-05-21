#!/usr/bin/env python

import argparse
from astropy.io import fits
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Process FITS file and display exposure and GTI information.')
    parser.add_argument('filename', type=str, help='Path to the FITS file')
    args = parser.parse_args()

    # Open the FITS file
    hdu = fits.open(args.filename)
    
    # Get the exposure from the header
    exposure = hdu[2].header["EXPOSURE"]
    print(f"Exposure: {exposure}")

    # Calculate the sum of the difference between STOP and START
    gti_sum = np.sum(hdu[2].data["STOP"] - hdu[2].data["START"])
    print(f"Sum of STOP - START: {gti_sum}")

    # Get the length of the STOP array
    gti_length = len(hdu[2].data["STOP"])
    print(f"Length of STOP array: {gti_length}")

    # Close the FITS file
    hdu.close()

if __name__ == "__main__":
    main()
