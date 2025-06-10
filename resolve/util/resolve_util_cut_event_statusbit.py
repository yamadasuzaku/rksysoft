#!/usr/bin/env python

import argparse
import numpy as np
from astropy.io import fits
import os

def parse_status_arg(status_str):
    """
    Parse the --status argument string into a NumPy array of 16 integers.
    Each value must be -1, 0, or 1, representing wildcard, False, or True respectively.
    """
    values = status_str.split(',')
    if len(values) != 16:
        raise ValueError("Status must contain exactly 16 comma-separated values (-1, 0, or 1).")
    for v in values:
        if v not in {'-1', '0', '1'}:
            raise ValueError("Each status value must be -1, 0, or 1.")
    return np.array([int(v) for v in values], dtype=int)

def main():
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description="Filter FITS rows by STATUS match with -1 as wildcard.")
    parser.add_argument("input_file", type=str, help="Path to input FITS file")
    parser.add_argument("output_file", type=str, help="Path to output FITS file")
    parser.add_argument("--status", type=str, required=True,
                        help="Comma-separated 16-element list with values -1 (ignore), 0 (must be False), or 1 (must be True)")
    
    args = parser.parse_args()
    status_pattern = parse_status_arg(args.status)

    # Open the FITS file and access the data from the second HDU (index 1)
    with fits.open(args.input_file) as hdul:
        data = hdul[1].data
        status_array = data["STATUS"]  # STATUS is a (N, 16) boolean array

        # Initialize a boolean mask with all True (i.e., all rows initially pass)
        mask = np.ones(len(status_array), dtype=bool)

        # Apply filtering conditions for each bit in STATUS
        for i, cond in enumerate(status_pattern):
            if cond == 1:
                # Keep only rows where the i-th bit is True
                mask &= status_array[:, i]
            elif cond == 0:
                # Keep only rows where the i-th bit is False
                mask &= ~status_array[:, i]
            # If cond == -1: do nothing (bit is ignored)

        # Show statistics of filtering
        print(f"Total rows: {len(status_array)}")
        print(f"Matched rows: {np.sum(mask)}")

        # Extract filtered rows
        filtered_data = data[mask]

        # Create a new FITS HDU with the filtered data and copy the original header
        new_hdu = fits.BinTableHDU(data=filtered_data, header=hdul[1].header)
        hdul_new = fits.HDUList([hdul[0], new_hdu])

        # Write to output file (overwrite if it exists)
        hdul_new.writeto(args.output_file, overwrite=True)
        print(f"Filtered FITS saved to: {args.output_file}")

if __name__ == "__main__":
    main()
