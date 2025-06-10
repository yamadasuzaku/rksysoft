#!/usr/bin/env python

import argparse
import numpy as np
from astropy.io import fits
import os

def parse_status_arg(status_str):
    """
    Parse the --status argument string into a NumPy array of 16 status codes.
    Each value must be 'T' (must be True), 'F' (must be False), or 'X' (don't care).
    """
    values = status_str.split(',')
    if len(values) != 16:
        raise ValueError("Status must contain exactly 16 comma-separated values (T, F, or X).")
    for v in values:
        if v not in {'T', 'F', 'X'}:
            raise ValueError("Each status value must be one of: T (True), F (False), X (ignore).")
    return values  # list of str: 'T', 'F', 'X'

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Filter FITS rows by STATUS pattern using T (True), F (False), X (don't care).")
    parser.add_argument("input_file", type=str, help="Path to input FITS file")
    parser.add_argument("output_file", type=str, help="Path to output FITS file")
    parser.add_argument("--status", type=str, required=True,
                        help="Comma-separated 16-character pattern using T (must be True), F (must be False), or X (ignored).")

    args = parser.parse_args()
    status_pattern = parse_status_arg(args.status)

    # Load the FITS file and access STATUS column
    with fits.open(args.input_file) as hdul:
        data = hdul[1].data
        status_array = data["STATUS"]  # shape: (N, 16), dtype=bool

        # Initialize selection mask to all True
        mask = np.ones(len(status_array), dtype=bool)

        # Apply filter according to status pattern
        for i, flag in enumerate(status_pattern):
            if flag == 'T':
                mask &= status_array[:, i]
            elif flag == 'F':
                mask &= ~status_array[:, i]
            # If flag == 'X', do nothing (ignore)

        # Report statistics
        print(f"Total rows: {len(status_array)}")
        print(f"Matched rows: {np.sum(mask)}")

        # Extract filtered rows and create output HDU
        filtered_data = data[mask]
        new_hdu = fits.BinTableHDU(data=filtered_data, header=hdul[1].header)
        hdul_new = fits.HDUList([hdul[0], new_hdu])
        hdul_new.writeto(args.output_file, overwrite=True)

        print(f"Filtered FITS saved to: {args.output_file}")

if __name__ == "__main__":
    main()
