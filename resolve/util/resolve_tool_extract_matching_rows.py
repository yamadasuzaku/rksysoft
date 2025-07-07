#!/usr/bin/env python3

import argparse
from astropy.io import fits
import numpy as np

def extract_and_match(pixel, file1, file2, output_file, debug=False):
    # Open the first FITS file
    with fits.open(file1) as hdul1:
        data1 = hdul1[1].data

        # Filter rows with the specified PIXEL number
        pixel_mask1 = data1['PIXEL'] == pixel
        if not np.any(pixel_mask1):
            print(f"No rows with PIXEL={pixel} found in {file1}")
            return

        # Extract unique combinations of the three columns
        keys1 = np.array([
            data1['WFRB_WRITE_LP'][pixel_mask1],
            data1['WFRB_SAMPLE_CNT'][pixel_mask1],
            data1['TRIG_LP'][pixel_mask1]
        ]).T

        if debug:
            print(f"--- Debug: {file1} ---")
            print(f"Total rows: {len(data1)}")
            print(f"Rows with PIXEL={pixel}: {np.sum(pixel_mask1)}")
            print("Extracted keys (WFRB_WRITE_LP, WFRB_SAMPLE_CNT, TRIG_LP):")
            for k in keys1[:5]:  # Show first 5 keys
                print(f"  {tuple(k)}")
            if len(keys1) > 5:
                print(f"  ... and {len(keys1)-5} more keys")
            print("-----------------------")

    # Open the second FITS file
    with fits.open(file2) as hdul2:
        data2 = hdul2[1].data
        header2 = hdul2[1].header  # Keep original header

        if debug:
            print(f"--- Debug: {file2} ---")
            print(f"Total rows: {len(data2)}")
            print(f"Rows with PIXEL={pixel}: {np.sum(data2['PIXEL'] == pixel)}")
            print("-----------------------")

        # Build mask for matching rows
        match_mask = np.zeros(len(data2), dtype=bool)
        for idx, key in enumerate(keys1):
            cond = (
                (data2['PIXEL'] == pixel) &
                (data2['WFRB_WRITE_LP'] == key[0]) &
                (data2['WFRB_SAMPLE_CNT'] == key[1]) &
                (data2['TRIG_LP'] == key[2])
            )
            match_count = np.sum(cond)
            if debug and idx < 5:  # Show only first 5 keys in detail
                print(f"Key {idx+1}/{len(keys1)} {tuple(key)}: Matches in {file2} = {match_count}")
            match_mask |= cond

        total_matches = np.sum(match_mask)
        if debug:
            print("-----------------------")
            print(f"Total matching rows found: {total_matches}")

        if total_matches == 0:
            print(f"No matching rows found in {file2}")
            return

        # Extract matching rows
        matching_rows = data2[match_mask]

        # Create new BinTableHDU from the matching rows
        hdu = fits.BinTableHDU(data=matching_rows, header=header2)

        # Write to output file
        hdu.writeto(output_file, overwrite=True)
        print(f"Saved {total_matches} matching rows to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract rows from second FITS file matching PIXEL and columns from first FITS file."
    )
    parser.add_argument('pixel', type=int, help="PIXEL number to filter (0–35)")
    parser.add_argument('file1', type=str, help="Path to first FITS file")
    parser.add_argument('file2', type=str, help="Path to second FITS file")
    parser.add_argument('-o', '--output', type=str, default="matched_output.fits",
                        help="Output FITS filename (default: matched_output.fits)")
    parser.add_argument('-d', '--debug', action='store_true',
                        help="Enable debug output")
    args = parser.parse_args()

    extract_and_match(args.pixel, args.file1, args.file2, args.output, args.debug)

if __name__ == "__main__":
    main()
