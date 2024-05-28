#!/usr/bin/env python

import argparse
import numpy as np
from astropy.io import fits

def calc_trigtime(WFRB_WRITE_LP, WFRB_SAMPLE_CNT, TRIG_LP):
    """
    Calculate the trigger time (SAMPLECNTTRIG_WO_VERNIER) based on the input parameters.
    """
    deltalap = (((WFRB_WRITE_LP >> 18) & 0x3f) - ((TRIG_LP >> 18) & 0x3f)) & 0x3f
    if deltalap == 63:
        deltalap = -1
    SAMPLECNTTRIG_WO_VERNIER = ((WFRB_SAMPLE_CNT - deltalap * 0x40000) + (TRIG_LP & 0x3ffff)) & 0xffffffff
    return SAMPLECNTTRIG_WO_VERNIER

def compute_diff_with_overflow(counter_list, bit_width):
    """
    Compute the differences between consecutive elements in counter_list, considering overflow.
    """
    max_value = (1 << bit_width) - 1
    diffs = []
    for i in range(1, len(counter_list)):
        diff = counter_list[i] - counter_list[i - 1]
        if diff < 0:
            diff += max_value + 1
        diffs.append(diff)
    return np.array(diffs)

def update_intervals(fits_file, output_file=None):
    """
    Update the PREV_INTERVAL and NEXT_INTERVAL columns in the FITS file.
    """
    with fits.open(fits_file, mode='update' if output_file is None else 'readonly') as hdul:
        data = hdul[1].data
        pixel_numbers = np.arange(36)

        for pixel in pixel_numbers:
            # Get indices of the current pixel
            pixel_indices = np.where(data['PIXEL'] == pixel)[0]
            WFRB_WRITE_LP = data['WFRB_WRITE_LP'][pixel_indices]
            WFRB_SAMPLE_CNT = data['WFRB_SAMPLE_CNT'][pixel_indices]
            TRIG_LP = data['TRIG_LP'][pixel_indices]

            # Calculate SAMPLECNTTRIG_WO_VERNIER for each event
            SAMPLECNTTRIG_WO_VERNIER = [calc_trigtime(wlp, wsp, tlp) for wlp, wsp, tlp in zip(WFRB_WRITE_LP, WFRB_SAMPLE_CNT, TRIG_LP)]

            # Compute previous and next intervals considering overflow
            prev_intervals = compute_diff_with_overflow(SAMPLECNTTRIG_WO_VERNIER, 24)
            next_intervals = compute_diff_with_overflow(SAMPLECNTTRIG_WO_VERNIER[::-1], 24)[::-1]

            # Debug print to check intervals
            print(f"Pixel: {pixel}, PREV_INTERVAL: {prev_intervals}")
            print(f"Pixel: {pixel}, NEXT_INTERVAL: {next_intervals}")

            # Update data arrays using direct indexing
            if len(prev_intervals) > 0:
                data['PREV_INTERVAL'][pixel_indices[1:]] = prev_intervals
            if len(next_intervals) > 0:
                data['NEXT_INTERVAL'][pixel_indices[:-1]] = next_intervals

            # Verify that the data array is updated
            print(f"Updated PREV_INTERVAL for pixel {pixel}: {data['PREV_INTERVAL'][pixel_indices]}")
            print(f"Updated NEXT_INTERVAL for pixel {pixel}: {data['NEXT_INTERVAL'][pixel_indices]}")

        if output_file:
            hdul.writeto(output_file, overwrite=True)
            print(f"{output_file} is overwritten.")
        else:
            hdul.flush()
            print(f"{fits_file} is updated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='This program adds PREV_INTERVAL and NEXT_INTERVAL to a FITS file.',
      epilog='''
        Example 1) Overwrite the original FITS file:
        resolve_tool_addcol_prev_next_interval.py xa000114000rsl_p0px1000_uf.evt 
        Example 2) Create a new file:
        resolve_tool_addcol_prev_next_interval.py xa000114000rsl_p0px1000_uf.evt -o xa000114000rsl_p0px1000_uf_prevnext.evt
      ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("fits_file", type=str, help="Path to the input FITS file")
    parser.add_argument("--output_file", "-o", type=str, help="Path to the output FITS file. If not specified, the input file will be updated", default=None)

    args = parser.parse_args()

    update_intervals(args.fits_file, args.output_file)
