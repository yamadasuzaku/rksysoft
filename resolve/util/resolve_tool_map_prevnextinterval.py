#!/usr/bin/env python

import argparse
import numpy as np
from astropy.io import fits

def map_intervals(source_file, target_file, output_file=None):
    """
    Map PREV_INTERVAL and NEXT_INTERVAL from the source file to the target file based on TIME.
    """
    with fits.open(source_file) as source_hdul, fits.open(target_file, mode='update' if output_file is None else 'readonly') as target_hdul:
        source_data = source_hdul[1].data
        target_data = target_hdul[1].data

        pixel_numbers = np.arange(36)
        total_pixels = len(pixel_numbers)

        for idx, pixel in enumerate(pixel_numbers):
            # Extract data for the current pixel
            source_pixel_mask = source_data['PIXEL'] == pixel
            target_pixel_mask = target_data['PIXEL'] == pixel

            source_times = source_data['TIME'][source_pixel_mask]
            target_times = target_data['TIME'][target_pixel_mask]

            source_prev_intervals = source_data['PREV_INTERVAL'][source_pixel_mask]
            source_next_intervals = source_data['NEXT_INTERVAL'][source_pixel_mask]

            target_prev_intervals = target_data['PREV_INTERVAL'][target_pixel_mask]
            target_next_intervals = target_data['NEXT_INTERVAL'][target_pixel_mask]

            # Ensure source_times are sorted
            sorted_indices = np.argsort(source_times)
            source_times = source_times[sorted_indices]
            source_prev_intervals = source_prev_intervals[sorted_indices]
            source_next_intervals = source_next_intervals[sorted_indices]

            # Find the closest source time for each target time using searchsorted
            indices = np.searchsorted(source_times, target_times)

            # Initialize prev and next intervals for the target pixel
            for i, index in enumerate(indices):
                if index >= len(source_times):
                    index = len(source_times) - 1
                elif index > 0 and (index == len(source_times) or abs(source_times[index-1] - target_times[i]) < abs(source_times[index] - target_times[i])):
                    index -= 1
                target_prev_intervals[i] = source_prev_intervals[index]
                target_next_intervals[i] = source_next_intervals[index]

            # Update the target data arrays
            target_data['PREV_INTERVAL'][target_pixel_mask] = target_prev_intervals
            target_data['NEXT_INTERVAL'][target_pixel_mask] = target_next_intervals

            # Print progress
            print(f"Completed pixel {pixel} ({idx+1}/{total_pixels})")

        if output_file:
            target_hdul.writeto(output_file, overwrite=True)
            print(f"{output_file} is overwritten.")
        else:
            target_hdul.flush()
            print(f"{target_file} is updated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='Map PREV_INTERVAL and NEXT_INTERVAL from a source FITS file to a target FITS file.',
      epilog='''
        Example 1) Overwrite the original target FITS file:
        resolve_tool_map_interval_quick.py xa000114000rsl_p0px1000_uf_prevnext.evt xa000114000rsl_a0pxpr_uf.evt 
        Example 2) Create a new target FITS file:
        resolve_tool_map_interval_quick.py xa000114000rsl_p0px1000_uf_prevnext.evt xa000114000rsl_a0pxpr_uf.evt -o xa000114000rsl_a0pxpr_uf_fillprenext.evt
      ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("source_file", type=str, help="Path to the source FITS file with PREV_INTERVAL and NEXT_INTERVAL columns")
    parser.add_argument("target_file", type=str, help="Path to the target FITS file to be updated")
    parser.add_argument("--output_file", "-o", type=str, help="Path to the output FITS file. If not specified, the target file will be updated", default=None)

    args = parser.parse_args()

    map_intervals(args.source_file, args.target_file, args.output_file)
