#!/usr/bin/env python

# history
# 2024.6.20, bug fixed for 16bit int overflow

import argparse
import numpy as np
from astropy.io import fits

def map_intervals(source_file, target_file, output_file=None, dT=10.0, debug=False):
    """
    Map PREV_INTERVAL and NEXT_INTERVAL from the source file to the target file based on TIME.
    """
    with fits.open(source_file) as source_hdul, fits.open(target_file, mode='update' if output_file is None else 'readonly') as target_hdul:
        source_data = source_hdul[1].data
        target_data = target_hdul[1].data

        # Initialize PREV_INTERVAL and NEXT_INTERVAL with -1
        target_data['PREV_INTERVAL'][:] = -1
        target_data['NEXT_INTERVAL'][:] = -1

        pixel_numbers = np.arange(36)
        total_pixels = len(pixel_numbers)

        for idx, pixel in enumerate(pixel_numbers):
            # Extract data for the current pixel
            source_pixel_mask = source_data['PIXEL'] == pixel
            target_pixel_mask = target_data['PIXEL'] == pixel

            source_times = source_data['TIME'][source_pixel_mask]
            target_times = target_data['TIME'][target_pixel_mask]

            source_prev_intervals = source_data['PREV_INTERVAL'][source_pixel_mask].astype(np.int32)
            source_next_intervals = source_data['NEXT_INTERVAL'][source_pixel_mask].astype(np.int32)
            source_status = source_data['STATUS']

            target_prev_intervals = target_data['PREV_INTERVAL'][target_pixel_mask].astype(np.int32)
            target_next_intervals = target_data['NEXT_INTERVAL'][target_pixel_mask].astype(np.int32)
            target_status = target_data['STATUS']

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

                # Check if the closest time difference is within dT
                if abs(source_times[index] - target_times[i]) <= dT:
                    target_prev_intervals[i] = source_prev_intervals[index]
                    target_next_intervals[i] = source_next_intervals[index]
                    target_status[i] = source_status[index]

                    if debug:
                        print("**debug** difference is small", source_times[index], target_times[i], source_times[index] - target_times[i])
                        print("**debug**                    ", source_prev_intervals[index], source_next_intervals[index])                        
                        print("**debug**                    ", target_prev_intervals[i], target_next_intervals[i])                                                
                else:
                    print("[Warning] difference is huge. ", source_times[index], target_times[i], source_times[index] - target_times[i])
                    print("**debug**                    ", target_prev_intervals[i], target_next_intervals[i])                        

            # Clip the intervals to fit into int16 range
            target_prev_intervals = np.clip(target_prev_intervals, -32768, 32767)
            target_next_intervals = np.clip(target_next_intervals, -32768, 32767)

            # Update the target data arrays
            target_data['PREV_INTERVAL'][target_pixel_mask] = target_prev_intervals.astype(np.int16)
            target_data['NEXT_INTERVAL'][target_pixel_mask] = target_next_intervals.astype(np.int16)

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
    parser.add_argument("--dT", type=float, help="Maximum allowed time difference for matching (default: 10.0)", default=10.0)
    parser.add_argument('--debug', '-d', action='store_true', help='Flag to debug')

    args = parser.parse_args()

    map_intervals(args.source_file, args.target_file, args.output_file, args.dT, debug = args.debug)
