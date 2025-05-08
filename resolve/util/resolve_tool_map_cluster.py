#!/usr/bin/env python

"""
Script Name: resolve_tool_map_cluster.py

Description:
This script map the 14th and 15th bits in the STATUS column of a FITS file
for each pixel and adds a new column for row indices for pseudo-Ls events, 
as a reference to evnt file, to pulse records.  

History:
- ver 1, 2025.2.26, first draft. 
"""

import argparse
import matplotlib.pyplot as plt
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)
import numpy as np
from astropy.io import fits
from astropy.time import Time
import datetime
import random
from matplotlib.ticker import MaxNLocator

import os 

# Define global variables
MJD_REFERENCE_DAY = 58484
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')  # Reference time in Modified Julian Day

def compute_datetimes(mjdrefi, times):
    reftime = Time(mjdrefi, format='mjd')
    return [reftime.datetime + datetime.timedelta(seconds=float(t)) for t in times]


def map_clusters(source_file, target_file, output_file=None, dT=10.0, debug=False):
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

            target_prev_intervals = target_data['PREV_INTERVAL'][target_pixel_mask].astype(np.int32)
            target_next_intervals = target_data['NEXT_INTERVAL'][target_pixel_mask].astype(np.int32)

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


def main():

    parser = argparse.ArgumentParser(
      description='Map ICLUSTER, IMEMBER from a source FITS file to a target FITS file.',
      epilog='''
        Example 1) Create a new target FITS file:
        resolve_tool_map_interval_quick.py xa000114000rsl_p0px1000_uf_prevnext.evt xa000114000rsl_a0pxpr_uf.evt -o xa000114000rsl_a0pxpr_uf_fillprenext.evt
      ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("source_file", type=str, help="Path to the source FITS file with ICLUSTER, IMEMBER columns")
    parser.add_argument("target_file", type=str, help="Path to the target FITS file to be updated")
    parser.add_argument('--outname', '-o', type=str, help='fname tag used for output file name', default="addcluster_")
    parser.add_argument("--dT", type=float, help="Maximum allowed time difference for matching (default: 10.0)", default=10.0)
    parser.add_argument('--debug', '-d', action='store_true', help='Flag to debug')
    parser.add_argument('--usepixels', '-p', type=str, help='Comma-separated list of pixels to plot', default=','.join(map(str, range(36))))

    args = parser.parse_args()

    source_file = args.source_file
    target_file = args.target_file
    outname = args.outname
    dT = args.dT
    usepixels = list(map(int, args.usepixels.split(',')))
    debug = args.debug

    output_fits = outname + target_file

    # Open the input FITS file

    with fits.open(source_file) as source_hdul, fits.open(target_file) as target_hdul:

        # Access the STATUS and PIXEL columns
        source_pixel_data =  source_hdul[1].data['PIXEL']
        target_pixel_data =  target_hdul[1].data['PIXEL']

        source_data = source_hdul[1].data
        target_data = target_hdul[1].data

        n_rows = len(target_pixel_data)
        # Create a new column for cluster index
        icluster = np.zeros(n_rows, dtype=np.int32)
        imember = np.zeros(n_rows, dtype=np.int32)

        # Process each pixel independently
        for pixel in usepixels:
            print(f"..... pixel = {pixel}")
            # Get rows corresponding to the current pixel
            source_pixel_mask = source_data['PIXEL'] == pixel
            target_pixel_mask = target_data['PIXEL'] == pixel

            # Process the event list to find cluster
            source_times = source_data['TIME'][source_pixel_mask]
            target_times = target_data['TIME'][target_pixel_mask]

            n_rows_pixel = len(target_times)
            target_tmp_icluster = np.zeros(n_rows_pixel, dtype=np.int32)
            target_tmp_imember = np.zeros(n_rows_pixel, dtype=np.int32)

            # get IMEMBER ICLUSTER from the source file
            source_imember = source_data['IMEMBER'][source_pixel_mask]
            source_icluster = source_data['ICLUSTER'][source_pixel_mask]

            target_imember = source_data['IMEMBER'][source_pixel_mask]
            target_icluster = source_data['ICLUSTER'][source_pixel_mask]


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
                    target_tmp_imember[i] = source_imember[index]
                    target_tmp_icluster[i] = source_icluster[index]
                    if debug:
                        print("**debug** difference is small", source_times[index], target_times[i], \
                                source_times[index] - target_times[i], target_tmp_imember[i], target_tmp_icluster[i])
                else:
                    print("[Warning] difference is huge. ", source_times[index], target_times[i], source_times[index] - target_times[i])

            # Modify ICLUSTER, IMEMBER for the current pixel
            icluster[target_pixel_mask] = target_tmp_icluster
            imember[target_pixel_mask] = target_tmp_imember

        # Add the row indices as a new column
        col_defs = target_hdul[1].columns
        new_col1 = fits.Column(name='ICLUSTER', format='J', array=icluster)
        new_col2 = fits.Column(name='IMEMBER', format='J', array=imember)
#        new_cols = fits.ColDefs(col_defs + new_col1 + new_col2)

        new_cols = []
        for col in col_defs+new_col1+new_col2:
            new_cols.append(col)

        # Create a new binary table with the updated columns
        new_hdu = fits.BinTableHDU.from_columns(new_cols, header=target_hdul[1].header)
        target_hdul[1] = new_hdu
        # Save changes to a new file
        target_hdul.writeto(output_fits, overwrite=True)
        print(f"Modified FITS file saved as: {output_fits}")

if __name__ == "__main__":
    main()
