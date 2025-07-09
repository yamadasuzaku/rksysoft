#!/usr/bin/env python

"""
Script Name: resolve_tool_map_cluster.py

Description:
This script maps cluster information from a source FITS file to a target FITS file.

History:
- ver 1, 2025.2.26, first draft
- ver 1.1, 2025.7.7, added cluster-mode option for extended columns
- ver 1.2, 2025.7.7, refactored to reduce duplicate code
"""

import argparse
import numpy as np
from astropy.io import fits
import os
import sys
import matplotlib.pyplot as plt

def map_clusters(source_file, target_file, output_file=None, dT=10.0, cluster_mode="simple", debug=False):
    """
    Map cluster information from the source file to the target file based on TIME.
    """
    # Define mode -> column names mapping
    mode_to_columns = {
        "simple": [("ICLUSTER", "IMEMBER")],
        "extended": [
            ("ICLUSTERL", "IMEMBERL"),
            ("ICLUSTERS", "IMEMBERS")
        ]
    }

    if cluster_mode not in mode_to_columns:
        print(f"ERROR: Unsupported cluster_mode '{cluster_mode}'")
        sys.exit(1)

    diff_list = [] 

    with fits.open(source_file) as source_hdul, fits.open(target_file) as target_hdul:
        source_data = source_hdul[1].data
        target_data = target_hdul[1].data

        n_rows = len(target_data)
        # Initialize arrays for each column pair
        result_arrays = {}
        for col_pair in mode_to_columns[cluster_mode]:
            for col_name in col_pair:
                result_arrays[col_name] = np.zeros(n_rows, dtype=np.int32)

        # Process pixel by pixel
        for pixel in np.unique(target_data['PIXEL']):
            print(f"..... pixel = {pixel}")
            source_mask = source_data['PIXEL'] == pixel
            target_mask = target_data['PIXEL'] == pixel
            print(f"target_mask.sum() = {np.sum(target_mask)}")

            source_times = source_data['TIME'][source_mask]
            target_times = target_data['TIME'][target_mask]

            # Extract source columns for current pixel
            source_columns = {}
            for col_pair in mode_to_columns[cluster_mode]:
                for col_name in col_pair:
                    source_columns[col_name] = source_data[col_name][source_mask]

            indices = np.searchsorted(source_times, target_times)

            for i, index in enumerate(indices):
                if index >= len(source_times):
                    index = len(source_times) - 1
                elif index > 0 and abs(source_times[index-1] - target_times[i]) < abs(source_times[index] - target_times[i]):
                    index -= 1

                time_diff = abs(source_times[index] - target_times[i])  # diffを計算

                if abs(time_diff) <= dT:
                    # Assign matched cluster/member values
                    for col_pair in mode_to_columns[cluster_mode]:
                        for col_name in col_pair:
                            abs_idx = np.where(target_mask)[0][i]
                            result_arrays[col_name][abs_idx] = source_columns[col_name][index]
                            if debug:
                                print(f"source_columns[{col_name}][{index}] = {source_columns[col_name][index]}")
                                print(f"Assigning result_arrays[{col_name}][{abs_idx}] = {source_columns[col_name][index]}")
                else:
                    if debug:
                        print(f"[Warning] Pixel {pixel}, Row {i}: Time difference too large")

                if debug:
                    diff_list.append(time_diff)

        if debug and diff_list:
            plt.figure(figsize=(8, 6))
            plt.hist(diff_list, bins=50, edgecolor='black')
            plt.title('Histogram of Time Differences')
            plt.xlabel('Time Difference (s)')
            plt.ylabel('Counts')

            # diff==0とdiff!=0の数
            num_zero = np.sum(np.array(diff_list) == 0)
            num_nonzero = np.sum(np.array(diff_list) != 0)

            # テキストを描画
            plt.text(0.95, 0.95, f"diff=0: {num_zero}\ndiff≠0: {num_nonzero}",
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     horizontalalignment='right',
                     fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.8))

            plt.show()

        # Add new columns
        col_defs = target_hdul[1].columns
        new_cols = list(col_defs)  # Start with existing columns
        for col_name, array in result_arrays.items():
            new_cols.append(fits.Column(name=col_name, format='J', array=array))



        # Write new HDU
        new_hdu = fits.BinTableHDU.from_columns(new_cols, header=target_hdul[1].header)
        target_hdul[1] = new_hdu

        # Save
        if output_file:
            target_hdul.writeto(output_file, overwrite=True)
            print(f"Modified FITS file saved as: {output_file}")
        else:
            target_hdul.flush()
            print(f"{target_file} updated in place.")


def main():
    parser = argparse.ArgumentParser(
        description="Map cluster information from a source FITS file to a target FITS file.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("source_file", type=str, help="Path to source FITS file")
    parser.add_argument("target_file", type=str, help="Path to target FITS file")
    parser.add_argument("--outname", "-o", type=str, default="addcluster_", help="Output file name prefix")
    parser.add_argument("--dT", type=float, default=0.001, help="Max allowed time difference for matching")
    parser.add_argument("--cluster-mode", choices=["simple", "extended"], default="extended",
                        help="Cluster mode: simple (ICLUSTER, IMEMBER) or extended (ICLUSTERL, IMEMBERL, ICLUSTERS, IMEMBERS)")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug output")

    args = parser.parse_args()
    output_fits = args.outname + os.path.basename(args.target_file)

    map_clusters(args.source_file, args.target_file, output_file=output_fits, dT=args.dT,
                 cluster_mode=args.cluster_mode, debug=args.debug)


if __name__ == "__main__":
    main()
