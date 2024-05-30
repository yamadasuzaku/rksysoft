#!/usr/bin/env python

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

# Constants
MJD_REFERENCE_DAY = 58484
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')
TIME_50MK = 150526800.0

def parse_filter_conditions(conditions):
    filters = []
    for condition in conditions.split(","):
        col, value = condition.split("==")
        filters.append((col.strip(), float(value.strip())))
    return filters

def apply_filters(data, filters):
    mask = np.ones(len(data), dtype=bool)
    for col, value in filters:
        mask &= (data[col] == value)
    return data[mask]

def plot_fits_data(file_name, x_col, y_col, hdu, title, outfname, tolerance, filters=None, plotflag=False, markers=".", datatime_flag=True):
    print(f"Opening FITS file: {file_name}")
    with fits.open(file_name) as hdul:
        data = hdul[hdu].data  # Access the data from the specified HDU

        # Apply filters if any
        if filters:
            print("Applying filters...")
            data = apply_filters(data, filters)
            print(f"Filtered data: {len(data)} rows")

        # Convert columns to numpy arrays for easier plotting
        x_data = data[x_col]
        if datatime_flag:
            print("Converting time data...")
            x_data = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in x_data])
        y_data = data[y_col]

        # Split the sequence based on the tolerance value
        print("Splitting the sequence...")
        split_sequences = split_sequence(y_data, tolerance)

        # Visualization
        print("Creating plots...")
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        plt.suptitle(title)

        # Top plot: Visualizing the continuous segments
        axs[0].plot(x_data, y_data, markers, label=y_col)

        for j, (indexes, segment) in enumerate(split_sequences):
            if j == 0:
                axs[0].plot(x_data[indexes], y_data[indexes], "ro", alpha=0.2, label="meet condition")
                axs[0].legend()
            else:
                axs[0].plot(x_data[indexes], y_data[indexes], "ro", alpha=0.2)

        axs[0].axhline(y=tolerance, color='r', linestyle='--', label="threshold = " + str(tolerance))
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel(y_col)
        axs[0].set_title(f'Continuous Segments Where {y_col}[i] < {tolerance} continues more than once.')
        axs[0].set_ylim(-10, 150)

        # Bottom plot: Histogram of segment lengths
        segment_lengths = [len(segment) for _, segment in split_sequences]
        bins = np.arange(0.5, max(segment_lengths) + 0.5, 1)
        axs[1].hist(segment_lengths, bins=bins, edgecolor='black', label="Length in a subset of " + y_col)
        axs[1].set_xticks(np.arange(2, max(segment_lengths) + 1))
        axs[1].set_xlabel('Length of the continuous Segments more than one')
        axs[1].set_ylabel('Frequency')
        axs[1].set_yscale("log")
        axs[1].set_title('Histogram of Segment Lengths')
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(outfname)
        print(f"Saved plot as {outfname}")
        if plotflag:
            plt.show()
        else:
            plt.close()

def split_sequence(seq, k, minnum=2):
    print("Starting sequence splitting...")
    result = []
    current_segment = []
    current_indices = []

    for i, num in enumerate(seq):
        if num < k:
            current_segment.append(num)
            current_indices.append(i)
        else:
            if len(current_segment) >= minnum:
                result.append((current_indices, current_segment))
                print(f"Found segment: {current_segment} at indices {current_indices}")
            current_segment = []
            current_indices = []

    if len(current_segment) >= minnum:
        result.append((current_indices, current_segment))
        print(f"Found segment: {current_segment} at indices {current_indices}")

    print("Sequence splitting complete.")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This program is used to check more than two-something groups for each pixel',
        epilog='''
        Example:
        resolve_ana_pixel_Ls_mksubgroup.py xa000114000rsl_p0px1000_cl.evt TIME TRIG_LP -f "PIXEL==9" -p
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file_name", type=str, help="Path to the FITS file")
    parser.add_argument("x_col", type=str, help="Column name for the x-axis")
    parser.add_argument("y_col", type=str, help="Column name for the y-axis")
    parser.add_argument('--hdu', '-n', type=int, default=1, help='Number of FITS HDU')
    parser.add_argument("--filters", '-f', type=str, help="Comma-separated filter conditions in the format 'COLUMN==VALUE'", default="")
    parser.add_argument('--plot', '-p', action='store_true', default=False, help='Flag to display plot')
    parser.add_argument("--markers", '-m', type=str, help="Marker type", default=".")
    parser.add_argument('--tolerance', '-t', type=float, default=100, help='Tolerance for subset threshold')

    args = parser.parse_args()
    filter_conditions = parse_filter_conditions(args.filters) if args.filters else None
    title = f"{args.file_name} : filtered with {args.filters}"
    print(filter_conditions)
    ftag = args.file_name.split(".")[0]
    if args.filters:
        outfname = f"subset_{ftag}_{filter_conditions[0][0]}{int(filter_conditions[0][1]):02d}.png"
    else:
        outfname = f"subset_{ftag}.png"

    print("Starting plot generation...")
    plot_fits_data(args.file_name, args.x_col, args.y_col, args.hdu, title, outfname, args.tolerance, filter_conditions, args.plot, args.markers)
    print("Plot generation complete.")
