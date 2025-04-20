#!/usr/bin/env python

import h5py
import numpy as np
import matplotlib.pyplot as plt
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)
import argparse
from datetime import datetime
from scipy.signal import find_peaks

# Set up argument parser
parser = argparse.ArgumentParser(description="Plot histograms of timestamps from HDF5 file.")
parser.add_argument('input_file', type=str, help="Path to the input HDF5 file (required).")
parser.add_argument('-o', '--output_file', type=str, default='output_checkphase.png', help="Output PNG file name (default: output.png).")
parser.add_argument('--bins', type=int, default=4000, help="Number of bins for the second histogram (default: 400).")
parser.add_argument('--xlim', type=float, nargs=2, metavar=('XMIN', 'XMAX'), help="Set x-axis limits for the second histogram (default: auto).")
parser.add_argument('--ylim', type=float, nargs=2, metavar=('YMIN', 'YMAX'), help="Set y-axis limits for the second histogram (default: auto).")
args = parser.parse_args()

# Open the HDF5 file in read-only mode
with h5py.File(args.input_file, "r") as f:
    # Retrieve the list of all channel keys (chanNN)
    chan_keys = list(f.keys())
    
    # Initialize a list to store timestamps
    all_timestamps = []
    
    # Loop through each channel and append the 'timestamp' data to the list
    for chan in chan_keys:
        timestamps = f[chan]['timestamp'][()]
        all_timestamps.append(timestamps)
    
    # Convert the list of timestamps into a single numpy array
    all_timestamps_array = np.concatenate(all_timestamps)

    # Sort the timestamps in ascending order
    all_timestamps_array = np.sort(all_timestamps_array)
    
    # Get the start and stop timestamps (first and last values in the array)
    tstart = all_timestamps_array[0]
    tstop = all_timestamps_array[-1]
    
    # Convert the start and stop POSIX timestamps into UTC datetime format
    start_time = datetime.utcfromtimestamp(tstart)
    stop_time = datetime.utcfromtimestamp(tstop)
    
    # Print the start and stop POSIX timestamps, and their UTC datetime equivalents
    print(f"tstart = {tstart} -- tstop = {tstop}")
    print(f"start_time = {start_time} -- stop_time = {stop_time}")

# Create a figure for plotting the histograms
plt.figure(figsize=(18, 6))

# First plot: Histogram of all timestamps
plt.subplot(1, 3, 1)
plt.hist(all_timestamps_array, bins=50, color='blue', alpha=0.7)
plt.title(f'Histogram of all_timestamps_array\n({args.input_file})')
plt.xlabel('Timestamp')
plt.ylabel('Frequency')

# Add text with tstart, tstop, start_time, stop_time, and event count
event_count = len(all_timestamps_array)
text_str = (
    f"tstart: {tstart}\ntstop: {tstop}\n"
    f"start_time: {start_time}\nstop_time: {stop_time}\n"
    f"Event Count: {event_count}"
)
plt.text(0.05, 0.25, text_str, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

# Second plot: Histogram of the differences between consecutive timestamps
plt.subplot(1, 3, 2)
plt.hist(np.diff(all_timestamps_array), bins=args.bins, color='green', alpha=0.7)
plt.yscale("log")
# Set x and y limits for the second plot if provided, otherwise use auto
if args.xlim:
    plt.xlim(args.xlim)
if args.ylim:
    plt.ylim(args.ylim)

plt.title(f'Histogram of np.diff(all_timestamps_array)\n({args.input_file})')
plt.xlabel('Difference in Timestamps')
plt.ylabel('Frequency')

# Third plot: Histogram of remainders when divided by the period (1/25 Hz)
plt.subplot(1, 3, 3)
period = 1 / 25  # 25 Hz
remainders = np.mod(all_timestamps_array, period)  # Calculate the remainder (fractional part) of each timestamp divided by the period
counts, bin_edges, _ = plt.hist(remainders, bins=400, color='red', alpha=0.7)
# Find the peak(s) in the histogram
peaks, _ = find_peaks(counts)

# Find the index of the peak with the maximum y value
max_peak_index = peaks[np.argmax(counts[peaks])]

# Get the x and y values of the peak with the maximum y value
max_peak_x = bin_edges[max_peak_index]
max_peak_y = counts[max_peak_index]

# Print the peak x and y values
print(f"Peak with the highest y value: x = {max_peak_x}, y = {max_peak_y}")
plt.text(0.05, 0.65, f"highest y at x = {max_peak_x}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
# Mark the peak with the highest y value on the plot
plt.plot(max_peak_x, max_peak_y, 'bo', markersize=8)
plt.yscale("log")
plt.title(f'Remainders of timestamps mod {period}')
plt.xlabel('Remainder (mod period)')
plt.ylabel('Frequency')

# Adjust the layout to ensure there is no overlap between subplots
plt.tight_layout()

# Save the plot to the specified output file
plt.savefig(args.output_file)

# Display the plot
plt.show()
