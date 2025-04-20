#!/usr/bin/env python 

import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

from scipy.signal import find_peaks

PERIOD_25Hz=0.04 # 0.04 sec = 25 Hz

def process_timestamps(input_file, period=1/25, bins=400):
    """
    Process the timestamps from the given HDF5 file, calculate the remainders
    when divided by the given period, and find the peak with the maximum y value.

    Parameters:
    input_file (str): Path to the input HDF5 file.
    period (float): Period to calculate the remainders (default 1/25 for 25 Hz).
    bins (int): Number of bins to use for histogram calculation (default 400).

    Returns:
    tuple: Peak x, y values of the highest peak in the histogram.
    """
    # Open the HDF5 file in read-only mode
    with h5py.File(input_file, "r") as f:
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
    
    # Calculate the remainder (fractional part) of each timestamp divided by the period
    remainders = np.mod(all_timestamps_array, period)

    # Calculate the histogram of the remainders
    counts, bin_edges = np.histogram(remainders, bins=bins, range=(0, period))

    # Find the peak(s) in the histogram
    peaks, _ = find_peaks(counts)

    # Find the index of the peak with the maximum y value
    max_peak_index = peaks[np.argmax(counts[peaks])]

    # Get the x and y values of the peak with the maximum y value
    max_peak_x = bin_edges[max_peak_index]
    max_peak_y = counts[max_peak_index]
    event_count = len(all_timestamps_array)

    return max_peak_x, max_peak_y, event_count

def main():
    """
    Main function to handle argument parsing and call the processing function.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process timestamps and find peak values.")
    parser.add_argument('input_file', type=str, help="Path to the input HDF5 file.")
    parser.add_argument('--period_min', type=float, default=0.0399999, help="Minimum period value (default: 0.0399999).")
    parser.add_argument('--period_max', type=float, default=0.0400001, help="Maximum period value (default: 0.0400001).")
    parser.add_argument('--period_step', type=float, default=0.25e-8, help="Step size for period (default: 1e-7).")
    parser.add_argument('--bins', type=int, default=400, help="Number of bins for histogram (default: 400).")
    parser.add_argument('-o', '--output_file', type=str, default='output_find_bestphase.png', help="Output PNG file name (default: output.png).")
    
    # Parse arguments
    args = parser.parse_args()

    # Create lists to store period vs. peak values
    periods = []
    max_peak_x_values = []
    max_peak_y_values = []

    # Loop through the period range and calculate the peak values for each period
    period = args.period_min
    while period <= args.period_max:
        max_peak_x, max_peak_y, event_count = process_timestamps(args.input_file, period=period, bins=args.bins)
        
        # Store the results
        periods.append(period)
        max_peak_x_values.append(max_peak_x)
        max_peak_y_values.append(max_peak_y)
        
        # Increment the period
        period += args.period_step

    periods = np.array(periods)
    # Plot the period vs. max_peak_x and period vs. max_peak_y
    plt.figure(figsize=(12, 6))
    # Plot period vs max_peak_x
    plt.subplot(1, 2, 1)
    plt.plot(periods - PERIOD_25Hz, max_peak_x_values, color='blue', marker='o', markersize=4)
    plt.title(f"Period vs max_peak_x\n {args.input_file}")
    plt.xlabel("Period (s) - 0.04 sec ")
    plt.ylabel("max_peak_x")
    plt.grid(alpha=0.8)
    
    # Plot period vs max_peak_y
    plt.subplot(1, 2, 2)    
    plt.plot(periods - PERIOD_25Hz, max_peak_y_values, color='red', marker='o', markersize=4)
    plt.text(0.05, 0.25, f"event_count = {event_count}", 
        transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    plt.title("Period vs max_peak_y")
    plt.xlabel("Period (s) - 0.04 sec")
    plt.ylabel("max_peak_y")
    plt.grid(alpha=0.8)

    # Display the plots
    plt.tight_layout()
    plt.savefig(args.output_file)
    plt.show()

if __name__ == "__main__":
    main()
