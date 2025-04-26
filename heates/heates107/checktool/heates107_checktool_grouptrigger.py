#!/usr/bin/env python

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import re

# Set matplotlib parameters for better readability
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8})

def parse_arguments():
    """Set up and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot timestamps, energies, and channels from HDF5 file.")
    parser.add_argument('input_file', type=str, help="Path to the input HDF5 file (required).")
    parser.add_argument('-o', '--output_file', type=str, default='output_check_grouptrigger.png',
                        help="Output PNG file name (default: output_check_grouptrigger.png).")
    return parser.parse_args()

def extract_and_sort_channel_keys(f):
    """Extract channel keys and sort them based on channel numbers."""
    chan_keys = list(f.keys())
    # Separate 4-digit and shorter channel keys
    chan_keys_4digit = [key for key in chan_keys if re.match(r'^chan\d{4}$', key)]
    chan_keys_short = [key for key in chan_keys if re.match(r'^chan\d{1,3}$', key)]

    # Extract channel numbers
    chan_keys_4digit_int = [int(re.sub(r'\D', '', key)) for key in chan_keys_4digit]
    chan_keys_short_int = [int(re.sub(r'\D', '', key)) for key in chan_keys_short]

    # Sort channel keys together with their numbers
    chan_keys_4digit_sorted = sorted(zip(chan_keys_4digit_int, chan_keys_4digit))
    chan_keys_short_sorted = sorted(zip(chan_keys_short_int, chan_keys_short))

    return chan_keys_4digit_sorted, chan_keys_short_sorted

def load_data(f, chan_keys_short_sorted):
    """Load timestamp, energy, and channel number from the file."""
    all_timestamps = []
    all_energy = []
    all_chan = []
    all_num = []

    for i, (channum, chan) in enumerate(chan_keys_short_sorted):
        print(f"Loading channel: {chan}")
        timestamps = f[chan]['timestamp'][()]
        energy = f[chan]['energy'][()]

        all_timestamps.extend(timestamps)
        all_energy.extend(energy)
        all_chan.extend([channum] * len(timestamps))
        all_num.extend([i] * len(timestamps))

    return np.array(all_timestamps), np.array(all_energy), np.array(all_chan), np.array(all_num)

def plot_data(all_num, all_timestamps, all_energy, all_chan, output_file):
    """Plot the timestamps, energies, and channel numbers."""
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axs[0].plot(all_num, all_timestamps, ".")
    axs[0].set_ylabel('Timestamp')

    axs[1].plot(all_num, all_energy, ".")
    axs[1].set_ylabel('Energy')

    axs[2].plot(all_num, all_chan, ".")
    axs[2].set_ylabel('Channel')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()

def main():
    args = parse_arguments()

    with h5py.File(args.input_file, "r") as f:
        chan_keys_4digit_sorted, chan_keys_short_sorted = extract_and_sort_channel_keys(f)

        print(f"4-digit channels (sorted) (#={len(chan_keys_4digit_sorted)}): {[k for k, _ in chan_keys_4digit_sorted]}")
        print(f"Short channels (sorted) (#={len(chan_keys_short_sorted)}): {[k for k, _ in chan_keys_short_sorted]}")

        all_timestamps, all_energy, all_chan, all_num = load_data(f, chan_keys_short_sorted)

        # Print number of events
        print(f"Total number of events: {len(all_timestamps)}")
        
        # Print first and last events for checking
        if len(all_timestamps) > 0:
            print(f"First event: Timestamp={all_timestamps[0]}, Energy={all_energy[0]}, Channel={all_chan[0]}")
            print(f"Last event: Timestamp={all_timestamps[-1]}, Energy={all_energy[-1]}, Channel={all_chan[-1]}")

    plot_data(all_num, all_timestamps, all_energy, all_chan, args.output_file)

if __name__ == "__main__":
    main()
