#!/usr/bin/env python

import h5py
import numpy as np
import argparse
import re

# Set up argument parser
parser = argparse.ArgumentParser(description="Process and sort channel keys in the HDF5 file.")
parser.add_argument('input_file', type=str, help="Path to the input HDF5 file (required).")
args = parser.parse_args()

# Open the HDF5 file in read-only mode
with h5py.File(args.input_file, "r") as f:
    # Retrieve the list of all channel keys (chanNN)
    chan_keys = list(f.keys())

    # Separate channels into 4-digit and 3-digit or less categories
    chan_keys_4digit = [key for key in chan_keys if re.match(r'^chan\d{4}$', key)]  # Bi TES (4-digit channels)
    chan_keys_short = [key for key in chan_keys if re.match(r'^chan\d{1,3}$', key)]  # Sn TES (1-3 digit channels)

    # Extract the numeric part of the 'chan' keys and convert them to integers
    chan_keys_4digit_int = [int(re.sub(r'\D', '', key)) for key in chan_keys_4digit]  # Extract numbers from 4-digit channels
    chan_keys_short_int = [int(re.sub(r'\D', '', key)) for key in chan_keys_short]  # Extract numbers from 1-3 digit channels

    # Sort the numeric values in ascending order
    chan_keys_4digit_int_sorted = sorted(chan_keys_4digit_int)  # Sort 4-digit channel numbers
    chan_keys_short_int_sorted = sorted(chan_keys_short_int)  # Sort 1-3 digit channel numbers

    # Output the sorted results
    print(f"Sorted 4-digit channel keys as integers (#={len(chan_keys_4digit_int_sorted)}):", chan_keys_4digit_int_sorted)
    print(f"Sorted 3-digit or less channel keys as integers (#={len(chan_keys_short_int_sorted)}):", chan_keys_short_int_sorted)
