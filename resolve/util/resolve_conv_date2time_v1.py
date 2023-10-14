#!/usr/bin/env python
# coding: utf-8

import sys
import datetime
from astropy.time import Time

def print_usage_and_exit(script_name):
    """Print the correct usage of the script and exit."""
    print(f'Usage: # python {script_name} <start_date> <end_date> (e.g., 2023-09-27T15:39:47 2023-09-28T14:00:21)')
    sys.exit()

def compute_seconds_from_reference(date_string, reference_time):
    """Compute the total seconds difference from a reference time for a given date string."""
    date_object = datetime.datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S")
    return (date_object - reference_time.datetime).total_seconds()

def main():
    # Check for the correct number of arguments
    if len(sys.argv) != 3:
        print_usage_and_exit(sys.argv[0])

    # MJD reference day 01 Jan 2019 00:00:00
    MJD_REFERENCE_DAY = 58484
    reference_time = Time(MJD_REFERENCE_DAY, format='mjd')

    start_date = sys.argv[1]
    end_date = sys.argv[2]

    start_seconds = compute_seconds_from_reference(start_date, reference_time)
    end_seconds = compute_seconds_from_reference(end_date, reference_time)

    print(start_seconds, end_seconds)

if __name__ == "__main__":
    main()
