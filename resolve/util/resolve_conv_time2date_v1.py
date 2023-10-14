#!/usr/bin/env python
# coding: utf-8

import sys
import datetime
from astropy.time import Time

def print_usage_and_exit(script_name):
    """Print the correct usage of the script and exit."""
    print(f'Usage: # python {script_name} <start_time> <end_time> (e.g., 149528387.0 149608821.0)')
    sys.exit()

def compute_date_from_reference(date_sec, reference_time):
    """Compute date from a reference time for a given second."""
    date = reference_time.datetime + datetime.timedelta(seconds=float(date_sec))
    return date

def main():
    # Check for the correct number of arguments
    if len(sys.argv) != 3:
        print_usage_and_exit(sys.argv[0])

    # MJD reference day 01 Jan 2019 00:00:00
    MJD_REFERENCE_DAY = 58484
    reference_time = Time(MJD_REFERENCE_DAY, format='mjd')

    start_sec = sys.argv[1]
    end_sec = sys.argv[2]

    start_date = compute_date_from_reference(start_sec, reference_time)
    end_date = compute_date_from_reference(end_sec, reference_time)

    print(start_date, end_date)

if __name__ == "__main__":
    main()
