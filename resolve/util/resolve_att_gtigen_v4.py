#!/usr/bin/env python
# coding: utf-8

import sys
import datetime
import astropy.io.fits as fits
from astropy.time import Time
import numpy as np
import matplotlib.pylab as plt 
import argparse

# Define constants and file names
FILENAME = "xa003702170gen_a0.hk2.gz"
OUTPUT_PNG = "gengti231014.png"
GTI_START_OUTPUT = "gti_start.txt"
GTI_STOP_OUTPUT = "gti_stop.txt"
NCYCLES = 100 # somehow arbitray number to fill zero. 
FIND_MAX = 1e4 # somehow arbitray number to stop calculation 

def load_hk_data(hdulist, extension, column_name, dtype=None):
    """
    Load housekeeping data from the FITS file.

    Parameters:
    - hdulist: HDU list from the FITS file.
    - extension: Name of the FITS file extension.
    - column_name: Name of the column to retrieve data from.
    - dtype: Data type to convert the array to (optional).

    Returns:
    - Numpy array containing the data from the specified FITS file column.
    """
    if dtype == None:
        return np.array(hdulist[extension].data[column_name])
    else:
        return np.array(hdulist[extension].data[column_name], dtype=dtype)

def check_hk(hk):
    """
    Check the housekeeping data.

    Parameters:
    - hk: Array of housekeeping data.

    Returns:
    - String containing the set, length, and difference of the hk data.
    """
    return f"check_hk (set,length,diff) = {set(hk)} {len(hk)} {set(np.diff(hk))}"

def compute_datetimes(mjdrefi, times):
    """
    Compute datetime objects from MJD times.

    Parameters:
    - mjdrefi: Modified Julian Date reference.
    - times: Array of time data.

    Returns:
    - List of datetime objects corresponding to the input times.
    """    
    reftime = Time(mjdrefi, format='mjd')
    return [reftime.datetime + datetime.timedelta(seconds=float(t)) for t in times]

def zero_fill(xlist, ncycle = 1):
    """
    Fill zero values in the array with previous non-zero values.

    Parameters:
    - xlist: Input list or array to fill.
    - ncycle: Number of cycles to perform the fill operation (default is 1).

    Returns:
    - Numpy array with zeros filled.
    """
    for _ in range(ncycle):
        xlist = [xlist[i-1] if ((xlist[i] == 0) or (i == 0)) else xlist[i] for i in range(len(xlist))]
    return np.array(xlist)

def zero_fill_preval(xlist):
    """
    Fill zero values in the array with preceding non-zero values.

    Parameters:
    - xlist: Input list or array to fill.

    Returns:
    - Numpy array with zeros filled.
    """
    tmp_x = []
    first_zero_flag = True
    value_used_to_fill = -1
    for i, x in enumerate(xlist):
        if x != 0 or i == 0:
            tmp_x.append(x)
            first_zero_flag = True
        elif first_zero_flag:
            value_used_to_fill = xlist[i-1]
            tmp_x.append(value_used_to_fill)
            first_zero_flag = False
        else:
            tmp_x.append(value_used_to_fill)
    return np.array(tmp_x)

def find_gti_times(time2, ads_stt_sts_mod, ads_kf_up_mod, reftime2, is_start):
    """
    Find GTI (Good Time Interval) start and stop times.

    Parameters:
    - Detailed parameter description. 

    Returns:
    - gti_times: Times marking the start or end of GTIs.
    - gti_datetimes: Datetime objects marking the start or end of GTIs.
    """

    gti_ids = np.where(np.diff(ads_kf_up_mod) == (1 if is_start else -1))[0]

    print("gti_ids = ", gti_ids, " is_start = ", is_start)

    gti_times = []
    gti_datetimes = []

    for k, cid in enumerate(gti_ids):
        if is_start:
            if k == 0:
                continue # let's skip first, b/c it is easy way when we don't know that the first event is zero or NULL. 
            gti_times.append(time2[cid]+240.) # add 4min 
            gti_datetimes.append(reftime2.datetime + datetime.timedelta(seconds=float(gti_times[-1])))
        else:
            gti_times.append(time2[cid]+600.) # add 10min 
            gti_datetimes.append(reftime2.datetime + datetime.timedelta(seconds=float(gti_times[-1])))

    return gti_times, gti_datetimes

def save_to_file(filename, times, datetimes):  
    print(f"..... {filename} is created.")
    with open(filename, 'w') as f:
        for t, dt in zip(times, datetimes):
            print("     ", t, dt)
            f.write(f"{t}, {dt}\n")

def plot_figures(filename, datetime1, cont_mode_mod, cont_sub_mode_mod, datetime2, ads_stt_sts_mod, ads_kf_up_mod, gti_start_datetime, gti_end_datetime):
    """
    Plot figures displaying various data over time.

    Parameters:
    - filename: Name of the data file being plotted.
    - datetime1, datetime2: Arrays of datetime objects for x-axis.
    - cont_mode_mod, cont_sub_mode_mod, ads_stt_sts_mod, ads_kf_up_mod: Data arrays for y-axis.
    - gti_start_datetime, gti_end_datetime: GTI start and end times to be marked on the plot.

    Generates and shows the plot, and saves it as a PNG file.
    """
    F = plt.figure(figsize=(12,6))
    ax = plt.subplot(1,1,1)
    plt.title(filename + f" NCYCLES for zero padding is {NCYCLES}")
    plt.plot(datetime1, cont_mode_mod,           "r.", label="ACPA_CONT_MODE (mod)")
    plt.plot(datetime1, cont_sub_mode_mod + 0.1, "g.", label="ACPA_CONT_SUB_MODE (mod)")
    plt.plot(datetime2, ads_stt_sts_mod + 0.2,   "b.", label="ACPA_ADS_STT_STS (mod)")
    plt.plot(datetime2, ads_kf_up_mod + 0.3,     "m.", label="ACPA_ADS_KF_UP (mod)")
    plt.xlabel("Time")
    plt.ylabel("Raw values (slightly shifted)")

    for gti_st in gti_start_datetime:
        ax.axvline(gti_st, ymin=0.1, ymax=0.9, c="c", ls="--", alpha=0.7, label="GTI START")
    for gti_st in gti_end_datetime:
        ax.axvline(gti_st, ymin=0.1, ymax=0.9, c="m", ls="--", alpha=0.7, label="GTI STOP")
    
    plt.legend(loc='center left', bbox_to_anchor=(1., .5))
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG)
    print(f"..... {OUTPUT_PNG} is created.")
    plt.show()

def main(FILENAME, OUTPUT_PNG):
    """
    Main execution logic for processing and plotting FITS file data.
    
    - Opens the specified FITS file.
    - Processes data from two extensions in the file.
    - Finds GTI start and stop times.
    - Saves GTI times to specified output files.
    - Plots various data and GTI times, saving the plot to a PNG file.
    """    
    with fits.open(FILENAME) as hdulist:
        # First extension processing
        extension = "HK_ACPA_HK_NOM"
        acpa_cont_mode = load_hk_data(hdulist, extension, "ACPA_CONT_MODE", dtype=np.int64)
        print("ACPA_CONT_MODE", check_hk(acpa_cont_mode))
        acpa_cont_sub_mode = load_hk_data(hdulist, extension, "ACPA_CONT_SUB_MODE", dtype=np.int64)
        print("ACPA_CONT_SUB_MODE", check_hk(acpa_cont_sub_mode))        
        time1 = load_hk_data(hdulist, extension, "TIME")
        datetime1 = compute_datetimes(hdulist[extension].header["MJDREFI"], time1)
        reftime1 = Time(hdulist[extension].header["MJDREFI"], format='mjd')

        # Second extension processing
        extension = "HK_ACPA_AOCS_HK_SHD_1HZ_1"
        acpa_ads_stt_sts = load_hk_data(hdulist, extension, "ACPA_ADS_STT_STS", dtype=np.int64)
        print("ACPA_ADS_STT_STS", check_hk(acpa_ads_stt_sts))        
        acpa_ads_kf_up = load_hk_data(hdulist, extension, "ACPA_ADS_KF_UP", dtype=np.int64)
        print("ACPA_ADS_KF_UP", check_hk(acpa_ads_kf_up))        
        time2 = load_hk_data(hdulist, extension, "TIME")
        datetime2 = compute_datetimes(hdulist[extension].header["MJDREFI"], time2)
        reftime2 = Time(hdulist[extension].header["MJDREFI"], format='mjd')

        # Fill zeros with the previous values
        acpa_cont_mode_mod = zero_fill_preval(acpa_cont_mode)
        acpa_cont_sub_mode_mod = zero_fill_preval(acpa_cont_sub_mode)

        # Fill zeros by NCYCYLES
        acpa_ads_stt_sts_mod = zero_fill(acpa_ads_stt_sts, NCYCLES)
        acpa_ads_kf_up_mod = zero_fill(acpa_ads_kf_up, NCYCLES)

        # Calculate GTI
        gti_start_time, gti_start_datetime = find_gti_times(time2, acpa_ads_stt_sts_mod, acpa_ads_kf_up_mod, reftime2, True)
        gti_end_time, gti_end_datetime     = find_gti_times(time2, acpa_ads_stt_sts_mod, acpa_ads_kf_up_mod, reftime2, False)

        # Save files
        save_to_file(GTI_START_OUTPUT, gti_start_time, gti_start_datetime)
        save_to_file(GTI_STOP_OUTPUT, gti_end_time, gti_end_datetime)

        # Plot results
        plot_figures(FILENAME, datetime1, acpa_cont_mode_mod, acpa_cont_sub_mode_mod, datetime2, acpa_ads_stt_sts_mod, acpa_ads_kf_up_mod, gti_start_datetime, gti_end_datetime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process FITS file data.')
    parser.add_argument('filename', type=str, help='Path to the FITS file.')
    parser.add_argument('--output_png', type=str, default="gengti.png", help='Path for the output PNG file.')

    args = parser.parse_args()

    main(args.filename, args.output_png)
