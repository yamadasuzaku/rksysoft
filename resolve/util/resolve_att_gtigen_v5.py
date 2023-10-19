#!/usr/bin/env python
# coding: utf-8

import sys
import datetime
import astropy.io.fits as fits
from astropy.time import Time
import numpy as np
import matplotlib.pylab as plt 
import argparse

def load_hk_data(hdulist, extension, column_name, dtype=None):
    return np.array(hdulist[extension].data[column_name], dtype=dtype)

def check_hk(hk):
    return f"check_hk (set,length,diff) = {set(hk)} {len(hk)} {set(np.diff(hk))}"

def compute_datetimes(mjdrefi, times):
    reftime = Time(mjdrefi, format='mjd')
    return [reftime.datetime + datetime.timedelta(seconds=float(t)) for t in times]

def zero_fill(xlist, ncycle=1):
    for _ in range(ncycle):
        xlist = [xlist[i-1] if (x == 0 or i == 0) else x for i, x in enumerate(xlist)]
    return np.array(xlist)

def zero_fill_preval(xlist):
    tmp_x = []
    value_used_to_fill = -1
    for i, x in enumerate(xlist):
        if x != 0 or i == 0:
            tmp_x.append(x)
            value_used_to_fill = x
        else:
            tmp_x.append(value_used_to_fill)
    return np.array(tmp_x)

def find_gti_times(time2, ads_stt_sts_mod, ads_kf_up_mod, reftime2, is_start):
    gti_ids = np.where(np.diff(ads_kf_up_mod) == (1 if is_start else -1))[0]
    gti_times = [(t + (240 if is_start else 600)) for i, t in enumerate(time2[gti_ids])]
#    gti_times = [(t + (240 if is_start else 600)) for i, t in enumerate(time2[gti_ids]) if not (is_start and i == 0)]
    gti_datetimes = [reftime2.datetime + datetime.timedelta(seconds=float(t)) for t in gti_times]
    return gti_times, gti_datetimes

def save_to_file(filename, times, datetimes):
    with open(filename, 'w') as f:
        for t, dt in zip(times, datetimes):
            f.write(f"{t}, {dt}\n")

def plot_figures(filename, datetime1, cont_mode_mod, cont_sub_mode_mod, datetime2, ads_stt_sts_mod, ads_kf_up_mod, gti_start_datetime, gti_end_datetime, ncycles, output_png):
    F = plt.figure(figsize=(12,6))
    ax = plt.subplot(1,1,1)
    plt.title(f"{filename} NCYCLES for zero padding is {ncycles}")
    plt.plot(datetime1, cont_mode_mod, "r.", label="ACPA_CONT_MODE (mod)")
    plt.plot(datetime1, cont_sub_mode_mod + 0.1, "g.", label="ACPA_CONT_SUB_MODE (mod)")
    plt.plot(datetime2, ads_stt_sts_mod + 0.2, "b.", label="ACPA_ADS_STT_STS (mod)")
    plt.plot(datetime2, ads_kf_up_mod + 0.3, "m.", label="ACPA_ADS_KF_UP (mod)")
    plt.xlabel("Time")
    plt.ylabel("Raw values (slightly shifted)")

    for gti_st in gti_start_datetime:
        ax.axvline(gti_st, ymin=0.1, ymax=0.9, c="c", ls="--", alpha=0.7, label="GTI START")
    for gti_st in gti_end_datetime:
        ax.axvline(gti_st, ymin=0.1, ymax=0.9, c="m", ls="--", alpha=0.7, label="GTI STOP")
    
    plt.legend(loc='center left', bbox_to_anchor=(1., .5))
    plt.tight_layout()
    plt.savefig(output_png)
    plt.show()

def main(filename, output_png, ncycles, gti_start_output, gti_stop_output):
    with fits.open(filename) as hdulist:
        # Process data
        extension = "HK_ACPA_HK_NOM"
        acpa_cont_mode = load_hk_data(hdulist, extension, "ACPA_CONT_MODE", dtype=np.int64)
        acpa_cont_sub_mode = load_hk_data(hdulist, extension, "ACPA_CONT_SUB_MODE", dtype=np.int64)
        time1 = load_hk_data(hdulist, extension, "TIME")
        datetime1 = compute_datetimes(hdulist[extension].header["MJDREFI"], time1)

        extension = "HK_ACPA_AOCS_HK_SHD_1HZ_1"
        acpa_ads_stt_sts = load_hk_data(hdulist, extension, "ACPA_ADS_STT_STS", dtype=np.int64)
        acpa_ads_kf_up = load_hk_data(hdulist, extension, "ACPA_ADS_KF_UP", dtype=np.int64)
        time2 = load_hk_data(hdulist, extension, "TIME")
        datetime2 = compute_datetimes(hdulist[extension].header["MJDREFI"], time2)
        reftime2 = Time(hdulist[extension].header["MJDREFI"], format='mjd')

        acpa_cont_mode_mod = zero_fill_preval(acpa_cont_mode)
        acpa_cont_sub_mode_mod = zero_fill_preval(acpa_cont_sub_mode)
        acpa_ads_stt_sts_mod = zero_fill(acpa_ads_stt_sts, ncycles)
        acpa_ads_kf_up_mod = zero_fill(acpa_ads_kf_up, ncycles)

        gti_start_time, gti_start_datetime = find_gti_times(time2, acpa_ads_stt_sts_mod, acpa_ads_kf_up_mod, reftime2, True)
        gti_end_time, gti_end_datetime = find_gti_times(time2, acpa_ads_stt_sts_mod, acpa_ads_kf_up_mod, reftime2, False)

        save_to_file(gti_start_output, gti_start_time, gti_start_datetime)
        save_to_file(gti_stop_output, gti_end_time, gti_end_datetime)

        plot_figures(filename, datetime1, acpa_cont_mode_mod, acpa_cont_sub_mode_mod, datetime2, acpa_ads_stt_sts_mod, acpa_ads_kf_up_mod, gti_start_datetime, gti_end_datetime, ncycles, output_png)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process FITS file data.')
    parser.add_argument('filename', type=str, help='Path to the FITS file.')
    parser.add_argument('--output_png', type=str, default="gengti.png", help='Path for the output PNG file.')
    parser.add_argument('--ncycles', type=int, default=100, help='Number of cycles to fill zero.')
    parser.add_argument('--gti_start_output', type=str, default="gti_start.txt", help='Output file for GTI start times.')
    parser.add_argument('--gti_stop_output', type=str, default="gti_stop.txt", help='Output file for GTI stop times.')

    args = parser.parse_args()

    main(args.filename, args.output_png, args.ncycles, args.gti_start_output, args.gti_stop_output)
