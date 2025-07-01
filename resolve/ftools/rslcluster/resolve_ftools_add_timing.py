#!/usr/bin/env python

import os
import argparse
import numpy as np
from astropy.io import fits

def identify_timing_min(events, events_not, debug=False):
    time1 = events["TIME"]
    time2 = events_not["TIME"]
    pixel2 = events_not["PIXEL"]
    loresph2 = events_not["LO_RES_PH"]

    sort_idx = np.argsort(time2)
    time2_sorted = time2[sort_idx]
    pixel2_sorted = pixel2[sort_idx]
    loresph2_sorted = loresph2[sort_idx]

    indices = np.searchsorted(time2_sorted, time1)

    # 安全な t2A, t2B の生成
    t2A = np.full_like(time1, np.nan, dtype=np.float64)
    t2B = np.full_like(time1, np.nan, dtype=np.float64)

    valid_before = indices > 0
    valid_after = indices < len(time2_sorted)

    t2A[valid_before] = time2_sorted[indices[valid_before] - 1]
    t2B[valid_after] = time2_sorted[indices[valid_after]]

    delta_before = -1 * (time1 - t2A)
    delta_after  = -1 * (t2B - time1)
#    delta_before = time1 - t2A
#    delta_after = t2B - time1


    abs_before = np.abs(delta_before)
    abs_after = np.abs(delta_after)

    use_after = (abs_after < abs_before) | np.isnan(t2A)

    mintime_sec = np.where(use_after, delta_after, -delta_before)

    # pixel of min interval 
    pixel2_before = np.full_like(time1, -1, dtype=np.int16)
    pixel2_after = np.full_like(time1, -1, dtype=np.int16)

    pixel2_before[valid_before] = pixel2_sorted[indices[valid_before] - 1]
    pixel2_after[valid_after] = pixel2_sorted[indices[valid_after]]

    mintime_pixel = np.where(use_after, pixel2_after, pixel2_before)

    # lo_res_ph of min interval 
    loresph2_before = np.full_like(time1, -1, dtype=np.int16)
    loresph2_after  = np.full_like(time1, -1, dtype=np.int16)

    loresph2_before[valid_before] = loresph2_sorted[indices[valid_before] - 1]
    loresph2_after[valid_after] = loresph2_sorted[indices[valid_after]]

    mintime_loresph = np.where(use_after, loresph2_after, loresph2_before)

    if debug:
        for i in range(min(5, len(time1))):
            print(f"[DEBUG] t1={time1[i]:.4f}, t2A={t2A[i]:.4f}, t2B={t2B[i]:.4f}, mintime={mintime_sec[i]:.4f}, pixel={mintime_pixel[i]}, lorespha={mintime_loresph[i]}")

    return mintime_sec, mintime_pixel, mintime_loresph

def identify_timing_min_withAC(events, events_not, ac_data, debug=False):
    time1 = events["TIME"]
    time2 = events_not["TIME"]
    pixel2 = events_not["PIXEL"]
    loresph2 = events_not["LO_RES_PH"]
    timeac = ac_data["TIME"]
    ac_duration = ac_data["DURATION"]
    ac_adc_sample_max = ac_data["ADC_SAMPLE_MAX"]
    ac_pi = ac_data["PI"]
    ac_pha = ac_data["PHA"]

    sort_idx = np.argsort(time2)
    time2_sorted = time2[sort_idx]
    pixel2_sorted = pixel2[sort_idx]
    loresph2_sorted = loresph2[sort_idx]

    sort_ac_idx = np.argsort(timeac)
    timeac = timeac[sort_ac_idx]
    ac_duration = ac_duration[sort_ac_idx]
    ac_adc_sample_max = ac_adc_sample_max[sort_ac_idx]
    ac_pi = ac_pi[sort_ac_idx]
    ac_pha = ac_pha[sort_ac_idx]

    indices = np.searchsorted(time2_sorted, time1)
    indices_ac = np.searchsorted(timeac, time1)

    t2A = np.full_like(time1, np.nan, dtype=np.float64)
    t2B = np.full_like(time1, np.nan, dtype=np.float64)
    t2A_ac = np.full_like(time1, np.nan, dtype=np.float64)
    t2B_ac = np.full_like(time1, np.nan, dtype=np.float64)

    valid_before = indices > 0
    valid_after = indices < len(time2_sorted)
    valid_before_ac = indices_ac > 0
    valid_after_ac = indices_ac < len(timeac)

    t2A[valid_before] = time2_sorted[indices[valid_before] - 1]
    t2B[valid_after] = time2_sorted[indices[valid_after]]
    t2A_ac[valid_before_ac] = timeac[indices_ac[valid_before_ac] - 1]
    t2B_ac[valid_after_ac] = timeac[indices_ac[valid_after_ac]]

    delta_before = -1 * (time1 - t2A)
    delta_after  = -1 * (t2B - time1)

    delta_before_ac = -1 * (time1 - t2A_ac)
    delta_after_ac  = -1 * (t2B_ac - time1)


    abs_before = np.abs(delta_before)
    abs_after = np.abs(delta_after)

    abs_before_ac = np.abs(delta_before_ac)
    abs_after_ac = np.abs(delta_after_ac)

    use_after = (abs_after < abs_before) | np.isnan(t2A)
    use_after_ac = (abs_after_ac < abs_before_ac) | np.isnan(t2A_ac)

    mintime_sec = np.where(use_after, delta_after, -delta_before)
    mintime_ac_sec = np.where(use_after_ac, delta_after_ac, -delta_before_ac)

    # ac of duration  
    ac_duration_before = np.full_like(time1, -1, dtype=np.int16)
    ac_duration_after = np.full_like(time1, -1, dtype=np.int16)
    ac_duration_before[valid_before_ac] = ac_duration[indices_ac[valid_before_ac] - 1]
    ac_duration_after[valid_after_ac] = ac_duration[indices_ac[valid_after_ac]]
    mintime_ac_duration = np.where(use_after_ac, ac_duration_after, ac_duration_before)

    # ac of adc_sample_max  
    ac_adc_sample_max_before = np.full_like(time1, -1, dtype=np.int16)
    ac_adc_sample_max_after = np.full_like(time1, -1, dtype=np.int16)
    ac_adc_sample_max_before[valid_before_ac] = ac_adc_sample_max[indices_ac[valid_before_ac] - 1]
    ac_adc_sample_max_after[valid_after_ac] = ac_adc_sample_max[indices_ac[valid_after_ac]]
    mintime_ac_adc_sample_max = np.where(use_after_ac, ac_adc_sample_max_after, ac_adc_sample_max_before)

    # ac of pi  
    ac_pi_before = np.full_like(time1, -1, dtype=np.int16)
    ac_pi_after = np.full_like(time1, -1, dtype=np.int16)
    ac_pi_before[valid_before_ac] = ac_pi[indices_ac[valid_before_ac] - 1]
    ac_pi_after[valid_after_ac] = ac_pi[indices_ac[valid_after_ac]]
    mintime_ac_pi = np.where(use_after_ac, ac_pi_after, ac_pi_before)

    # ac of pha  
    ac_pha_before = np.full_like(time1, -1, dtype=np.int16)
    ac_pha_after = np.full_like(time1, -1, dtype=np.int16)
    ac_pha_before[valid_before_ac] = ac_pha[indices_ac[valid_before_ac] - 1]
    ac_pha_after[valid_after_ac] = ac_pha[indices_ac[valid_after_ac]]
    mintime_ac_pha = np.where(use_after_ac, ac_pha_after, ac_pha_before)

    # pixel of min interval 
    pixel2_before = np.full_like(time1, -1, dtype=np.int16)
    pixel2_after = np.full_like(time1, -1, dtype=np.int16)

    pixel2_before[valid_before] = pixel2_sorted[indices[valid_before] - 1]
    pixel2_after[valid_after] = pixel2_sorted[indices[valid_after]]

    mintime_pixel = np.where(use_after, pixel2_after, pixel2_before)

    # lo_res_ph of min interval 
    loresph2_before = np.full_like(time1, -1, dtype=np.int16)
    loresph2_after  = np.full_like(time1, -1, dtype=np.int16)

    loresph2_before[valid_before] = loresph2_sorted[indices[valid_before] - 1]
    loresph2_after[valid_after] = loresph2_sorted[indices[valid_after]]

    mintime_loresph = np.where(use_after, loresph2_after, loresph2_before)

    if debug:
        for i in range(min(5, len(time1))):
            print(f"[DEBUG] t1={time1[i]:.4f}, t2A={t2A[i]:.4f}, t2B={t2B[i]:.4f}, mintime={mintime_sec[i]:.4f}, pixel={mintime_pixel[i]}, lorespha={mintime_loresph[i]} mintime_ac_sec={mintime_ac_sec[i]:.4f}")

    return mintime_sec, mintime_pixel, mintime_loresph, mintime_ac_sec, mintime_ac_duration, mintime_ac_adc_sample_max, mintime_ac_pi, mintime_ac_pha


def process_all_pixels(data, pixel_list, debug=False):
    n_rows = len(data)
    mintime_sec_total = np.full(n_rows, np.nan, dtype=np.float64)
    mintime_pixel_total = np.full(n_rows, -1, dtype=np.int16)
    mintime_loresph_total = np.full(n_rows, -1, dtype=np.int16)

    for pixel in pixel_list:
        mask = data["PIXEL"] == pixel
        mintime_sec, mintime_pixel, mintime_loresph = identify_timing_min(data[mask], data[~mask], debug=debug)
        mintime_sec_total[mask] = mintime_sec
        mintime_pixel_total[mask] = mintime_pixel
        mintime_loresph_total[mask] = mintime_loresph

        if debug:
            print(f"[INFO] Processed pixel {pixel}, {np.sum(mask)} events")

    return mintime_sec_total, mintime_pixel_total, mintime_loresph_total

def process_all_pixels_withAC(data, pixel_list, acfile, debug=False):
    n_rows = len(data)
    mintime_sec_total = np.full(n_rows, np.nan, dtype=np.float64)
    mintime_pixel_total = np.full(n_rows, -1, dtype=np.int16)
    mintime_loresph_total = np.full(n_rows, -1, dtype=np.int16)
    mintime_ac_sec_total = np.full(n_rows, np.nan, dtype=np.float64)

    mintime_ac_duration_total = np.full(n_rows, -1, dtype=np.int16)
    mintime_ac_adc_sample_max_total = np.full(n_rows, -1, dtype=np.int16)
    mintime_ac_pi_total = np.full(n_rows, -1, dtype=np.int16)
    mintime_ac_pha_total = np.full(n_rows, -1, dtype=np.int16)

    with fits.open(acfile) as hdul:
        ac_data_all = hdul[1].data
        ac_header = hdul[1].header
        ac_cols = hdul[1].columns
        mask = (ac_data_all["PSP_ID"] == 1) & (ac_data_all["AC_ITYPE"] == 0)
        ac_data = ac_data_all[mask]
        print(f"..... debug : len(ac_data_all)={len(ac_data_all)} to len(ac_data)={len(ac_data)} ")

        for pixel in pixel_list:
            mask = data["PIXEL"] == pixel

            mintime_sec, mintime_pixel, mintime_loresph, mintime_ac_sec, \
                mintime_ac_duration, mintime_ac_adc_sample_max, mintime_ac_pi, mintime_ac_pha \
            = identify_timing_min_withAC(data[mask], data[~mask], ac_data, debug=debug)

            mintime_sec_total[mask] = mintime_sec
            mintime_pixel_total[mask] = mintime_pixel
            mintime_loresph_total[mask] = mintime_loresph
            mintime_ac_sec_total[mask] = mintime_ac_sec

            mintime_ac_duration_total[mask] = mintime_ac_duration
            mintime_ac_adc_sample_max_total[mask] = mintime_ac_adc_sample_max
            mintime_ac_pi_total[mask] = mintime_ac_pi
            mintime_ac_pha_total[mask] = mintime_ac_pha

            if debug:
                print(f"[INFO] Processed pixel {pixel}, {np.sum(mask)} events")

    return mintime_sec_total, mintime_pixel_total, mintime_loresph_total, \
             mintime_ac_sec_total, mintime_ac_duration_total, mintime_ac_adc_sample_max_total, mintime_ac_pi_total, mintime_ac_pha_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_fits", help="Input FITS file")
    parser.add_argument("--outname", "-o", default="mintime_", help="Output FITS file prefix")
    parser.add_argument("--acfile", "-a", default=None, help="Input Anti-co file")
    parser.add_argument("--usepixels", "-p", default=",".join(map(str, range(36))), help="Comma-separated pixel numbers")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    outname = args.outname + os.path.basename(args.input_fits)
    pixel_list = list(map(int, args.usepixels.split(",")))

    if args.acfile == None:

        with fits.open(args.input_fits) as hdul:
            data = hdul[1].data
            header = hdul[1].header
            cols = hdul[1].columns

            mintime_sec, mintime_pixel, mintime_loresph = process_all_pixels(data, pixel_list, debug=args.debug)

            # 上書き防止
            col_names_to_remove = {"MINTIME_SEC", "MINTIME_PIXEL",  "MINTIME_LORESPH"}
            new_coldefs = fits.ColDefs([col for col in cols if col.name not in col_names_to_remove])

            # 新しいカラムを追加
            new_coldefs += fits.ColDefs([
                fits.Column(name="MINTIME_SEC", format="D", array=mintime_sec),
                fits.Column(name="MINTIME_PIXEL", format="1I", array=mintime_pixel),
                fits.Column(name="MINTIME_LORESPH", format="1I", array=mintime_loresph),            
            ])

            new_hdu = fits.BinTableHDU.from_columns(new_coldefs, header=header)
            hdul[1] = new_hdu
            hdul.writeto(outname, overwrite=True)
            print(f"[DONE] Saved to: {outname}")

    else:

        with fits.open(args.input_fits) as hdul:
            data = hdul[1].data
            header = hdul[1].header
            cols = hdul[1].columns

            mintime_sec, mintime_pixel, mintime_loresph, mintime_ac_sec,\
                            mintime_ac_duration, mintime_ac_adc_sample_max, mintime_ac_pi, mintime_ac_pha \
                     = process_all_pixels_withAC(data, pixel_list, args.acfile, debug=args.debug)

            # 上書き防止
            col_names_to_remove = {"MINTIME_SEC", "MINTIME_PIXEL",  "MINTIME_LORESPH"}
            new_coldefs = fits.ColDefs([col for col in cols if col.name not in col_names_to_remove])

            # 新しいカラムを追加
            new_coldefs += fits.ColDefs([
                fits.Column(name="MINTIME_SEC", format="D", array=mintime_sec),
                fits.Column(name="MINTIME_PIXEL", format="1I", array=mintime_pixel),
                fits.Column(name="MINTIME_LORESPH", format="1I", array=mintime_loresph),            
                fits.Column(name="MINTIME_AC_SEC", format="D", array=mintime_ac_sec),
                fits.Column(name="MINTIME_AC_DURATION", format="1I", array=mintime_ac_duration),
                fits.Column(name="MINTIME_AC_ADC_SAMPLE_MAX", format="1I", array=mintime_ac_adc_sample_max),
                fits.Column(name="MINTIME_AC_PI", format="1I", array=mintime_ac_pi),
                fits.Column(name="MINTIME_AC_PHA", format="1I", array=mintime_ac_pha)

            ])

            new_hdu = fits.BinTableHDU.from_columns(new_coldefs, header=header)
            hdul[1] = new_hdu
            hdul.writeto(outname, overwrite=True)
            print(f"[DONE] Saved to: {outname}")


if __name__ == "__main__":
    main()
