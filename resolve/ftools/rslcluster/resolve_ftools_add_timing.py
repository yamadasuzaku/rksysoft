#!/usr/bin/env python

import os
import argparse
import numpy as np
from astropy.io import fits

def identify_timing_min(events, events_not, debug=False):
    """
    For each event in `events`, find the nearest preceding and succeeding events in `events_not`,
    and compute timing and pixel-related information.

    Parameters:
    ----------
    events : FITS table
        Events of interest (e.g., for a specific pixel)
    events_not : FITS table
        Events to compare against (e.g., all other pixels except calibration pixel)
    debug : bool
        If True, print debug information for first few events

    Returns:
    -------
    mintime_sec : ndarray
        Time difference to nearest event (in seconds)
    mintime_pixel : ndarray
        Pixel ID of the nearest event
    mintime_loresph : ndarray
        LO_RES_PH value of the nearest event
    """

    # Extract relevant columns from events and events_not
    time1 = events["TIME"]              # Times of interest (e.g., for one pixel)
    time2 = events_not["TIME"]          # Times of all other events
    pixel2 = events_not["PIXEL"]        # Pixel IDs of other events
    loresph2 = events_not["LO_RES_PH"]  # LO_RES_PH of other events

    # Sort events_not by time for efficient searching
    sort_idx = np.argsort(time2)
    time2_sorted = time2[sort_idx]
    pixel2_sorted = pixel2[sort_idx]
    loresph2_sorted = loresph2[sort_idx]

    # Find insertion indices: where time1 would fit into time2_sorted
    indices = np.searchsorted(time2_sorted, time1)

    # Pre-allocate arrays for nearest times (before and after)
    t2A = np.full_like(time1, np.nan, dtype=np.float64)  # Previous event time
    t2B = np.full_like(time1, np.nan, dtype=np.float64)  # Next event time

    # Determine which indices are valid for accessing before/after elements
    valid_before = indices > 0
    valid_after = indices < len(time2_sorted)

    # Fill t2A and t2B arrays where valid
    t2A[valid_before] = time2_sorted[indices[valid_before] - 1]
    t2B[valid_after] = time2_sorted[indices[valid_after]]

    # Compute time deltas to previous and next events
    delta_before = -1 * (time1 - t2A)
    delta_after  = -1 * (t2B - time1)

    # Compute absolute deltas for comparison
    abs_before = np.abs(delta_before)
    abs_after = np.abs(delta_after)

    # Decide whether the after event is closer, or before is NaN
    use_after = (abs_after < abs_before) | np.isnan(t2A)

    # Select minimum time difference (negative if before, positive if after)
    mintime_sec = np.where(use_after, delta_after, -delta_before)

    # Determine the pixel ID of the nearest event
    pixel2_before = np.full_like(time1, -1, dtype=np.int16)
    pixel2_after = np.full_like(time1, -1, dtype=np.int16)
    pixel2_before[valid_before] = pixel2_sorted[indices[valid_before] - 1]
    pixel2_after[valid_after] = pixel2_sorted[indices[valid_after]]
    mintime_pixel = np.where(use_after, pixel2_after, pixel2_before)

    # Determine LO_RES_PH of the nearest event
    loresph2_before = np.full_like(time1, -1, dtype=np.int16)
    loresph2_after = np.full_like(time1, -1, dtype=np.int16)
    loresph2_before[valid_before] = loresph2_sorted[indices[valid_before] - 1]
    loresph2_after[valid_after] = loresph2_sorted[indices[valid_after]]
    mintime_loresph = np.where(use_after, loresph2_after, loresph2_before)

    # Debug output for first few entries
    if debug:
        for i in range(min(5, len(time1))):
            print(f"[DEBUG] t1={time1[i]:.4f}, t2A={t2A[i]:.4f}, t2B={t2B[i]:.4f}, "
                  f"mintime={mintime_sec[i]:.4f}, pixel={mintime_pixel[i]}, "
                  f"loresph={mintime_loresph[i]}")

    return mintime_sec, mintime_pixel, mintime_loresph


def identify_timing_min_withAC(events, events_not, ac_data, debug=False):
    """
    Extended version of identify_timing_min that also considers Anti-coincidence (AC) data.

    Parameters:
    ----------
    events : FITS table
        Events of interest (e.g., for a specific pixel)
    events_not : FITS table
        Events to compare against (e.g., all other pixels except calibration pixel)
    ac_data : FITS table
        Anti-coincidence data table
    debug : bool
        If True, print debug information for first few events

    Returns:
    -------
    Tuple of arrays with timing and AC-related information
    """

    # Extract relevant columns from events and events_not
    time1 = events["TIME"]              # Times of interest (for one pixel)
    time2 = events_not["TIME"]          # Times of other events
    pixel2 = events_not["PIXEL"]        # Pixel IDs of other events
    loresph2 = events_not["LO_RES_PH"]  # LO_RES_PH of other events

    # Extract relevant columns from AC data
    timeac = ac_data["TIME"]
    ac_duration = ac_data["DURATION"]
    ac_adc_sample_max = ac_data["ADC_SAMPLE_MAX"]
    ac_pi = ac_data["PI"]
    ac_pha = ac_data["PHA"]

    # Sort events_not and AC data by time for efficient searching
    sort_idx = np.argsort(time2)
    time2_sorted = time2[sort_idx]
    pixel2_sorted = pixel2[sort_idx]
    loresph2_sorted = loresph2[sort_idx]

    sort_ac_idx = np.argsort(timeac)
    timeac_sorted = timeac[sort_ac_idx]
    ac_duration_sorted = ac_duration[sort_ac_idx]
    ac_adc_sample_max_sorted = ac_adc_sample_max[sort_ac_idx]
    ac_pi_sorted = ac_pi[sort_ac_idx]
    ac_pha_sorted = ac_pha[sort_ac_idx]

    # Find insertion indices for events_not and AC data
    indices = np.searchsorted(time2_sorted, time1)
    indices_ac = np.searchsorted(timeac_sorted, time1)

    # Pre-allocate arrays for nearest times
    t2A = np.full_like(time1, np.nan, dtype=np.float64)
    t2B = np.full_like(time1, np.nan, dtype=np.float64)
    t2A_ac = np.full_like(time1, np.nan, dtype=np.float64)
    t2B_ac = np.full_like(time1, np.nan, dtype=np.float64)

    # Determine which indices are valid for accessing before/after elements
    valid_before = indices > 0
    valid_after = indices < len(time2_sorted)
    valid_before_ac = indices_ac > 0
    valid_after_ac = indices_ac < len(timeac_sorted)

    # Fill t2A and t2B arrays for events_not
    t2A[valid_before] = time2_sorted[indices[valid_before] - 1]
    t2B[valid_after] = time2_sorted[indices[valid_after]]

    # Fill t2A_ac and t2B_ac arrays for AC data
    t2A_ac[valid_before_ac] = timeac_sorted[indices_ac[valid_before_ac] - 1]
    t2B_ac[valid_after_ac] = timeac_sorted[indices_ac[valid_after_ac]]

    # Compute time deltas to previous and next events
    delta_before = -1 * (time1 - t2A)
    delta_after  = -1 * (t2B - time1)
    delta_before_ac = -1 * (time1 - t2A_ac)
    delta_after_ac  = -1 * (t2B_ac - time1)

    # Compute absolute deltas for comparison
    abs_before = np.abs(delta_before)
    abs_after = np.abs(delta_after)
    abs_before_ac = np.abs(delta_before_ac)
    abs_after_ac = np.abs(delta_after_ac)

    # Decide whether after event/AC is closer, or before is NaN
    use_after = (abs_after < abs_before) | np.isnan(t2A)
    use_after_ac = (abs_after_ac < abs_before_ac) | np.isnan(t2A_ac)

    # Select minimum time difference
    mintime_sec = np.where(use_after, delta_after, -delta_before)
    mintime_ac_sec = np.where(use_after_ac, delta_after_ac, -delta_before_ac)

    # Retrieve additional AC-related parameters for nearest events
    def retrieve_ac_parameter(param_sorted, valid_before_ac, valid_after_ac):
        before = np.full_like(time1, -1, dtype=np.int16)
        after = np.full_like(time1, -1, dtype=np.int16)
        before[valid_before_ac] = param_sorted[indices_ac[valid_before_ac] - 1]
        after[valid_after_ac] = param_sorted[indices_ac[valid_after_ac]]
        return np.where(use_after_ac, after, before)

    mintime_ac_duration = retrieve_ac_parameter(ac_duration_sorted, valid_before_ac, valid_after_ac)
    mintime_ac_adc_sample_max = retrieve_ac_parameter(ac_adc_sample_max_sorted, valid_before_ac, valid_after_ac)
    mintime_ac_pi = retrieve_ac_parameter(ac_pi_sorted, valid_before_ac, valid_after_ac)
    mintime_ac_pha = retrieve_ac_parameter(ac_pha_sorted, valid_before_ac, valid_after_ac)

    # Determine pixel ID and LO_RES_PH of nearest event (like in identify_timing_min)
    pixel2_before = np.full_like(time1, -1, dtype=np.int16)
    pixel2_after = np.full_like(time1, -1, dtype=np.int16)
    pixel2_before[valid_before] = pixel2_sorted[indices[valid_before] - 1]
    pixel2_after[valid_after] = pixel2_sorted[indices[valid_after]]
    mintime_pixel = np.where(use_after, pixel2_after, pixel2_before)

    loresph2_before = np.full_like(time1, -1, dtype=np.int16)
    loresph2_after  = np.full_like(time1, -1, dtype=np.int16)
    loresph2_before[valid_before] = loresph2_sorted[indices[valid_before] - 1]
    loresph2_after[valid_after] = loresph2_sorted[indices[valid_after]]
    mintime_loresph = np.where(use_after, loresph2_after, loresph2_before)

    # Debug output for first few entries
    if debug:
        for i in range(min(5, len(time1))):
            print(f"[DEBUG] t1={time1[i]:.4f}, t2A={t2A[i]:.4f}, t2B={t2B[i]:.4f}, "
                  f"mintime={mintime_sec[i]:.4f}, pixel={mintime_pixel[i]}, "
                  f"loresph={mintime_loresph[i]}, mintime_ac={mintime_ac_sec[i]:.4f}")

    return (
        mintime_sec, mintime_pixel, mintime_loresph,
        mintime_ac_sec, mintime_ac_duration,
        mintime_ac_adc_sample_max, mintime_ac_pi, mintime_ac_pha
    )

def process_all_pixels(data, pixel_list, debug=False):
    """
    Process all specified pixels to compute nearest event timings.

    Parameters:
    ----------
    data : FITS table
        Input event data
    pixel_list : list
        List of pixel IDs to process
    debug : bool
        If True, print debug information

    Returns:
    -------
    Tuple of arrays with timing, pixel, and LO_RES_PH information
    """
    n_rows = len(data)

    # Pre-allocate output arrays
    mintime_sec_total = np.full(n_rows, np.nan, dtype=np.float64)
    mintime_pixel_total = np.full(n_rows, -1, dtype=np.int16)
    mintime_loresph_total = np.full(n_rows, -1, dtype=np.int16)

    for pixel in pixel_list:
        # Create masks: select events for this pixel, and exclude calibration pixel (12)
        mask = data["PIXEL"] == pixel
        mask_notself_notcal = ~mask & (data["PIXEL"] != 12)

        # Compute nearest timings
        mintime_sec, mintime_pixel, mintime_loresph = identify_timing_min(
            data[mask], data[mask_notself_notcal], debug=debug
        )

        # Fill output arrays
        mintime_sec_total[mask] = mintime_sec
        mintime_pixel_total[mask] = mintime_pixel
        mintime_loresph_total[mask] = mintime_loresph

        if debug:
            print(f"[INFO] Processed pixel {pixel}, {np.sum(mask)} events")

    return mintime_sec_total, mintime_pixel_total, mintime_loresph_total


def process_all_pixels_withAC(data, pixel_list, acfile, debug=False):
    """
    Process all specified pixels to compute nearest event timings, including Anti-coincidence (AC) data.

    Parameters:
    ----------
    data : FITS table
        Input event data
    pixel_list : list
        List of pixel IDs to process
    acfile : str
        Path to the AC FITS file
    debug : bool
        If True, print debug information

    Returns:
    -------
    Tuple of arrays with timing, pixel, LO_RES_PH, and AC-related information
    """
    n_rows = len(data)

    # Pre-allocate output arrays for results
    mintime_sec_total = np.full(n_rows, np.nan, dtype=np.float64)
    mintime_pixel_total = np.full(n_rows, -1, dtype=np.int16)
    mintime_loresph_total = np.full(n_rows, -1, dtype=np.int16)
    mintime_ac_sec_total = np.full(n_rows, np.nan, dtype=np.float64)

    mintime_ac_duration_total = np.full(n_rows, -1, dtype=np.int16)
    mintime_ac_adc_sample_max_total = np.full(n_rows, -1, dtype=np.int16)
    mintime_ac_pi_total = np.full(n_rows, -1, dtype=np.int16)
    mintime_ac_pha_total = np.full(n_rows, -1, dtype=np.int16)

    # Open AC FITS file and filter relevant data
    with fits.open(acfile) as hdul:
        ac_data_all = hdul[1].data
        mask = (ac_data_all["PSP_ID"] == 1) & (ac_data_all["AC_ITYPE"] == 0)
        ac_data = ac_data_all[mask]

        print(f"[INFO] Filtered AC data: {len(ac_data_all)} → {len(ac_data)} rows")

        # Process each pixel separately
        for pixel in pixel_list:
            print(f"\n=== Processing pixel {pixel} ===")

            # Create masks: select events for this pixel, and exclude calibration pixel (12)
            mask_pixel = data["PIXEL"] == pixel
            mask_notself = ~mask_pixel
            mask_notcal = data["PIXEL"] != 12
            mask_notself_notcal = mask_notself & mask_notcal

            # Perform validation checks for debug
            if debug:
                count_a = np.sum(mask_pixel)
                count_b = np.sum(mask_notself)
                count_c = np.sum(mask_notcal)
                count_d = np.sum(mask_notself_notcal)
                total_events = len(data)

                print(f"a) PIXEL == {pixel}: {count_a}")
                print(f"b) PIXEL != {pixel}: {count_b}")
                print(f"c) PIXEL != 12: {count_c}")
                print(f"d) PIXEL != {pixel} AND PIXEL != 12: {count_d}")

                # Check 1: a + b should match total_events
                if count_a + count_b == total_events:
                    print(f"Check 1 passed: a + b == total_events ({total_events})")
                else:
                    print(f"Check 1 failed: a + b = {count_a + count_b}, expected {total_events}")

                # Check 2: d <= b because calibration pixel is excluded
                if count_d <= count_b:
                    print(f"Check 2 passed: d <= b ({count_d} <= {count_b})")
                else:
                    print(f"Check 2 failed: d > b ({count_d} > {count_b})")

            # Compute nearest timings and AC-related data
            results = identify_timing_min_withAC(
                data[mask_pixel], data[mask_notself_notcal], ac_data, debug=debug
            )
            (
                mintime_sec, mintime_pixel, mintime_loresph,
                mintime_ac_sec, mintime_ac_duration,
                mintime_ac_adc_sample_max, mintime_ac_pi, mintime_ac_pha
            ) = results

            # Store results in total arrays
            mintime_sec_total[mask_pixel] = mintime_sec
            mintime_pixel_total[mask_pixel] = mintime_pixel
            mintime_loresph_total[mask_pixel] = mintime_loresph
            mintime_ac_sec_total[mask_pixel] = mintime_ac_sec
            mintime_ac_duration_total[mask_pixel] = mintime_ac_duration
            mintime_ac_adc_sample_max_total[mask_pixel] = mintime_ac_adc_sample_max
            mintime_ac_pi_total[mask_pixel] = mintime_ac_pi
            mintime_ac_pha_total[mask_pixel] = mintime_ac_pha

            if debug:
                print(f"[INFO] Finished pixel {pixel}, {np.sum(mask_pixel)} events")

    return (
        mintime_sec_total, mintime_pixel_total, mintime_loresph_total,
        mintime_ac_sec_total, mintime_ac_duration_total,
        mintime_ac_adc_sample_max_total, mintime_ac_pi_total, mintime_ac_pha_total
    )


def main():
    """
    Main entry point: parses arguments, reads FITS file,
    computes timing information (with or without AC data),
    and writes output FITS file with additional columns.
    """
    parser = argparse.ArgumentParser(description="Compute minimum time differences between events")
    parser.add_argument("input_fits", help="Path to the input FITS file")
    parser.add_argument("--outname", "-o", default="mintime_", help="Prefix for the output FITS file")
    parser.add_argument("--acfile", "-a", default=None, help="Path to the Anti-coincidence (AC) FITS file")
    parser.add_argument("--usepixels", "-p", default=",".join(map(str, range(36))),
                        help="Comma-separated list of pixel numbers to process (default: all 0-35)")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    # Generate output filename by adding prefix to input filename
    outname = args.outname + os.path.basename(args.input_fits)

    # Parse pixel list from comma-separated string
    pixel_list = list(map(int, args.usepixels.split(",")))

    with fits.open(args.input_fits) as hdul:
        data = hdul[1].data
        header = hdul[1].header
        cols = hdul[1].columns

        if args.acfile is None:
            # Case without AC file
            mintime_sec, mintime_pixel, mintime_loresph = process_all_pixels(
                data, pixel_list, debug=args.debug
            )

            # Remove old columns if they exist
            col_names_to_remove = {"MINTIME_SEC", "MINTIME_PIXEL", "MINTIME_LORESPH"}
            new_coldefs = fits.ColDefs([col for col in cols if col.name not in col_names_to_remove])

            # Add new columns
            new_coldefs += fits.ColDefs([
                fits.Column(name="MINTIME_SEC", format="D", array=mintime_sec),
                fits.Column(name="MINTIME_PIXEL", format="1I", array=mintime_pixel),
                fits.Column(name="MINTIME_LORESPH", format="1I", array=mintime_loresph),
            ])

        else:
            # Case with AC file
            results = process_all_pixels_withAC(data, pixel_list, args.acfile, debug=args.debug)
            (
                mintime_sec, mintime_pixel, mintime_loresph,
                mintime_ac_sec, mintime_ac_duration,
                mintime_ac_adc_sample_max, mintime_ac_pi, mintime_ac_pha
            ) = results

            # Remove old columns if they exist
            col_names_to_remove = {"MINTIME_SEC", "MINTIME_PIXEL", "MINTIME_LORESPH"}
            new_coldefs = fits.ColDefs([col for col in cols if col.name not in col_names_to_remove])

            # Add new columns including AC-related ones
            new_coldefs += fits.ColDefs([
                fits.Column(name="MINTIME_SEC", format="D", array=mintime_sec),
                fits.Column(name="MINTIME_PIXEL", format="1I", array=mintime_pixel),
                fits.Column(name="MINTIME_LORESPH", format="1I", array=mintime_loresph),
                fits.Column(name="MINTIME_AC_SEC", format="D", array=mintime_ac_sec),
                fits.Column(name="MINTIME_AC_DURATION", format="1I", array=mintime_ac_duration),
                fits.Column(name="MINTIME_AC_ADC_SAMPLE_MAX", format="1I", array=mintime_ac_adc_sample_max),
                fits.Column(name="MINTIME_AC_PI", format="1I", array=mintime_ac_pi),
                fits.Column(name="MINTIME_AC_PHA", format="1I", array=mintime_ac_pha),
            ])

        # Create new HDU and write output FITS file
        new_hdu = fits.BinTableHDU.from_columns(new_coldefs, header=header)
        hdul[1] = new_hdu
        hdul.writeto(outname, overwrite=True)
        print(f"[DONE] Saved to: {outname}")


if __name__ == "__main__":
    main()
