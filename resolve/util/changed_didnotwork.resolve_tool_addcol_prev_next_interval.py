#!/usr/bin/env python

import argparse
import numpy as np
from astropy.io import fits

def calc_trigtime(WFRB_WRITE_LP, WFRB_SAMPLE_CNT, TRIG_LP):
    """
    Calculate SAMPLECNTTRIG_WO_VERNIER from WFRB/trigger counters.

    This function emulates the behavior of a 32-bit *unsigned* counter
    implemented in an embedded OS / firmware environment.

    Notes
    -----
    - All intermediate calculations are performed using Python's built-in
      `int` type (arbitrary precision) to avoid overflow/underflow errors.
    - The final result is explicitly masked to 32 bits to reproduce the
      wrap-around behavior of a uint32 counter.
    - The returned value is cast to `np.uint32`, which is suitable for
      writing into FITS uint32 columns.
    """

    # Convert inputs to plain Python integers.
    # This avoids accidental overflow when inputs originate from uint32 arrays.
    wlp = int(WFRB_WRITE_LP)
    wsp = int(WFRB_SAMPLE_CNT)
    tlp = int(TRIG_LP)

    # Compute lap difference using the upper 6 bits (bit[23:18]).
    # The '& 0x3f' enforces modulo-64 behavior, consistent with firmware logic.
    deltalap = (((wlp >> 18) & 0x3f) - ((tlp >> 18) & 0x3f)) & 0x3f

    # In the firmware convention, a value of 63 represents -1 (one lap backward).
    if deltalap == 63:
        deltalap = -1

    # Perform the counter calculation using Python int.
    # This step may temporarily produce negative values, which is acceptable here.
    raw = wsp - deltalap * 0x40000 + (tlp & 0x3ffff)

    # Explicitly apply 32-bit unsigned wrap-around
    # to emulate the behavior of a uint32 register.
    SAMPLECNTTRIG_WO_VERNIER = raw & 0xffffffff

    # Return as uint32 for compatibility with FITS columns
    return np.uint32(SAMPLECNTTRIG_WO_VERNIER)

# def calc_trigtime(WFRB_WRITE_LP, WFRB_SAMPLE_CNT, TRIG_LP):
#     """
#     Calculate the trigger time (SAMPLECNTTRIG_WO_VERNIER) based on the input parameters.
#     """
#     deltalap = (((WFRB_WRITE_LP >> 18) & 0x3f) - ((TRIG_LP >> 18) & 0x3f)) & 0x3f
#     if deltalap == 63:
#         deltalap = -1
#     SAMPLECNTTRIG_WO_VERNIER = ((WFRB_SAMPLE_CNT - deltalap * 0x40000) + (TRIG_LP & 0x3ffff)) & 0xffffffff
#     return SAMPLECNTTRIG_WO_VERNIER

def compute_diff_with_overflow(counter_list, bit_width, reverse=False):
    """
    Compute the differences between consecutive elements in `counter_list`,
    taking into account an N-bit unsigned counter overflow.

    Parameters
    ----------
    counter_list : array-like
        Sequence of counter values. Typically an array of uint32.
    bit_width : int
        Bit width of the hardware counter (e.g., 24 for a 24-bit counter).
    reverse : bool, optional
        If True, compute differences in the "reverse" direction, i.e.,
        diff = prev - curr (used for NEXT_INTERVAL with reversed arrays).

    Returns
    -------
    diffs : numpy.ndarray (dtype=uint32)
        Array of positive differences with overflow correction applied.
    """
    max_value = (1 << bit_width) - 1
    diffs = []
    minus_counter = 0

    for i in range(1, len(counter_list)):
        # Cast to Python int to avoid overflow/underflow in numpy uint32 arithmetic
        curr = int(counter_list[i])
        prev = int(counter_list[i - 1])

        if reverse:
            # For reversed arrays, we want prev - curr
            diff = prev - curr
        else:
            diff = curr - prev

        # If negative, emulate unsigned wrap-around
        if diff < 0:
            diff += max_value + 1

        # If still negative here, something is inconsistent with the assumptions
        if diff < 0:
            minus_counter += 1

        diffs.append(diff)

    print(f"debug: def compute_diff_with_overflow, minus_counter = {minus_counter}")
    # Use uint32 so that the result is consistent with hardware / FITS column types
    return np.array(diffs, dtype=np.uint32)


# def compute_diff_with_overflow(counter_list, bit_width, reverse=False):
#     """
#     Compute the differences between consecutive elements in counter_list, considering overflow.
#     """
#     max_value = (1 << bit_width) - 1
#     diffs = []
#     minus_counter = 0  
#     for i in range(1, len(counter_list)):
#         if reverse:
#             diff = -1 * (counter_list[i] - counter_list[i - 1])
#         else:
#             diff = counter_list[i] - counter_list[i - 1]
#         if diff < 0:
#             diff += max_value + 1
#         if diff < 0: # something wrong when diff + (max_value + 1) < 0
#             minus_counter +=1 
#         diffs.append(diff)
#     print(f"debug: def compute_diff_with_overflow, minus_counter = {minus_counter}")
#     return np.array(diffs)

def update_intervals(fits_file, output_file=None):
    """
    Update the PREV_INTERVAL and NEXT_INTERVAL columns in the FITS file.
    """
    with fits.open(fits_file, mode='update' if output_file is None else 'readonly') as hdul:
        data = hdul[1].data
        pixel_numbers = np.arange(36)

        for pixel in pixel_numbers:
            # Get indices of the current pixel
            pixel_indices = np.where(data['PIXEL'] == pixel)[0]
            WFRB_WRITE_LP = data['WFRB_WRITE_LP'][pixel_indices]
            WFRB_SAMPLE_CNT = data['WFRB_SAMPLE_CNT'][pixel_indices]
            TRIG_LP = data['TRIG_LP'][pixel_indices]

            # Calculate SAMPLECNTTRIG_WO_VERNIER for each event
            SAMPLECNTTRIG_WO_VERNIER = [calc_trigtime(wlp, wsp, tlp) for wlp, wsp, tlp in zip(WFRB_WRITE_LP, WFRB_SAMPLE_CNT, TRIG_LP)]

            # Compute previous and next intervals considering overflow
            prev_intervals = compute_diff_with_overflow(SAMPLECNTTRIG_WO_VERNIER, 24)
            next_intervals = compute_diff_with_overflow(SAMPLECNTTRIG_WO_VERNIER[::-1], 24, reverse=True)[::-1]

            # Debug print to check intervals
            print(f"Pixel: {pixel}, PREV_INTERVAL: {prev_intervals[:10]}")
            print(prev_intervals[:3])
            print(next_intervals[:3])
            print(f"Pixel: {pixel}, NEXT_INTERVAL: {next_intervals[:10]}")
            zcut = np.where(prev_intervals==0)[0]
            if len(zcut) > 0:
                print(f"Warning!! zero found in Pixel{pixel}, PREV[zcut]: {prev_intervals[zcut]}, TRIG_LP[zcut] {TRIG_LP[zcut]}")
            zcut = np.where(next_intervals==0)[0]
            if len(zcut) > 0:
                print(f"Warning!! zero found in Pixel{pixel}, NEXT[zcut]: {next_intervals[zcut]}, TRIG_LP[zcut] {TRIG_LP[zcut]}")

            # Update data arrays using direct indexing
            if len(prev_intervals) > 0:
                data['PREV_INTERVAL'][pixel_indices[1:]] = prev_intervals
            if len(next_intervals) > 0:
                data['NEXT_INTERVAL'][pixel_indices[:-1]] = next_intervals

            # Verify that the data array is updated
            print(f"Updated PREV_INTERVAL for pixel {pixel}: {data['PREV_INTERVAL'][pixel_indices]}")
            print(f"Updated NEXT_INTERVAL for pixel {pixel}: {data['NEXT_INTERVAL'][pixel_indices]}")

        if output_file:
            hdul.writeto(output_file, overwrite=True)
            print(f"{output_file} is overwritten.")
        else:
            hdul.flush()
            print(f"{fits_file} is updated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='This program adds PREV_INTERVAL and NEXT_INTERVAL to a FITS file.',
      epilog='''
        Example 1) Overwrite the original FITS file:
        resolve_tool_addcol_prev_next_interval.py xa000114000rsl_p0px1000_uf.evt 
        Example 2) Create a new file:
        resolve_tool_addcol_prev_next_interval.py xa000114000rsl_p0px1000_uf.evt -o xa000114000rsl_p0px1000_uf_prevnext.evt
      ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("fits_file", type=str, help="Path to the input FITS file")
    parser.add_argument("--output_file", "-o", type=str, help="Path to the output FITS file. If not specified, the input file will be updated", default=None)

    args = parser.parse_args()

    update_intervals(args.fits_file, args.output_file)
