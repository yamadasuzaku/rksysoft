#!/usr/bin/env python 
import argparse
from astropy.io import fits
import numpy as np

def main():
    # Setup argparse
    parser = argparse.ArgumentParser(description="Process FITS file and extract +/-1 second data around a condition.")
    parser.add_argument("fname", type=str, help="File name to be processed.")
    parser.add_argument("--threshold",'-t', type=float, default=65000.,  help="Threshold for column.")
    parser.add_argument("--column", '-c', type=str, default='PHA',  help="Threshold for column.")    
    parser.add_argument("--timewindow", '-w', type=float, default=1.0,  help="Time window at the threshold")        
    parser.add_argument("--itype",'-i', type=int, default=None,  help="ITYPE (default=None)")

    args = parser.parse_args()

    fname = args.fname
    threshold = args.threshold
    column = args.column
    timewindow = args.timewindow
    itype = args.itype

    # Prepare the output filename
    outftag = fname.replace(".evt", "").replace(".gz", "")
    output_fname = f"sc_{outftag}_{column}_thre{int(threshold):d}_dt{int(timewindow):d}_cuttype{itype}.evt"

    # Load FITS data
    hdu = fits.open(args.fname)[1]
    data = hdu.data
    # Extract relevant columns
    time_array = data["TIME"]
    itype_array = data["ITYPE"]
    ref_array = data[column]

    # Condition: lo_array greater than or equal to the threshold
    if itype == None:
        condition_indices = np.where(ref_array >= threshold)[0]
    else:
        condition_indices = np.where( (ref_array >= threshold) & (itype_array == itype))[0]

    # Prepare an empty mask to keep +/- timewindow second data
    mask = np.zeros(len(time_array), dtype=bool)

    print(f"Length changes from {len(time_array)} to {len(condition_indices)}.")

    # Loop over each index where the condition is satisfied
    for idx in condition_indices:
        current_time = time_array[idx]
        # Find indices within +/-1 second
        time_condition = (time_array >= current_time - timewindow) & (time_array <= current_time + timewindow)
        mask |= time_condition  # Use a boolean OR to combine the conditions

    # Apply the mask to filter the data
    filtered_data = data[mask]

    # Save the filtered data to a new FITS file
    hdu_new = fits.BinTableHDU(filtered_data, header=hdu.header)
    hdu_new.writeto(output_fname, overwrite=True)
    print(f"Filtered data saved to {output_fname}")

if __name__ == "__main__":
    main()
