#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)
import numpy as np
from astropy.io import fits
from astropy.time import Time
import datetime
import random

# global variables
template_length_MR=219
# Define global variables
MJD_REFERENCE_DAY = 58484
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')  # Reference time in Modified Julian Day

def compute_datetimes(mjdrefi, times):
    reftime = Time(mjdrefi, format='mjd')
    return [reftime.datetime + datetime.timedelta(seconds=float(t)) for t in times]

def is_cluster_start(event, event_n1, event_n2, prev_event, key1 = "NEXT_INTERVAL", para1=template_length_MR, debug=True):
    """Determine the condition for starting a new cluster."""
    # Logic for cluster start condition
    status =  (event[key1] < para1) & (event_n1[key1] < para1) & (event_n2[key1] < para1)
    if debug: 
        print(f"{status} = ({event[key1]} < {para1}) & ({event_n1[key1]} < {para1}) & ({event_n2[key1]} < {para1})")
    return status  # Return True/False based on the condition

def is_cluster_continue(event, current_cluster, key1 = "NEXT_INTERVAL", para1=120, debug=True):
    """Check if the current event continues the existing cluster."""
    # Logic for cluster continuation condition
    status = event[key1] < para1
    if debug:
        print(f"{status} = {event[key1]} < {para1}")
    return status  # Return True/False based on the condition

def is_cluster_end(event, current_cluster, para1=2, debug=True):
    """Determine if the cluster has reached its end."""
    # Logic for cluster end condition
    status = len(current_cluster) >= para1
    if debug:
        print(f"{status} = {len(current_cluster)} >= {para1}")
    return status  # Return True/False based on the condition

def confirm_cluster(current_cluster, icluster, usepixel, key1="ITYPE", debug=True):
    """Confirm and finalize the cluster, applying any necessary checks or filters."""
    # Logic for confirming or saving the finalized cluster
    if debug:
        plot_onecluster(current_cluster, icluster, usepixel)
        for i, mcluster in enumerate(current_cluster):
            print(i, mcluster[key1])
    status = True
    return status  # Return good or bad if any

def plot_onecluster(cluster, icluster, usepixel, keys=["LO_RES_PH", "DERIV_MAX", "ITYPE", "RISE_TIME", "PREV_INTERVAL", "NEXT_INTERVAL"], output_dir="."):
    """
    Visualize the data for a single cluster and save the plot to a PNG file.

    Parameters:
        cluster (list): The cluster data (list of dictionaries).
        keys (list): Keys to extract data from each event.
        output_dir (str): Directory to save the output PNG file.
    """
    # Extract data from the cluster
    data = {key: [event[key] for event in cluster] for key in keys}
    times = [event["TIME"] for event in cluster]
    imember_datetime = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=time) for time in times])
    imember_cluster = np.arange(len(cluster)) + 1

    # Print cluster information to standard output
    print("Cluster Information:")
    for key in keys:
        print(f"{key}: {data[key]}")
    print(f"TIME: {times}")
    print(f"DATETIME: {imember_datetime}")

    # Determine the number of rows and columns for subplots
    num_keys = len(keys)
    num_cols = 2  # Fixed number of columns
    num_rows = (num_keys + num_cols - 1) // num_cols  # Calculate rows dynamically

    # Create a plot
    plt.figure(figsize=(12, 2 * num_rows))
    for i, key in enumerate(keys):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.plot(imember_cluster, data[key], marker='o')
        plt.title(key, fontsize=10)
        plt.xlabel("Cluster Member", fontsize=8)
        plt.ylabel(key, fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

    # Add time and datetime information as text
    time_info = f"ICLUSTER = {icluster} PIXEL = {usepixel} \n time range: {times[0]} - {times[-1]}, datetime range: {imember_datetime[0]} - {imember_datetime[-1]}"
    plt.figtext(0.5, 0.91, time_info, ha="center", fontsize=8, wrap=True)

    # Adjust layout to minimize gaps
    plt.tight_layout(rect=[0, 0.01, 1, 0.90])

    # Generate file name based on cluster size and keys
    filename = f"cluster_trend_pixel{usepixel:02d}_ic{icluster:03d}_clen{len(cluster)}.png"
    filepath = f"{output_dir}/{filename}"

    # Save plot to file
    plt.savefig(filepath)
    plt.show()
    plt.close()

    print(f"Cluster plot saved as: {filepath}")


def process_event_list(event_list, usepixel, nicutlength, debug=True):

    if debug: print(f"start process_event_list : {event_list[0]}, {usepixel}, {nicutlength}")
    icluster_array, imember_array, bit14_array, bit15_array = [],[],[],[]

    pixel_list  = event_list["PIXEL"]
    if debug: print(f"pixel_list : {type(pixel_list)}")

    pixel_mask = np.where(pixel_list == usepixel)[0]
    if debug: print(f"pixel_mask : {pixel_mask}")

    """Main function to process the event list and identify clusters."""
    clusters = []  # List to store confirmed clusters
    current_cluster = []  # Temporary storage for the ongoing cluster
    prev_event = None  # Placeholder for the previous event

    # Loop through the event list
    for k, idx in enumerate(pixel_mask): 
        if debug: print(f"idx = {idx}")
        icluster, imember, bit14, bit15 = 0,0,False,False # init

        event = event_list[pixel_mask[k]]
        # Check the next event and the event after that (ensure safe access by validating the range).
        event_n1 = event_list[pixel_mask[k + 1]] if k + 1 < len(pixel_mask) else None
        event_n2 = event_list[pixel_mask[k + 2]] if k + 2 < len(pixel_mask) else None

        if not current_cluster:  # current_cluster is empty
            # Check for a new cluster start condition
            if event_n1 is not None and event_n2 is not None:  # Execute only if there are at least three remaining events.
                if is_cluster_start(event, event_n1, event_n2, prev_event):
                    if debug: print(f"Cluster candidate found.")
                    current_cluster.append(event)
                    bit14 = True
                    icluster = len(clusters)
                    imember = len(current_cluster)

        else:
            # Process the ongoing cluster
            if is_cluster_continue(event, current_cluster, para1=nicutlength):
                if debug: print(f"Cluster continues.")
                current_cluster.append(event)
                bit15 = True
                icluster = len(clusters) 
                imember = len(current_cluster)
            else:
                if is_cluster_end(event, current_cluster):
                    if debug: print(f"Cluster closed. Store the bufffer.")
                    # Finalize and save the cluster
                    icluster = len(clusters)
                    imember = len(current_cluster)
                    bit15 = True
                    clusters.append(current_cluster)
                    confirm_cluster(current_cluster, icluster, usepixel) # check the content for debug
                    current_cluster = []  # Reset the cluster
                else:
                    if debug: print(f"Cluster is not closed. Delete the bufffer.")
                    current_cluster = []  # Reset the cluster                    
        prev_event = event  # Update the previous event
        icluster_array.append(icluster)
        imember_array.append(imember)
        bit14_array.append(bit14)
        bit15_array.append(bit15)

    # Check and confirm the final cluster, if any
    if current_cluster:
        confirmed_cluster = confirm_cluster(current_cluster)
        if confirmed_cluster:
            clusters.append(confirmed_cluster)

    return clusters, icluster_array, imember_array, bit14_array, bit15_array

def main():
    parser = argparse.ArgumentParser(description="Modify specific bits in the STATUS column of a FITS file for each pixel.")
    parser.add_argument('input_fits', help="Path to the input FITS file")
    parser.add_argument('--usepixels', '-p', type=str, help='Comma-separated list of pixels to plot', default=','.join(map(str, range(36))))
    parser.add_argument('--nicutlength', '-ni', type=int, default=120, help='The interval of the NEXT_INTERVAL')
    parser.add_argument('--debug', '-d', action='store_true', default=False, help='The debug flag')

    args = parser.parse_args()

    input_fits = args.input_fits
    usepixels = list(map(int, args.usepixels.split(',')))
    nicutlength = args.nicutlength
    debug = args.debug

    output_fits = "cluster_" + input_fits

    # Open the input FITS file
    with fits.open(input_fits) as hdul:
        # Access the STATUS and PIXEL columns
        status_data = hdul[1].data['STATUS']
        pixel_data = hdul[1].data['PIXEL']
        n_rows = len(status_data)
        # Create a new column for cluster index
        icluster = np.zeros(n_rows, dtype=np.int32)
        imember = np.zeros(n_rows, dtype=np.int32)

        # Process each pixel independently
        for pixel in usepixels:
            # Get rows corresponding to the current pixel
            pixel_mask = (pixel_data == pixel) # bool type with a length of n_rows 
            # Process the event list to find clusters
            clusters, icluster_array, imember_array, bit14_array, bit15_array = process_event_list(hdul[1].data, pixel, nicutlength, debug=debug)

            if debug:
                print(f"Cluster Number in {np.sum(pixel_mask)} = {len(clusters)}, {len(icluster_array)}, {len(imember_array)}, {len(bit14_array)}, {len(bit15_array)}")
            # Ensure lengths match the number of rows for the current pixel
            if len(bit14_array) != np.sum(pixel_mask) or len(bit15_array) != np.sum(pixel_mask) or len(icluster_array) != np.sum(pixel_mask) or len(imember_array) != np.sum(pixel_mask):
                raise ValueError(f"Length mismatch for pixel {pixel} in bit arrays.")

            # Modify the STATUS column and ICLUSTER, IMEMBER for the current pixel
            icluster[pixel_mask] = icluster_array
            imember[pixel_mask] = imember_array
            status_data[pixel_mask, 14] = bit14_array
            status_data[pixel_mask, 15] = bit15_array

        # Add the row indices as a new column
        col_defs = hdul[1].columns
        new_col1 = fits.Column(name='ICLUSTER', format='J', array=icluster)
        new_col2 = fits.Column(name='IMEMBER', format='J', array=imember)
        new_cols = fits.ColDefs(col_defs + new_col1 + new_col2)

        # Create a new binary table with the updated columns
        new_hdu = fits.BinTableHDU.from_columns(new_cols, header=hdul[1].header)
        hdul[1] = new_hdu
        # Save changes to a new file
        hdul.writeto(output_fits, overwrite=True)
        print(f"Modified FITS file saved as: {output_fits}")

if __name__ == "__main__":
    main()
