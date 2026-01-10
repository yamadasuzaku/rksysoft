#!/usr/bin/env python

"""
Script Name: resolve_ana_pixel_Ls_define_cluster.py

Description:
This script modifies the 14th and 15th bits in the STATUS column of a FITS file
for each pixel and adds a new column for row indices for pseudo-Ls events. 

History:
- ver 1, 2024.12.19, first draft, collaborated with Yuusuke Uchida-san to resolve a bug in FITS writing.
"""

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
from matplotlib.ticker import MaxNLocator

import os 

# 出力ディレクトリの作成
output_dir = "fig_cluster"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory '{output_dir}' created.")
else:
    print(f"Directory '{output_dir}' already exists.")

# global variables
TEMPLETE_LENGTH_MR=219
# Define global variables
MJD_REFERENCE_DAY = 58484
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')  # Reference time in Modified Julian Day

def compute_datetimes(mjdrefi, times):
    reftime = Time(mjdrefi, format='mjd')
    return [reftime.datetime + datetime.timedelta(seconds=float(t)) for t in times]

def is_cluster_start(event, event_n1, event_n2, prev_event, key1 = "NEXT_INTERVAL", para1_start=TEMPLETE_LENGTH_MR, para1_continue=120, debug=True):
    """Determine the condition for starting a new cluster."""
    # Logic for cluster start condition
    status =  (event[key1] < para1_start) & (event_n1[key1] < para1_continue) & (event_n2[key1] < para1_continue)
#    if debug: 
    if status:
        print(f"is_cluster_start {status} = ({event[key1]} < {para1_start}) & ({event_n1[key1]} < {para1_continue}) & ({event_n2[key1]} < {para1_continue})")
    return status  # Return True/False based on the condition

def is_cluster_continue(event, current_cluster, key1 = "NEXT_INTERVAL", para1_continue=120, debug=False):
    """Check if the current event continues the existing cluster."""
    # Logic for cluster continuation condition
    status = event[key1] < para1_continue
    if debug:
        print(f"is_cluster_continue {status} = {event[key1]} < {para1_continue}")
    return status  # Return True/False based on the condition

def is_cluster_end(event, current_cluster, min_cluster_num=3, debug=False):
    """Determine if the cluster has reached its end."""
    # Logic for cluster end condition
    status = len(current_cluster) >= min_cluster_num
    if debug:
        print(f"is_cluster_end {status} = {len(current_cluster)} >= {min_cluster_num}")
    return status  # Return True/False based on the condition

def confirm_cluster(current_cluster, icluster, usepixel, event_list, pixel_mask, pixel_mask_index, key1="ITYPE", debug=False, show=False, do_plot=True):
    """Confirm and finalize the cluster, applying any necessary checks or filters."""
    # Logic for confirming or saving the finalized cluster
    if debug:
        print(f"icluster = {icluster}, pixel = {usepixel} key1 = {key1}")
        for i, mcluster in enumerate(current_cluster):
            print(i, mcluster[key1])
    if do_plot:
        plot_onecluster(current_cluster, icluster, usepixel, event_list, pixel_mask, pixel_mask_index, debug=debug, show=show)

    status = True
    return status  # Return good or bad if any

def plot_onecluster(cluster, icluster, usepixel, event_list, pixel_mask, pixel_mask_index, debug=False, \
    keys=["LO_RES_PH", "DERIV_MAX", "ITYPE", "RISE_TIME", "PREV_INTERVAL", "NEXT_INTERVAL"], output_dir="fig_cluster", show=False):
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
    imember_length = len(cluster)
    imember_cluster = np.arange(imember_length) + 1

    # get out of cluster evenst 
    pixel_mask_index
    prev_n1 = event_list[pixel_mask[pixel_mask_index - imember_length]] if (pixel_mask_index - imember_length < len(pixel_mask)) or (pixel_mask_index - imember_length >= 0) else None
    next_n1 = event_list[pixel_mask[pixel_mask_index + 1]] if (pixel_mask_index +1 < len(pixel_mask)-1 ) and (pixel_mask_index +1 >= 0) else None

    # Print cluster information to standard output
    print(f"Cluster Information: pixel={usepixel}, icluster={icluster}, cluster length = {len(cluster)}")
    print(f"     TIME: {times[0]} to {times[-1]}")
    print(f"     DATETIME: {imember_datetime[0]} to {imember_datetime[-1]}")

    for key in keys:
        print(f"     {key}: {data[key]}")

    # Determine the number of rows and columns for subplots
    num_keys = len(keys)
    num_cols = 2  # Fixed number of columns
    num_rows = (num_keys + num_cols - 1) // num_cols  # Calculate rows dynamically

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 2 * num_rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, key in enumerate(keys):
        ax = axes[i]  # Get the corresponding axis

        # Plot cluster data
        ax.plot(imember_cluster, data[key], marker='o', color="b", label="cluster")

        # Set titles and labels
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("Cluster Member", fontsize=8)
        ax.set_ylabel(key, fontsize=8)

        # Set integer ticks on the x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Annotate values on the data points
        for x, y in zip(imember_cluster, data[key]):
            ax.annotate(f'{y}', xy=(x, y), xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=8)

        if prev_n1 is not None:
            ax.plot(0, prev_n1[key], marker='o', color="y", label="out of cluster (-1)")
            # Annotate the previous event value
            ax.annotate(f'{prev_n1[key]}', xy=(0, prev_n1[key]), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=8, color="y")

        if next_n1 is not None:
            ax.plot(imember_length + 1, next_n1[key], marker='s', color="y", label="out of cluster (+1)")
            # Annotate the previous event value
            ax.annotate(f'{next_n1[key]}', xy=(imember_length + 1, next_n1[key]), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=8, color="y")

        # Add legend at one time 
        if i == 0:
            ax.legend(fontsize=8)

    # Remove unused subplots
    for j in range(len(keys), len(axes)):
        fig.delaxes(axes[j])

    # Add time and datetime information as text
    time_info = (f"ICLUSTER = {icluster} PIXEL = {usepixel} \n "
                 f"time range: {times[0]} - {times[-1]}, datetime range: {imember_datetime[0]} - {imember_datetime[-1]}")
    fig.text(0.5, 0.91, time_info, ha="center", fontsize=8, wrap=True)

    # Adjust layout to minimize gaps
    fig.tight_layout(rect=[0, 0.01, 1, 0.90])

    # Generate file name based on cluster size and keys
    filename = f"cluster_trend_pixel{usepixel:02d}_ic{icluster:03d}_clen{len(cluster)}.png"
    filepath = f"{output_dir}/{filename}"

    # Save plot to file
    plt.savefig(filepath)
    if show:
        plt.show()
    plt.close()

    print(f"Plot saved to {filepath}")

def process_event_list(event_list, usepixel, para1_start=TEMPLETE_LENGTH_MR, para1_continue=120, min_cluster_num=3, \
                                                                                           debug=True, show=False, output_dir="output_dir"):

    if debug: print(f"start process_event_list : {event_list[0]}, {usepixel}, {para1_start} {para1_continue} {min_cluster_num}")
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
    for pixel_mask_index, idx in enumerate(pixel_mask): 
#        if debug: print(f"idx = {idx}")
        icluster, imember, bit14, bit15 = 0,0,False,False # init

        event = event_list[pixel_mask[pixel_mask_index]]
        # Check the next event and the event after that (ensure safe access by validating the range).
        event_n1 = event_list[pixel_mask[pixel_mask_index + 1]] if pixel_mask_index + 1 < len(pixel_mask) else None
        event_n2 = event_list[pixel_mask[pixel_mask_index + 2]] if pixel_mask_index + 2 < len(pixel_mask) else None

        if not current_cluster:  # current_cluster is empty
            # Check for a new cluster start condition
            if event_n1 is not None and event_n2 is not None:  # Execute only if there are at least three remaining events.
                if is_cluster_start(event, event_n1, event_n2, prev_event, para1_start=para1_start, para1_continue=para1_continue, debug=debug):
                    if debug: print(f"Cluster candidate found.")
                    current_cluster.append(event)
                    bit14 = True
                    icluster = len(clusters) + 1
                    imember = len(current_cluster)

        else:
            # Process the ongoing cluster
            if is_cluster_continue(event, current_cluster, para1_continue=para1_continue, debug=debug):
                if debug: print(f"Cluster continues.")
                current_cluster.append(event)
                bit15 = True
                icluster = len(clusters) + 1 
                imember = len(current_cluster)
            else:
                if is_cluster_end(event, current_cluster, min_cluster_num=min_cluster_num, debug=debug):
                    if debug: print(f"Cluster closed. Store the bufffer.")
                    current_cluster.append(event)
                    imember = len(current_cluster)                    
                    # Finalize and save the cluster
                    icluster = len(clusters) + 1
                    clusters.append(current_cluster)
                    bit15 = True
                    confirm_cluster(current_cluster, icluster, usepixel, event_list, pixel_mask, pixel_mask_index, debug=debug, show=show) # check the content for debug
                    current_cluster = []  # Reset the cluster
                else:
                    imember = len(current_cluster)
                    print(f"Cluster is not closed. Delete the IMEMBER {imember}, ICLUSTER and current_cluster.")
                    # delete the stored imember, icluster stored in a length of imember
                    for i in range(imember): 
                        imember_array[ -1-i] = 0 # reset
                        icluster_array[ -1-i] = 0 # reset
                    # delete the stored current_cluster
                    current_cluster = []  # reset
        prev_event = event  # Update the previous event
        icluster_array.append(icluster)
        imember_array.append(imember)
        bit14_array.append(bit14)
        bit15_array.append(bit15)
        # print(f"icluster_array[-10:-1] = {icluster_array[-10:-1]}")
        # print(f"imember_array[-10:-1] = {imember_array[-10:-1]}")

    if current_cluster:
        current_cluster = []  # Do not store the last buffer

    return clusters, icluster_array, imember_array, bit14_array, bit15_array

def main():
    parser = argparse.ArgumentParser(description="Modify specific bits in the STATUS column of a FITS file for each pixel.")
    parser.add_argument('input_fits', help="Path to the input FITS file")
    parser.add_argument('--usepixels', '-p', type=str, help='Comma-separated list of pixels to plot', default=','.join(map(str, range(36))))
    parser.add_argument('--para1_start', '-p1s', type=int, default=TEMPLETE_LENGTH_MR, help='The interval of the NEXT_INTERVAL')
    parser.add_argument('--para1_continue', '-p2c', type=int, default=200, help='The interval of the NEXT_INTERVAL')
    parser.add_argument('--min_cluster_num', '-m', type=int, default=3, help='The interval of the NEXT_INTERVAL')

    parser.add_argument('--debug', '-d', action='store_true', default=False, help='The debug flag')
    parser.add_argument('--show', '-s', action='store_true', default=False, help='plt.show is used.')
    parser.add_argument('--outname', '-o', type=str, help='fname tag used for output file name', default="addcluster_")

    args = parser.parse_args()

    input_fits = args.input_fits
    usepixels = list(map(int, args.usepixels.split(',')))
    para1_start = args.para1_start
    para1_continue = args.para1_continue
    min_cluster_num = args.min_cluster_num

    debug = args.debug
    show = args.show
    outname = args.outname

    output_fits = outname + input_fits

    # Open the input FITS file

    with fits.open(input_fits) as hdul:
        # Access the STATUS and PIXEL columns
        status_data = hdul[1].data['STATUS']
        pixel_data = hdul[1].data['PIXEL']

        # create PREV/NEXT_INTERVAL to store them as int64
        prev_interval_col = fits.Column(name='PREV_INTERVAL', format='K')
        prev_interval_col.array = hdul[1].data['PREV_INTERVAL'].astype(np.int64)
        next_interval_col = fits.Column(name='NEXT_INTERVAL', format='K')
        next_interval_col.array = hdul[1].data['NEXT_INTERVAL'].astype(np.int64)

        n_rows = len(status_data)
        # Create a new column for cluster index
        icluster = np.zeros(n_rows, dtype=np.int64)
        imember = np.zeros(n_rows, dtype=np.int64)

        # Process each pixel independently
        for pixel in usepixels:
            # Get rows corresponding to the current pixel
            pixel_mask = (pixel_data == pixel) # bool type with a length of n_rows 
            # Process the event list to find clusters
            clusters, icluster_array, imember_array, bit14_array, bit15_array = process_event_list(hdul[1].data, pixel, para1_start=para1_start,\
                            para1_continue=para1_continue, min_cluster_num=min_cluster_num, debug=debug, show=show, output_dir = output_dir)

            if debug:
                print(f"Cluster number is {len(clusters)}; check all the same length {np.sum(pixel_mask)}, {len(icluster_array)}, {len(imember_array)}, {len(bit14_array)}, {len(bit15_array)}")
            # Ensure lengths match the number of rows for the current pixel
            if len(bit14_array) != np.sum(pixel_mask) or len(bit15_array) != np.sum(pixel_mask) or len(icluster_array) != np.sum(pixel_mask) or len(imember_array) != np.sum(pixel_mask):
                raise ValueError(f"Length mismatch for pixel {pixel} in bit arrays.")

            # Modify the STATUS column and ICLUSTER, IMEMBER for the current pixel
            icluster[pixel_mask] = icluster_array
            imember[pixel_mask] = imember_array
#            print(f"pixel={pixel} bit14_array = {bit14_array}")
#            print(f"pixel={pixel} bit15_array = {bit15_array}")
            status_data[pixel_mask, 14] = bit14_array
            status_data[pixel_mask, 15] = bit15_array

        # Add the row indices as a new column
        col_defs = hdul[1].columns
        col_defs['STATUS'].array = status_data

        new_col1 = fits.Column(name='ICLUSTER', format='J', array=icluster)
        new_col2 = fits.Column(name='IMEMBER', format='J', array=imember)
        new_cols = fits.ColDefs(col_defs + new_col1 + new_col2)

        new_cols = []
        for col in col_defs+new_col1+new_col2 :
            if col.name == 'PREV_INTERVAL' : 
                new_cols.append(prev_interval_col)
            elif col.name == 'NEXT_INTERVAL' :
                new_cols.append(next_interval_col)
            else :
                new_cols.append(col)

        # Create a new binary table with the updated columns
        new_hdu = fits.BinTableHDU.from_columns(new_cols, header=hdul[1].header)
        hdul[1] = new_hdu
        # Save changes to a new file
        hdul.writeto(output_fits, overwrite=True)
        print(f"Modified FITS file saved as: {output_fits}")

if __name__ == "__main__":
    main()
