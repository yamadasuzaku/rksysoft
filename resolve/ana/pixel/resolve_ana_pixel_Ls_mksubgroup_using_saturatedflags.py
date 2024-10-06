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

# Constants
MJD_REFERENCE_DAY = 58484
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')
TIME_50MK = 150526800.0

# according to 
RECORD_LEN=1024
# /Users/syamada/work/ana/resolve/resolve_git/eclipse/2307_XRISM_PFT_TC11/main2/main2-TC11_01z_VAB_230705_RT.cps
# 4034    # CSXS3070R (1) SET_PRE_TRIG_LEN_HM 140 27 (nominal), SET_MIN_MAX_SHIFT
PRE_TRIG_LEN_H = 140
PRE_TRIG_LEN_M = 27
PULSE_THRES = 120
SECOND_THRES_MIN = 120 
SECOND_TRIG_GAP_LEN = 25 # 2 ms
SLOPE_DETECT_LEN = 20 # 1.6 ms
SLOPE_SKIP_LEN =0 # 0 ms
SECOND_THRES_FRAC = 20 # 1.6 ms
SECOND_THRES_USE_LEN = 75 # 6 ms
SPARE_LEN=8 # offset 
CLIPTHRES=12235
FALL_END_THRES=0

def parse_filter_conditions(conditions):
    filters = []
    for condition in conditions.split(","):
        col, value = condition.split("==")
        filters.append((col.strip(), float(value.strip())))
    return filters

def apply_filters(data, filters):
    mask = np.ones(len(data), dtype=bool)
    for col, value in filters:
        mask &= (data[col] == value)
    return data[mask]

def plot_fits_data(file_name, x_col, y_col, hdu, title, outfname, tolerance, filters=None, plotflag=False, markers=".", datatime_flag=True, \
                       trigcol1name="LO_RES_PH", trigcol2name="ITYPE", trigcol3name="RISE_TIME", trigcol4name="PHA"):

    print(f"Opening FITS file: {file_name}")
    with fits.open(file_name) as hdul:
        data = hdul[hdu].data  # Access the data from the specified HDU

        # Apply filters if any
        if filters:
            print("Applying filters...")
            data = apply_filters(data, filters)
            print(f"Filtered data: {len(data)} rows")

        # Convert columns to numpy arrays for easier plotting
        x_data = data[x_col]
        if datatime_flag:
            print("Converting time data...")
            x_data = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in x_data])
        y_data = data[y_col]

        trigcol1 = data[trigcol1name]
        trigcol2 = data[trigcol2name]
        trigcol3 = data[trigcol3name]
        trigcol4 = data[trigcol4name]

        # Split the sequence based on the tolerance value
        print("Splitting the sequence...")
        split_sequences = split_sequence(y_data, tolerance, minnum=2, trigcol1=trigcol1, trigcol2=trigcol2, trigcol3=trigcol3, trigcol4=trigcol4)

        # Visualization
        print("Creating plots...")
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        plt.suptitle(title)

        # Top plot: Visualizing the continuous segments
        axs[0].plot(x_data, y_data, markers, label=y_col)

        all_segment = []
        for j, (indexes, segment, tcol1, tcol2, tcol3, tcol4) in enumerate(split_sequences):
            all_segment.extend(segment)
            if j == 0:
                axs[0].plot(x_data[indexes], y_data[indexes], "ro", alpha=0.2, label="meet condition")
                axs[0].legend()
            else:
                axs[0].plot(x_data[indexes], y_data[indexes], "ro", alpha=0.2)

        axs[0].axhline(y=tolerance, color='r', linestyle='--', label="threshold = " + str(tolerance))
        axs[0].set_xlabel(x_col)
        axs[0].set_ylabel(y_col)
        axs[0].set_title(f'Continuous Segments Where {y_col}[i] < {tolerance} continues more than once.')
        axs[0].set_ylim(-10, 150)

        # Middle plot: Histogram of segment lengths
        segment_lengths = [len(segment) for _, segment, _, _, _, _ in split_sequences]
        bins = np.arange(0.5, max(segment_lengths) + 0.5, 1)
        axs[1].hist(segment_lengths, bins=bins, edgecolor='black', label="Length in a subset of " + y_col)
 #       axs[1].set_xticks(np.arange(2, max(segment_lengths) + 1))
        axs[1].set_xlabel('Length of the continuous Segments more than one')
        axs[1].set_ylabel('Frequency')
        axs[1].set_xlim(0,20)
        axs[1].set_yscale("log")
        axs[1].set_title('Histogram of Segment Lengths')
        axs[1].legend()

        # Bottom plot: Histogram of segment (e.g., PREV_INTERVAL) 
        bins = np.arange(0.5, 80 + 0.5, 1)
        axs[2].hist(all_segment, bins=bins, edgecolor='black', label="Countents of " + y_col)
#        axs[2].set_xticks(np.arange(2, max(segment_lengths) + 1))
        axs[2].set_xlabel('Countents of ' + y_col)
        axs[2].set_ylabel('Frequency')
        axs[2].set_yscale("log")
        axs[2].set_title('Histogram of Segment contents')
        axs[2].legend()

        plt.tight_layout()
        plt.savefig(outfname)
        print(f"Saved plot as {outfname}")
        if plotflag:
            plt.show()
        else:
            plt.close()

        ###########################################################################################
        ## plot as a function of consecutive number
        ###########################################################################################

        fig, axs = plt.subplots(5, 1, figsize=(10, 8))
        plt.suptitle(title)
        for j, (indexes, segment, tcol1, tcol2, tcol3, tcol4) in enumerate(split_sequences):
            input_order = np.arange(len(segment))
            if j == 0:
                axs[0].plot(input_order, segment, ".-", alpha=0.2, label=y_col)
                axs[0].legend()
            else:
                axs[0].plot(input_order, segment, ".-", alpha=0.2)

#        axs[0].axvline(x=tolerance, color='r', linestyle='--', label="threshold = " + str(tolerance))
        axs[0].set_xlabel("consecutive number")
        axs[0].set_ylabel(y_col)
        axs[0].set_title(f'Continuous Segments Where {y_col}[i] < {tolerance} continues more than once.')
#        axs[0].set_xlim(-10, 150)

        for j, (indexes, segment, tcol1, tcol2, tcol3, tcol4) in enumerate(split_sequences):
            input_order = np.arange(len(segment))
            if j == 0:
                axs[1].plot(input_order, tcol1, ".-", alpha=0.2, label=trigcol1name)
                axs[1].legend()
            else:
                axs[1].plot(input_order, tcol1, ".-", alpha=0.2)

        axs[1].axhline(y=CLIPTHRES, color='r', linestyle='--', label="threshold = " + str(CLIPTHRES))
        axs[1].set_xlabel("consecutive number")
        axs[1].set_ylabel(trigcol1name)
#        axs[1].set_title(f'Continuous Segments Where {y_col}[i] < {tolerance} continues more than once.')
#        axs[0].set_xlim(-10, 150)

        for j, (indexes, segment, tcol1, tcol2, tcol3, tcol4) in enumerate(split_sequences):
            input_order = np.arange(len(segment))
            if j == 0:
                axs[2].plot(input_order, tcol2, ".-", alpha=0.2, label=trigcol2name)
                axs[2].legend()
            else:
                axs[2].plot(input_order, tcol2, ".-", alpha=0.2)

#        axs[0].axvline(x=tolerance, color='r', linestyle='--', label="threshold = " + str(tolerance))
        axs[2].set_xlabel("consecutive number")
        axs[2].set_ylabel(trigcol2name)
#        axs[2].set_title(f'Continuous Segments Where {y_col}[i] < {tolerance} continues more than once.')
#        axs[0].set_xlim(-10, 150)

        for j, (indexes, segment, tcol1, tcol2, tcol3, tcol4) in enumerate(split_sequences):
            input_order = np.arange(len(segment))
            if j == 0:
                axs[3].plot(input_order, tcol3, ".-", alpha=0.2, label=trigcol3name)
                axs[3].legend()
            else:
                axs[3].plot(input_order, tcol3, ".-", alpha=0.2)

#        axs[0].axvline(x=tolerance, color='r', linestyle='--', label="threshold = " + str(tolerance))
        axs[3].set_xlabel("consecutive number")
        axs[3].set_ylabel(trigcol3name)
#        axs[2].set_title(f'Continuous Segments Where {y_col}[i] < {tolerance} continues more than once.')
#        axs[0].set_xlim(-10, 150)


        for j, (indexes, segment, tcol1, tcol2, tcol3, tcol4) in enumerate(split_sequences):
            input_order = np.arange(len(segment))
            if j == 0:
                axs[4].plot(input_order, tcol4, ".-", alpha=0.2, label=trigcol4name)
                axs[4].legend()
            else:
                axs[4].plot(input_order, tcol4, ".-", alpha=0.2)

#        axs[0].axvline(x=tolerance, color='r', linestyle='--', label="threshold = " + str(tolerance))
        axs[4].set_xlabel("consecutive number")
        axs[4].set_ylabel(trigcol4name)
#        axs[2].set_title(f'Continuous Segments Where {y_col}[i] < {tolerance} continues more than once.')
#        axs[0].set_xlim(-10, 150)


        plt.tight_layout()
        outfname2 = "comp_" + outfname
        plt.savefig(outfname2)
        print(f"Saved plot as {outfname2}")
        if plotflag:
            plt.show()
        else:
            plt.close()



        ###########################################################################################
        ## plot as a function of tcols  
        ###########################################################################################

        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        plt.suptitle(title)

        for j, (indexes, segment, tcol1, tcol2, tcol3, tcol4) in enumerate(split_sequences):
            input_order = np.arange(len(segment))
            if j == 0:
                axs[0].plot(tcol1, tcol3, ".-", alpha=0.2, label=y_col)
                axs[0].legend()
            else:
                axs[0].plot(tcol1, tcol3, ".-", alpha=0.2, label=y_col)

#        axs[0].axvline(x=tolerance, color='r', linestyle='--', label="threshold = " + str(tolerance))
        axs[0].set_xlabel(trigcol1name)
        axs[0].set_ylabel(trigcol3name)
        axs[0].set_title(f'Continuous Segments Where {y_col}[i] < {tolerance} continues more than once.')
#        axs[0].set_xlim(-10, 150)

        for j, (indexes, segment, tcol1, tcol2, tcol3, tcol4) in enumerate(split_sequences):
            input_order = np.arange(len(segment))
            if j == 0:
                axs[1].plot(tcol1, tcol2, ".-", alpha=0.2, label=y_col)
                axs[1].legend()
            else:
                axs[1].plot(tcol1, tcol2, ".-", alpha=0.2, label=y_col)

#        axs[0].axvline(x=tolerance, color='r', linestyle='--', label="threshold = " + str(tolerance))
        axs[1].set_xlabel(trigcol1name)
        axs[1].set_ylabel(trigcol2name)
        axs[1].set_title(f'Continuous Segments Where {y_col}[i] < {tolerance} continues more than once.')
#        axs[0].set_xlim(-10, 150)

        for j, (indexes, segment, tcol1, tcol2, tcol3, tcol4) in enumerate(split_sequences):
            input_order = np.arange(len(segment))
            if j == 0:
                axs[2].plot(tcol1, tcol4, ".-", alpha=0.2, label=y_col)
                axs[2].legend()
            else:
                axs[2].plot(tcol1, tcol4, ".-", alpha=0.2, label=y_col)

#        axs[0].axvline(x=tolerance, color='r', linestyle='--', label="threshold = " + str(tolerance))
        axs[2].set_xlabel(trigcol1name)
        axs[2].set_ylabel(trigcol4name)
        axs[2].set_title(f'Continuous Segments Where {y_col}[i] < {tolerance} continues more than once.')
#        axs[0].set_xlim(-10, 150)



        plt.tight_layout()
        outfname2 = "comptcols_" + outfname
        plt.savefig(outfname2)
        print(f"Saved plot as {outfname2}")
        if plotflag:
            plt.show()
        else:
            plt.close()



def split_sequence(seq, tolerance, minnum=2, trigcol1=None, trigcol2=None, trigcol3=None, trigcol4=None):
    # sed is assumed to be PREV_INTERVAL
    print(f"Starting sequence splitting ... with tolerance of {tolerance}")
    result = []
    current_segment = []
    current_indices = []
    current_trigcol1 = []
    current_trigcol2 = []
    current_trigcol3 = []
    current_trigcol4 = []

    for i, num in enumerate(seq):
        if num < tolerance:
            current_indices.append(i)
            current_segment.append(num)
            current_trigcol1.append(trigcol1[i])
            current_trigcol2.append(trigcol2[i])
            current_trigcol3.append(trigcol3[i])
            current_trigcol4.append(trigcol4[i])
        else:
            if len(current_segment) >= minnum:
                result.append((current_indices, current_segment, current_trigcol1, current_trigcol2, current_trigcol3, current_trigcol4))
                print(f"Found segment: {current_segment} at indices {current_indices}, trigcol1 {current_trigcol1}, trigcol2 {current_trigcol2}, trigcol3 {current_trigcol3} trigcol4 {current_trigcol4}")
            current_indices = []
            current_segment = []
            current_trigcol1 = []
            current_trigcol2 = []
            current_trigcol3 = []
            current_trigcol4 = []

    if len(current_segment) >= minnum:
        result.append((current_indices, current_segment, current_trigcol1, current_trigcol2, current_trigcol3, current_trigcol4))
        print(f"Found segment: {current_segment} at indices {current_indices}, trigcol1 {current_trigcol1} trigcol2 {current_trigcol2} trigcol3 {current_trigcol3} trigcol4 {current_trigcol4}")

    print("Sequence splitting complete.")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This program is used to check more than two-something groups for each pixel',
        epilog='''
        Example:
        resolve_ana_pixel_Ls_mksubgroup_using_saturatedflags.py xa000114000rsl_p0px1000_uf_prevnext_cutclgti.fits TIME PREV_INTERVAL -f "PIXEL==14" -p
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file_name", type=str, help="Path to the FITS file")
    parser.add_argument("x_col", type=str, help="Column name for the x-axis")
    parser.add_argument("y_col", type=str, help="Column name for the y-axis")
    parser.add_argument("--trigcol1name", type=str, default="LO_RES_PH", help="The colomn used for trrigger condition, default is LO_RES_PH > CLIPTHRES")
    parser.add_argument("--trigcol2name", type=str, default="ITYPE", help="trigger column 2")
    parser.add_argument("--trigcol3name", type=str, default="RISE_TIME", help="trigger column 3")
    parser.add_argument("--trigcol4name", type=str, default="PHA", help="trigger column 4")
    parser.add_argument('--hdu', '-n', type=int, default=1, help='Number of FITS HDU')
    parser.add_argument("--filters", '-f', type=str, help="Comma-separated filter conditions in the format 'COLUMN==VALUE'", default="")
    parser.add_argument('--plot', '-p', action='store_true', default=False, help='Flag to display plot')
    parser.add_argument("--markers", '-m', type=str, help="Marker type", default=".")
    parser.add_argument('--tolerance', '-t', type=float, default=100, help='Tolerance for subset threshold')
    parser.add_argument('--datatime_flag', '-dtime', action='store_true', default=False, help='Flag to convert x_col into datetime.')

    args = parser.parse_args()
    filter_conditions = parse_filter_conditions(args.filters) if args.filters else None
    title = f"{args.file_name} : filtered with {args.filters}"
    print(filter_conditions)
    ftag = args.file_name.split(".")[0]
    if args.filters:
        outfname = f"subset_{ftag}_{filter_conditions[0][0]}{int(filter_conditions[0][1]):02d}.png"
    else:
        outfname = f"subset_{ftag}.png"

    print("Starting plot generation...")
    plot_fits_data(args.file_name, args.x_col, args.y_col, args.hdu, title, outfname, args.tolerance, 
        filters=filter_conditions, plotflag=args.plot, markers=args.markers, datatime_flag=args.datatime_flag, \
        trigcol1name = args.trigcol1name, trigcol2name = args.trigcol2name, trigcol3name = args.trigcol3name, trigcol4name = args.trigcol4name)
    print("Plot generation complete.")
