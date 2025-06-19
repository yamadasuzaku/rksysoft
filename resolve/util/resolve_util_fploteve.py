#!/usr/bin/env python

# Import necessary libraries
import argparse
import matplotlib.pyplot as plt
import sys
from astropy.time import Time
import datetime
import numpy as np
import re
from astropy.io import fits
import ast  # To safely evaluate string into dictionary

# Set default plot parameters
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

# Define global variables for reference time
MJD_REFERENCE_DAY = 58484  # Reference MJD (Modified Julian Date)
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')  # Set the reference time using astropy

# Default y_ranges dictionary
default_y_ranges = {
    "PHA": (0, 65535),
    "RISE_TIME": (0, 255),
    "LO_RES_PH": (0, 16383),
    "PREV_LO_RES_PH": (0, 16383),
    "DERIV_MAX": (0, 32767),
    "EPI": (0, 33000),
    "EPI2": (0, 33000),
    "PI": (0, 20000),
    "ITYPE": (0, 7),
    "PREV_ITYPE": (0, 7),
    "PIXEL": (0, 35),
    "TICK_SHIFT": (-8, 7),
    "NEXT_INTERVAL": (0, 300),
    "PREV_INTERVAL": (0, 300)
}

# Custom tick positions for specific columns
y_ticks_dict = {
    "ITYPE": [0, 1, 2, 3, 4, 5, 6, 7],
    "PIXEL": np.arange(36),
    "TICK_SHIFT": [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
}

# Function to manually scale values between 0 and 1 based on a range
def manual_scale(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Function to generate evenly spaced ticks for a given range (min, max) and number of ticks
def generate_ticks(y_min, y_max, n):
    """
    Generate 'n' ticks between y_min and y_max.
    """
    ticks = np.linspace(y_min, y_max, n)  # Create evenly spaced ticks
    return ticks

# Function to parse filter conditions (for example: 'PIXEL==9')
def parse_filter_conditions(conditions):
    filters = []
    # Regular expression to match filter conditions
    condition_pattern = re.compile(r"(.*?)(==|<=|>=|<|>|!=)(.*)")
    for condition in conditions.split(","):
        match = condition_pattern.match(condition.strip())
        if match:
            col, op, value = match.groups()  # Extract column, operator, and value
            filters.append((col.strip(), op, float(value.strip())))
        else:
            raise ValueError(f"Invalid filter condition: {condition}")
    return filters

# Function to apply filters to the data
def apply_filters(data, filters):
    mask = np.ones(len(data), dtype=bool)  # Start with a mask of all True
    for col, op, value in filters:
        if op == "==":
            mask &= (data[col] == value)
        elif op == "!=":
            mask &= (data[col] != value)
        elif op == "<":
            mask &= (data[col] < value)
        elif op == "<=":
            mask &= (data[col] <= value)
        elif op == ">":
            mask &= (data[col] > value)
        elif op == ">=":
            mask &= (data[col] >= value)
    return data[mask]

# Main function to plot FITS data for each event
def plot_fits_data_eachevent(file_names, x_col, x_hdus, y_cols, y_hdus, y_scales, title, outfname, filters=None,
                              plotflag=False, markers="o", debug=True, markersize=1, gtifiles=None, n=10, use_sort=True, use_colorsort=False):
    """
    Function to plot each event's data from FITS files. Supports filtering and scaling.
    """
    # Loop over all FITS files
    for file_name in file_names:
        # Open the FITS file
        with fits.open(file_name) as hdul:
            print(f"..... {file_name} is opened.")
            # Extract header information
            header = hdul[0].header
            obsid = header["OBS_ID"]
            target = header["OBJECT"]
            dateobs = header["DATE-OBS"]
            
            # Apply filters if any
            x_data = {}  # Initialize dictionary for x-axis data
            dtimelist = []  # List to store datetime values

            # Loop through y_cols and x_hdus to extract data for each column
            for ycol, xhdu in zip(y_cols, x_hdus):
                print(f"..... {x_col} is opened from HDU={xhdu}")
                data = hdul[xhdu].data
                print("len(data) = ", len(data))
                
                # Apply filters if defined
                if filters:
                    print("..... filters applied")
                    data = apply_filters(data, filters)
                
                # Store the x data
                x_data[ycol] = data[x_col]

                # If the x column is 'TIME', convert it to datetime
                if x_col == 'TIME':
                    time = x_data[ycol]
                    dtime = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in time])
                    x_data['datetime'] = dtime
                    dtimelist.append(dtime)

                if use_sort:
                    # Sort x_data based on x_col in descending order
                    sorted_indices = np.argsort(x_data[ycol])[::-1]  # Sort in descending order
                    x_data[ycol] = x_data[ycol][sorted_indices]

            # Extract y data for each y column
            y_data = {}
            for ycol, yhdu in zip(y_cols, y_hdus):
                print(f"..... {ycol} is opened from HDU={yhdu}")
                data = hdul[yhdu].data
                if filters:
                    print("..... filters applied")
                    data = apply_filters(data, filters)
                y_data[ycol] = data[ycol]

                if use_sort:
                    # Sort x_data based on x_col in descending order
                    sorted_indices = np.argsort(data[x_col])[::-1]  # Sort in descending order
                    y_data[ycol] = y_data[ycol][sorted_indices]                    

            # Print data lengths for debugging
            print("len(data) = ", len(data))
            print("len(x_data) = ", len(x_data[ycol]))
#            print("y_data = ", y_data)

            # Get the number of events (rows in the data)
            event_number = len(x_data[y_cols[0]])

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))

            # Use the "cool" colormap for event coloring
            cmap = plt.get_cmap("hsv")
            color_norm = plt.Normalize(vmin=0, vmax=event_number - 1)

            # Loop over each event and plot the data
            for i in range(event_number):
                one_set = []  # List to store one event's data
                x_one = x_data[y_cols[0]][i]
                min_val, max_val = default_y_ranges[x_col]
                scaled_val = manual_scale(x_one, min_val, max_val)
                one_set.append(scaled_val)
                for ycol in y_cols:
#                    print("y_data[ycol] = ", ycol, i, y_data[ycol][i])
                    min_val, max_val = default_y_ranges[ycol]
                    scaled_val = manual_scale(y_data[ycol][i], min_val, max_val)
                    one_set.append(scaled_val)
#                print("one_set = ", one_set)
                # Get color for the current event from the "cool" colormap
                color = cmap(color_norm(i)) if use_colorsort else None
                ax.plot(range(len(one_set)), one_set, 'o-', label=f"Event {i}", alpha=0.4, color=color, linewidth=0.4)

            # Customize x-axis labels
            x_labels = [x_col] + y_cols  # Combine x_col and y_cols for labels
            ax.set_xticks(range(len(x_labels)))  # Set positions for labels
            ax.set_xticklabels(x_labels)  # Set the labels themselves

            # Remove unnecessary borders
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            # Add auxiliary y-axes for each y-column
            for j, ycol in enumerate(x_labels):
                y_min, y_max = default_y_ranges[ycol]
                ticks_orig = generate_ticks(y_min, y_max, n)  # Generate ticks for this range
                ticks_orig = y_ticks_dict.get(ycol, ticks_orig)  # Use custom ticks if defined
                ticks_scaled = [(t - y_min) / (y_max - y_min) for t in ticks_orig]  # Scale ticks to [0, 1]

                # Create a new twin axis
                ax_twin = ax.twinx()
                ax_twin.set_frame_on(True)
                ax_twin.set_yticks(ticks_scaled)
                ax_twin.set_yticklabels([f"{int(t)}" for t in ticks_orig])  # Display ticks as integers
                ax_twin.spines['right'].set_position(('axes', j / len(x_labels)))  # Position of twin axis
                ax_twin.set_ylabel(ycol, rotation=270, labelpad=15, fontsize=9)

                # Customize appearance of the twin axis
                ax_twin.spines['right'].set_color('gray')
                ax_twin.spines['right'].set_linewidth(1)
                ax_twin.spines['right'].set_alpha(0.5)
                ax_twin.spines['right'].set_linestyle('--')  # Dotted line style for aesthetic

                # Remove unnecessary borders from the twin axis
                ax_twin.spines['top'].set_visible(False)
                ax_twin.spines['left'].set_visible(False)
                ax_twin.spines['bottom'].set_visible(False)
                ax_twin.tick_params(axis='y', labelsize=9)  # Set y-axis tick font size

            # Remove ticks on the main y-axis
            ax.set_yticks([])

            # Add labels and titles
            ax.set_xlabel(f'COLUMN')
            plt.suptitle(f"{title}")
            plt.figtext(0.05, 0.02, f"OBSID={obsid} {target} {dateobs} # {len(x_data[y_cols[0]])}", fontsize=8, color="gray")
            
            # Layout adjustments
#            plt.tight_layout()
            ax.set_ylim(0, 1)
            ax.set_xlim(0 - 0.05, len(x_labels) + 0.05)
            plt.subplots_adjust(hspace=0, top=0.9)  # Adjust space between plots and top margin

            # Save the plot to a file
            plt.savefig(outfname)
            print(f".....{outfname} is saved.")

            # Display the plot if requested
            if plotflag:
                plt.show()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='This program is used to check deltaT distribution for each pixel',
        epilog='''
            Example:
            resolve_util_fploteve.py xa000114000rsl_p0px1000_cl.evt TIME TRIG_LP,WFRB_WRITE_LP,WFRB_SAMPLE_CNT -f "PIXEL==9" -p
            resolve_util_fploteve.y xa300049010rsl_p0px3000_uf_prevnext_cutclgti.fits PHA 1,1 PI,EPI 1,1 --plot --filters itype==0
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("file_names", type=str, help="List of Path to the FITS file")
    parser.add_argument("x_col", type=str, help="Column name for the x-axis")
    parser.add_argument('x_hdus', type=str, help='List of Number of FITS HDU for X')
    parser.add_argument("y_cols", type=str, help="Comma-separated column names for the y-axis")
    parser.add_argument('y_hdus', type=str, help='List of Number of FITS HDU for Y')
    parser.add_argument("--filters", '-f', type=str, help="Comma-separated filter conditions", default="")
    parser.add_argument('--plot', '-p', action='store_true', default=False, help='Flag to display plot')
    parser.add_argument("--markers", '-m', type=str, help="Marker type", default="o")
    parser.add_argument("--markersize", '-k', type=float, help="Marker size", default=1)
    parser.add_argument("--outname", "-o", type=str, help="Output file name tag", default="_p_")
    parser.add_argument('--use_sort', '-s', action='store_false', default=True, help='Flag to use sort')
    parser.add_argument('--use_colorsort', '-c', action='store_true', default=False, help='Flag to use color for sort')
    # Argument for y_ranges (as a string that can be converted into a dictionary)
    parser.add_argument("--y_ranges", type=str, help="Dictionary of y_ranges, e.g., '{\"PHA\": (0, 16383), \"RISE_TIME\": (0, 255)}'")

    # Parse arguments and run the plotting function
    args = parser.parse_args()
    outname = args.outname
    file_names = [_ for _ in args.file_names.split(",")]
    print(f'file_names = {file_names}')
    
    x_hdus = [int(_) for _ in args.x_hdus.split(",")]
    print(f'x_hdus = {x_hdus}')
    y_hdus = [int(_) for _ in args.y_hdus.split(",")]
    print(f'y_hdus = {y_hdus}')

    y_cols = args.y_cols.split(",")
    filter_conditions = parse_filter_conditions(args.filters) if args.filters else None    
    title = f"{args.file_names} : filtered with {args.filters}"
    outfname = "fplot_" + args.file_names.replace(",", "_").replace(".", outname) + ".png"

    # If y_ranges argument is provided, update the default dictionary with the new values
    if args.y_ranges:
        user_y_ranges = ast.literal_eval(args.y_ranges)  # Convert string to dictionary
        # Update the default dictionary with user-provided values
        default_y_ranges.update(user_y_ranges)

    # Print out the final y_ranges (either default or updated)
    print(f"Final y_ranges: {default_y_ranges}")

    plot_fits_data_eachevent(file_names, args.x_col, x_hdus, y_cols, y_hdus, None, title, outfname,
                             filters=filter_conditions, plotflag=args.plot, markers=args.markers, markersize=args.markersize,
                             use_sort = args.use_sort, use_colorsort=args.use_colorsort)
