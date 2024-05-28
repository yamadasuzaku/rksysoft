#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt

params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

from astropy.io import fits
import numpy as np

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

def plot_fits_data(file_name, x_col, y_cols, hdu, title, outfname, filters=None, plotflag = False, markers = "o"):
    # Open the FITS file
    with fits.open(file_name) as hdul:
        data = hdul[hdu].data  # Access the data from HDU 2

        # Apply filters if any
        if filters:
            data = apply_filters(data, filters)

        # Convert columns to numpy arrays for easier plotting
        x_data = data[x_col]
        y_data = {col: data[col] for col in y_cols}

        # Create subplots
        num_plots = len(y_cols)
        fig_ysize = 8
        fig, axs = plt.subplots(num_plots, 1, figsize=(12, fig_ysize), sharex=True)

        if num_plots == 1:
            axs = [axs]  # Ensure axs is always a list for consistency

        for ax, y_col in zip(axs, y_cols):
            ax.plot(x_data, y_data[y_col], markers, label=y_col)
#            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.legend()

        axs[-1].set_xlabel(x_col)  # Set x-axis label only on the bottom plot
        plt.suptitle(title)

        plt.tight_layout()
        plt.savefig(outfname)
        print(f".....{outfname} is saved.")
        if plotflag:
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='This program is used to check deltaT distribution for each pixel',
      epilog='''
        Example:
        resolve_util_fplot.py xa000114000rsl_p0px1000_cl.evt TIME TRIG_LP,WFRB_WRITE_LP,WFRB_SAMPLE_CNT -f "PIXEL==9" -p
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)    
    parser.add_argument("file_name", type=str, help="Path to the FITS file")
    parser.add_argument("x_col", type=str, help="Column name for the x-axis")
    parser.add_argument("y_cols", type=str, help="Comma-separated column names for the y-axis")
    parser.add_argument('--hdu', '-n', type=int, default=1, help='Number of FITS HDUe')
    parser.add_argument("--filters", '-f', type=str, help="Comma-separated filter conditions in the format 'COLUMN==VALUE'", default="")
    parser.add_argument('--plot', '-p', action='store_true', default=False, help='Flag to display plot')
    parser.add_argument("--markers", '-m', type=str, help="marker type", default="o")

    args = parser.parse_args()

    y_cols_list = args.y_cols.split(",")
    filter_conditions = parse_filter_conditions(args.filters) if args.filters else None
    title = f"{args.file_name} : filtered with {args.filters}"
    outfname = "fplot_" + args.file_name.split(".")[0] + ".png"
    plot_fits_data(args.file_name, args.x_col, y_cols_list, args.hdu, title, outfname, filter_conditions, args.plot, args.markers)
