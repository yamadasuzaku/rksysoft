#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import sys
from astropy.time import Time
import datetime

params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

from astropy.io import fits
import numpy as np

# Define global variables
MJD_REFERENCE_DAY = 58484
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')  # Reference time in Modified Julian Day

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

def plot_fits_data(file_names, x_col, x_hdus, y_cols, y_hdus, y_scales, title, outfname, filters=None,\
                      plotflag = False, markers = "o", debug=True, markersize = 1, gtifiles = None):
    # Open the FITS file

    num_plots = len(y_cols)
    fig_ysize = 8
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, fig_ysize), sharex=True)

    for file_name in file_names:
        # Create subplots
        with fits.open(file_name) as hdul:
            print(f"..... {file_name} is opened.")
            # Apply filters if any

            # Get X data
            x_data = {}
            for ycol, xhdu in zip(y_cols, x_hdus):
                print(f"..... {x_col} is opened from HDU={xhdu}")
                data = hdul[xhdu].data
                if filters:
                    print("..... filters applied")
                    data = apply_filters(data, filters)
                x_data[ycol] = data[x_col]

                if x_col == 'TIME':
                    time = x_data[ycol]
                    dtime = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in time])
                    x_data['datetime'] = dtime

            # Get Y data
            y_data = {}
            for ycol, yhdu in zip(y_cols, y_hdus):
                print(f"..... {ycol} is opened from HDU={yhdu}")
                data = hdul[yhdu].data
                if filters:
                    print("..... filters applied")
                    data = apply_filters(data, filters)
                y_data[ycol] = data[ycol]

            if num_plots == 1:
                axs = [axs]  # Ensure axs is always a list for consistency

            for ax, y_col, yscale in zip(axs, y_cols, y_scales):
                ax.plot(x_data[y_col], y_data[y_col], markers, label=y_col, markersize=markersize)
                ax.set_ylabel(y_col)
                ax.set_yscale(yscale)
                ax.legend()

            axs[-1].set_xlabel(x_col)  # Set x-axis label only on the bottom plot

            if x_col == 'TIME':
                ax2 = axs[-1].twiny()
                ax2.plot(x_data['datetime'], y_data[y_cols[0]], markers, markersize=0)  # Hide the second plot
                ax2.set_xlabel('Date')
                ax2.xaxis.set_label_position('top')
                ax2.xaxis.set_ticks_position('top')
                fig.autofmt_xdate()

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
        resolve_util_fplot.py xa300049010rsl_p0px3000_uf_prevnext_cutclgti.fits PHA 1,1 PI,EPI 1,1 --plot --filters itype==0
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file_names", type=str, help="List of Path to the FITS file")
    parser.add_argument("x_col", type=str, help="Column name for the x-axis")
    parser.add_argument('x_hdus', type=str, help='List of Number of FITS HDU for X')
    parser.add_argument("y_cols", type=str, help="Comma-separated column names for the y-axis")
    parser.add_argument('y_hdus', type=str, help='List of Number of FITS HDU for Y')
    parser.add_argument("--filters", '-f', type=str, help="Comma-separated filter conditions in the format 'COLUMN==VALUE'", default="")
    parser.add_argument('--plot', '-p', action='store_true', default=False, help='Flag to display plot')
    parser.add_argument("--markers", '-m', type=str, help="marker type", default="o")
    parser.add_argument("--markersize", '-k', type=float, help="marker size", default=1)
    parser.add_argument("--y_cols_scale", '-s', type=str, help="Comma-separated column names for the y-axis", default=None)
    parser.add_argument("--gtifiles", type=str, help="Comma-separated column names for gtifiles", default=None)

    args = parser.parse_args()
    file_names = [_ for _ in args.file_names.split(",")]
    print(f'file_names = {file_names}')

    x_hdus = [int(_) for _ in args.x_hdus.split(",")]
    print(f'x_hdus = {x_hdus}')
    y_hdus = [int(_) for _ in args.y_hdus.split(",")]
    print(f'y_hdus = {y_hdus}')

    y_cols = args.y_cols.split(",")
    if args.y_cols_scale:
        y_scales = args.y_cols_scale.split(",")
    else:
        y_scales = ["linear" for _ in range(len(y_cols))]

    print(f'x_col = {args.x_col}')
    print(f'y_cols = {y_cols}')
    print(f'y_scales = {y_scales}')
    print(f'y_hdus = {y_hdus}')

    filter_conditions = parse_filter_conditions(args.filters) if args.filters else None
    title = f"{args.file_names} : filtered with {args.filters}"
    outfname = "fplot_" + args.file_names.replace(",", "_").replace(".", "_p_") + ".png"
    plot_fits_data(file_names, args.x_col, x_hdus, y_cols, y_hdus, y_scales, title, outfname,
                   filters=filter_conditions, plotflag=args.plot, markers=args.markers, markersize=args.markersize, gtifiles=args.gtifiles)
