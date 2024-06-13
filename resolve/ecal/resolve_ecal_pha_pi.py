#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import sys
import csv

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

def dump_to_csv(x_data, y_data, x_col, y_cols, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = [x_col]*len(y_cols) + y_cols
        writer.writerow(header)
        for i in range(len(next(iter(x_data.values())))):  # Iterate through the first x_data list length
            row = [x_data[y_col][i] for y_col in y_cols] + [y_data[y_col][i] for y_col in y_cols]
            writer.writerow(row)
    print(f"Data dumped to {csv_filename}")

def plot_fits_data(file_names, x_col, x_hdus, y_cols, y_hdus, y_scales, title, outfname, filters=None, plotflag=False, markers="o", debug=True, markersize=1, gtifiles=None):
    num_plots = len(y_cols)
    fig_ysize = 8
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, fig_ysize), sharex=True)
    all_x_data = {}
    all_y_data = {}

    for file_name in file_names:
        with fits.open(file_name) as hdul:
            print(f"..... {file_name} is opened.")
            x_data = {}
            y_data = {}
            for ycol, xhdu, yhdu in zip(y_cols, x_hdus, y_hdus):
                data = hdul[xhdu].data
                if filters:
                    data = apply_filters(data, filters)
                x_data[ycol] = data[x_col]

                data = hdul[yhdu].data
                if filters:
                    data = apply_filters(data, filters)
                y_data[ycol] = data[ycol]

            all_x_data.update(x_data)
            all_y_data.update(y_data)

            if num_plots == 1:
                axs = [axs]  # Ensure axs is always a list for consistency

            for ax, y_col, yscale in zip(axs, y_cols, y_scales):
                ax.plot(x_data[y_col], y_data[y_col], markers, label=y_col, markersize=markersize)
                ax.set_ylabel(y_col)
                ax.set_yscale(yscale)
                ax.legend()

            axs[-1].set_xlabel(x_col)
            plt.suptitle(title)

    plt.tight_layout()
    plt.savefig(outfname)
    print(f".....{outfname} is saved.")
    if plotflag:
        plt.show()
    
    csv_filename = outfname.replace(".png", ".csv")
    dump_to_csv(all_x_data, all_y_data, x_col, y_cols, csv_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This program is used to check deltaT distribution for each pixel',
        epilog='''
        Example:
        resolve_util_fplot.py xa000114000rsl_p0px1000_cl.evt TIME TRIG_LP,WFRB_WRITE_LP,WFRB_SAMPLE_CNT -f "PIXEL==9" -p
        resolve_util_fplot.py xa300049010rsl_p0px3000_uf_prevnext_cutclgti.fits PHA 1,1 PI,EPI 1,1 --plot --filters itype==0
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file_names", type=str, help="List of Path to the FITS file")
    parser.add_argument("x_col", type=str, help="Column name for the x-axis")
    parser.add_argument('x_hdus', type=str, help='List of Number of FITS HDU for X')
    parser.add_argument("y_cols", type=str, help="Comma-separated column names for the y-axis")
    parser.add_argument('y_hdus', type=str, help='List of Number of FITS HDU for Y')
    parser.add_argument("--filters", '-f', type=str, help="Comma-separated filter conditions in the format 'COLUMN==VALUE'", default="")
    parser.add_argument('--plot', '-p', action='store_true', default=False, help='Flag to display plot')
    parser.add_argument("--markers", '-m', type=str, help="marker type", default="o")
    parser.add_argument("--markersize", '-k', type=float, help="marker size", default=1)
    parser.add_argument("--y_cols_scale", '-s', type=str, help="Comma-separated y-axis scales", default=None)
    parser.add_argument("--gtifiles", type=str, help="Comma-separated column names for gtifiles", default=None)
    parser.add_argument("--outname", '-o', type=str, help="outputname", default=None)

    args = parser.parse_args()
    file_names = args.file_names.split(",")
    x_hdus = [int(x) for x in args.x_hdus.split(",")]
    y_hdus = [int(y) for y in args.y_hdus.split(",")]
    y_cols = args.y_cols.split(",")
    y_scales = args.y_cols_scale.split(",") if args.y_cols_scale else ["linear"] * len(y_cols)

    filter_conditions = parse_filter_conditions(args.filters) if args.filters else None
    title = f"{args.file_names} : filtered with {args.filters}"
    if args.outname == None:
        outfname = "fplot_" + args.file_names.replace(".evt", "_p_") + ".png"
    else:
        outfname = "fplot_" + args.file_names.replace(".evt", "_p" + args.outname) + ".png"

    plot_fits_data(file_names, args.x_col, x_hdus, y_cols, y_hdus, y_scales, title, outfname, filters=filter_conditions, plotflag=args.plot, markers=args.markers, markersize=args.markersize, gtifiles=args.gtifiles)
