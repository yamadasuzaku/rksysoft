#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import sys
from astropy.time import Time
import datetime

params = {'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

from astropy.io import fits
import numpy as np

# Define global variables
MJD_REFERENCE_DAY = 58484
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')  # Reference time in Modified Julian Day

import re

def parse_filter_conditions(conditions):
    filters = []
    # '!=' を含めた正規表現パターンに変更
    condition_pattern = re.compile(r"(.*?)(==|<=|>=|<|>|!=)(.*)")
    for condition in conditions.split(","):
        match = condition_pattern.match(condition.strip())
        if match:
            col, op, value = match.groups()
            filters.append((col.strip(), op, value.strip()))
        else:
            raise ValueError(f"Invalid filter condition: {condition}")
    return filters

def apply_filters(data, filters):
    mask = np.ones(len(data), dtype=bool)
    for col, op, value in filters:

        # STATUSビット指定かどうか判定
        m = re.match(r"STATUS(\d\d)", col)
        if m:
            bit_index = int(m.group(1))
            status_bits = data["STATUS"]  # shape: (N, 16)
            if not (0 <= bit_index < 16):
                raise ValueError(f"Bit index {bit_index} out of range for STATUS (0–15)")
            # b00, b01などを整数に変換
            if value.startswith("b"):
                val_int = int(value[1:], 2)
            else:
                val_int = int(value)
            target_bit = status_bits[:, bit_index]

            if op == "==":
                mask &= (target_bit == val_int)
            elif op == "!=":
                mask &= (target_bit != val_int)
            else:
                raise ValueError(f"Unsupported operator for STATUS filtering: {op}")

        else:
            value=float(value)
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

def plot_fits_data(file_names, x_col, x_hdus, y_cols, y_hdus, y_scales, title, outfname,
                   filters=None, plotflag=False, markers="o", debug=True, markersize=1,
                   gtifiles=None, yranges=None, xlim=None):
    # Open the FITS file

    num_plots = len(y_cols)
    fig_ysize = 8
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, fig_ysize), sharex=True)
    for file_name in file_names:
        # Create subplots
        with fits.open(file_name) as hdul:
            print(f"..... {file_name} is opened.")

            # obtain header info 
            header = hdul[0].header
            obsid = header["OBS_ID"]
            target = header["OBJECT"]
            dateobs = header["DATE-OBS"]

            # Apply filters if any

            # Get X data
            x_data = {}
            dtimelist = []
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
                    dtimelist.append(dtime)

            # Get Y data
            y_data = {}
            for ycol, yhdu in zip(y_cols, y_hdus):
                print(f"..... {ycol} is opened from HDU={yhdu}")
                data = hdul[yhdu].data
                if filters:
                    print("..... filters applied")
                    data = apply_filters(data, filters)

                if ycol == "STATUS":
                    # 各ビットの重み (MSBが左側: bit 15, ..., bit 0)
                    weights = 2**np.arange(15, -1, -1)  # array([32768, 16384, ..., 1])
                    # 各行を1つの整数に変換
                    status_int = np.sum(data["STATUS"] * weights, axis=1).astype(np.uint16)  # shape (N,)
                    y_data[ycol] = status_int
                else:
                    y_data[ycol] = data[ycol]

            if num_plots == 1:
                axs = [axs]  # Ensure axs is always a list for consistency

            for i, (ax, y_col, yscale) in enumerate(zip(axs, y_cols, y_scales)):
                ax.plot(x_data[y_col], y_data[y_col], markers, label=y_col, markersize=markersize)
                ax.set_ylabel(y_col)
                ax.set_yscale(yscale)
                ax.legend()

                # y軸の範囲を設定
                if yranges and i * 2 + 1 < len(yranges):
                    ymin = yranges[i * 2]
                    ymax = yranges[i * 2 + 1]
                    if ymin != 'auto' and ymax != 'auto':
                        ax.set_ylim(float(ymin), float(ymax))
                    elif ymin != 'auto':
                        ax.set_ylim(bottom=float(ymin))
                    elif ymax != 'auto':
                        ax.set_ylim(top=float(ymax))

                if xlim:
                    if 'xmin' in xlim and 'xmax' in xlim:
                        ax.set_xlim(xlim['xmin'], xlim['xmax'])
                    elif 'xmin' in xlim:
                        ax.set_xlim(left=xlim['xmin'])
                    elif 'xmax' in xlim:
                        ax.set_xlim(right=xlim['xmax'])


            axs[-1].set_xlabel(x_col)  # Set x-axis label only on the bottom plot

            if x_col == 'TIME':
                ax2 = axs[0].twiny()
                ax2.plot(dtimelist[0], y_data[y_cols[0]], markers, markersize=0)  # Hide the second plot
                ax2.set_xlabel('Date')
                ax2.xaxis.set_label_position('top')
                ax2.xaxis.set_ticks_position('top')

            plt.suptitle(f"{title}")

    print(f"OBSID={obsid} {target} {dateobs} #={len(data)}")
    plt.figtext(0.05,0.02,f"OBSID={obsid} {target} {dateobs} #={len(data)}", fontsize=8,color="gray")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, top=0.9)  # Adjust space between plots and top margin
#    fig.autofmt_xdate()  # does not work for tile plots
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
    parser.add_argument("--filters", '-f', type=str, help="Comma-separated filter conditions in the format 'COLUMN==VALUE' or 'COLUMN1<VALUE,COLUMN2>=VALUE'", default="")
    parser.add_argument('--plot', '-p', action='store_true', default=False, help='Flag to display plot')
    parser.add_argument("--markers", '-m', type=str, help="marker type", default="o")
    parser.add_argument("--markersize", '-k', type=float, help="marker size", default=1)
    parser.add_argument("--y_cols_scale", '-s', type=str, help="Comma-separated column names for the y-axis", default=None)
    parser.add_argument("--gtifiles", type=str, help="Comma-separated column names for gtifiles", default=None)
    parser.add_argument("--outname", "-o", type=str, help="outputfile name tag", default="_p_")
    parser.add_argument("--yrange", "-yr", type=str, help="Comma-separated ymin,ymax values or 'auto' for each y axis", default=None)
    parser.add_argument("--xrange", "-xr", type=str, help="x-axis range as 'min,max' or use 'auto' for autoscale", default=None)

    args = parser.parse_args()
    outname = args.outname
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

    if args.yrange:
        yranges = args.yrange.split(",")
        # 長さが2*num_plotsになるよう調整する（autoを補う）
        if len(yranges) % 2 != 0:
            raise ValueError("Invalid --yrange format. Provide even number of entries (min,max per plot).")
    else:
        yranges = None

    if args.xrange:
        xrange_parts = args.xrange.split(",")
        if len(xrange_parts) != 2:
            raise ValueError("Invalid --xrange format. Use 'min,max' or 'auto,auto'")
        xmin, xmax = xrange_parts
        xlim = {}
        if xmin != 'auto':
            xlim['xmin'] = float(xmin)
        if xmax != 'auto':
            xlim['xmax'] = float(xmax)
    else:
        xlim = None


    print(f'x_col = {args.x_col}')
    print(f'y_cols = {y_cols}')
    print(f'y_scales = {y_scales}')
    print(f'y_hdus = {y_hdus}')

    filter_conditions = parse_filter_conditions(args.filters) if args.filters else None    

    title = f"{args.file_names} : \nfiltered {args.filters}"
    outfname = "fplot_" + args.file_names.replace(",", "_").replace(".", outname) + ".png"

    plot_fits_data(file_names, args.x_col, x_hdus, y_cols, y_hdus, y_scales, title, outfname,
               filters=filter_conditions, plotflag=args.plot, markers=args.markers,
               markersize=args.markersize, gtifiles=args.gtifiles, yranges=yranges, xlim=xlim)
