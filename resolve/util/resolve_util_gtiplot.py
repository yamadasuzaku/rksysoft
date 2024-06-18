#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import sys
import datetime
import os 

params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

from astropy.io import fits
from astropy.time import Time
import numpy as np
from matplotlib.lines import Line2D

# Define global variables
MJD_REFERENCE_DAY = 58484
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')  # Reference time in Modified Julian Day

def plot_fits_data(file_names, title, outfname, \
                      plotflag = False, markers = "o", debug=True, markersize = 1):

    # GTI データを格納する辞書
    gtidic = {}
    index = 0

    for file_name in file_names:
        with fits.open(file_name) as hdulist:
            for hdu in hdulist:
                if 'GTI' in hdu.name:
                    print(f"..... {hdu.name} is opened.")
                    start = hdu.data["START"]
                    stop = hdu.data["STOP"]
                    # 辞書にデータを格納
                    gtidic[index] = {
                        'file_name': file_name,
                        'hdu_name': hdu.name,
                        'start': start,
                        'stop': stop
                    }
                    index += 1

    fig, axs = plt.subplots(1, 1, figsize=(12, 7))
    plt.subplots_adjust(right=0.75)  # make the right space bigger
    colors = plt.cm.viridis(np.linspace(0, 1, index))    

    # 凡例用のリストを初期化
    legend_elements = []

    # gtidic の内容を表示
    for idx, data in gtidic.items():
        print(f"Index: {idx}, File: {data['file_name']}, HDU: {data['hdu_name']}, START: {data['start']}, STOP: {data['stop']}")
        yval = 1 - (1/index) * idx
        sdate = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in data['start']])
        edate = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in data['stop']])
        for s,e in zip(sdate, edate):
            plt.plot([s, e], [yval,yval], marker='o', ms=2, color=colors[idx])
        shortfname = os.path.basename(data['file_name'])
#        plt.plot([], [], ' ', label=shortfname+":"+data['hdu_name'])
#        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=6)
        # 凡例要素を追加
        legend_elements.append(Line2D([0], [0], color=colors[idx], marker=markers, label=shortfname+":"+data['hdu_name']))
    # 凡例をプロット
    axs.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=6)

    plt.show()



    # for file_name in file_names:
    #     # Create subplots
    #     with fits.open(file_name) as hdulist:
    #         for hdu in hdulist:
    #             if 'GTI' in hdu.name:
    #                 print(f"..... {hdu.name} is opened.")
    #                 start = hdu.data["START"]
    #                 stop = hdu.data["START"]

    # # GTI データを格納する辞書
    # gtidic = {}
    # for file_name in file_names:
    #     with fits.open(file_name) as hdulist:
    #         for hdu in hdulist:
    #             if 'GTI' in hdu.name:
    #                 print(f"..... {hdu.name} is opened.")
    #                 start = hdu.data["START"]
    #                 stop = hdu.data["STOP"]
    #                 # 辞書にデータを格納
    #                 if file_name not in gtidic:
    #                     gtidic[file_name] = []
    #                 gtidic[file_name].append({
    #                     'hdu_name': hdu.name,
    #                     'start': start,
    #                     'stop': stop
    #                 })

    # # gtidic の内容を表示
    # for file_name, data in gtidic.items():
    #     for entry in data:
    #         print(f"File: {file_name}, HDU: {entry['hdu_name']}, START: {entry['start']}, STOP: {entry['stop']}")


    # for file_name in file_names:
    #     # Create subplots
    #     with fits.open(file_name) as hdulist:
    #         for hdu in hdulist:
    #             if 'GTI' in hdu.name:
    #                 print(f"..... {hdu.name} is opened.")
    #                 start = hdu.data["START"]
    #                 stop = hdu.data["START"]

    #                 for s,e in zip(start,stop):
    #                     plt.plot([s, e], [1, 1], marker='o')
    #                 plt.plot([], [], ' ', label=hdu.name)
    #                 plt.legend()

    #             else:
    #                 print(f"..... {hdu.name} is skipped, continue.")
    #                 continue
    # plt.show()
    #         # Apply filters if any
    #         # Get X data
    #         x_data = {}
    #         for ycol, xhdu in zip(y_cols,x_hdus):
    #             print(f"..... {x_col} is opened from HDU={xhdu}")
    #             data = hdul[xhdu].data
    #             if filters:
    #                 print("..... filters applied")
    #                 data = apply_filters(data, filters)
    #             x_data[ycol] = data[x_col]

    #         # Get Y data
    #         y_data = {}
    #         for ycol, yhdu in zip( y_cols, y_hdus):
    #             print(f"..... {ycol} is opened from HDU={yhdu}")
    #             data = hdul[yhdu].data
    #             if filters:
    #                 print("..... filters applied")
    #                 data = apply_filters(data, filters)
    #             y_data[ycol] = data[ycol]

    #         if num_plots == 1:
    #             axs = [axs]  # Ensure axs is always a list for consistency

    #         for ax, y_col, yscale in zip(axs, y_cols, y_scales):
    #             ax.plot(x_data[y_col], y_data[y_col], markers, label=y_col, markersize = markersize)
    #             ax.set_ylabel(y_col)
    #             ax.set_yscale(yscale)
    #             ax.legend()

    #         axs[-1].set_xlabel(x_col)  # Set x-axis label only on the bottom plot
    #         plt.suptitle(title)

    # plt.tight_layout()
    # plt.savefig(outfname)
    # print(f".....{outfname} is saved.")
    # if plotflag:
    #     plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='This program is used to check deltaT distribution for each pixel',
      epilog='''
        Example:
        resolve_util_gtiplot.py xa300049010rsl_p0px3000_uf_prevnext_cutclgti.fits PHA 1,1 PI,EPI 1,1 --plot --filters itype==0
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)    
    parser.add_argument("file_names", type=str, help="List of Path to the FITS file")
    parser.add_argument('--plot', '-p', action='store_true', default=False, help='Flag to display plot')
    parser.add_argument("--markers", '-m', type=str, help="marker type", default="o")
    parser.add_argument("--markersize", '-k', type=float, help="marker size", default=1)
    parser.add_argument("--y_cols_scale", '-s', type=str, help="Comma-separated column names for the y-axis",default=None)
    parser.add_argument("--gtifiles", type=str, help="Comma-separated column names for gtifiles",default=None)

    args = parser.parse_args()
    file_names = [_ for _ in args.file_names.split(",")]
    print(f'file_names = {file_names}')
    title = f"{args.file_names}"

    outfname = "gtiplot_" + args.file_names.replace(",","_").replace(".","_p_") + ".png"

    plot_fits_data(file_names, title, outfname, plotflag = args.plot, markers = args.markers, markersize = args.markersize)
