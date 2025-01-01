#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.time import Time
import datetime
import argparse
import sys
from matplotlib.colors import Normalize
import matplotlib.dates as mdates ######################################追加
import matplotlib.cm as cm
import os
from PIL import Image

# Constants
MJD_REFERENCE_DAY = 58484
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')
TIME_50MK = 150526800.0

# Plotting Configuration
plt.rcParams['font.family'] = 'serif'
usercmap = plt.get_cmap('jet')
cNorm = Normalize(vmin=0, vmax=35)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=usercmap)

# Type Information
itypename = [0, 1, 2, 3, 4]
typename = ["Hp", "Mp", "Ms", "Lp", "Ls"]
icol = ["r", "b", "c", "m", "y"]
ishape = [".", "s", "o", "*", "x"]

def ev_to_pi(ev):
    """Convert energy in eV to PI units."""
    return (ev - 0.5) * 2

def pi_to_ev(pi):
    """Convert PI units to energy in eV."""
    return pi * 0.5 + 0.5

def create_date_time_funcs(base_date, unit='seconds'):
    # https://qiita.com/yusuke_s_yusuke/items/9563f2484fdac29031d8
    base_date_num = mdates.date2num(base_date)  # 基準日付を内部形式に変換

    def date_to_time(date):
        """Convert matplotlib date to elapsed time."""
        elapsed_days = date - base_date_num
        if unit == 'seconds':
            elapsed_time = elapsed_days * 24 * 60 * 60
        elif unit == 'minutes':
            elapsed_time = elapsed_days * 24 * 60
        elif unit == 'hours':
            elapsed_time = elapsed_days * 24
        elif unit == 'days':
            elapsed_time = elapsed_days
        elif unit == 'years':
            elapsed_time = elapsed_days / 365.25  # 1年を約365.25日と定義
        else:
            raise ValueError("Invalid unit. Choose from 'seconds', 'minutes', 'hours', 'days', 'years'.")
        return elapsed_time

    def time_to_date(elapsed_time):
        """Convert elapsed time to matplotlib date."""
        if unit == 'seconds':
            elapsed_days = elapsed_time / (24 * 60 * 60)
        elif unit == 'minutes':
            elapsed_days = elapsed_time / (24 * 60)
        elif unit == 'hours':
            elapsed_days = elapsed_time / 24
        elif unit == 'days':
            elapsed_days = elapsed_time
        elif unit == 'years':
            elapsed_days = elapsed_time * 365.25  # 1年を約365.25日と定義
        else:
            raise ValueError("Invalid unit. Choose from 'seconds', 'minutes', 'hours', 'days', 'years'.")
        date = elapsed_days + base_date_num
        return date

    return date_to_time, time_to_date

# Define the energy range
emin, emax = 0, 20000  # Energy range in eV
pimin, pimax = ev_to_pi(emin), ev_to_pi(emax)
rebin = 10
binnum = int((pimax - pimin) / rebin)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot gain history from a FITS file.')
    parser.add_argument('filename', help='The name of the FITS file to process.')
    parser.add_argument('--hk1', '-k', type=str, help='hk1file', default=None)
    parser.add_argument('--reverse_axes', '-r', action='store_false', help='Reverse x-axis to show only time instead of date and time')
    parser.add_argument('--show', '-s', action='store_true', help='plt.show()を実行するかどうか。defaultはplotしない。')    
    parser.add_argument('--paper', '-p', action='store_true', help='論文モード (eps保存、タイトルにファイル名非表示)')    
    return parser.parse_args()

def open_fits_data(fname):
    if not os.path.isfile(fname):
        print("ERROR: File not found", fname)
        sys.exit(1)
    return fits.open(fname)[1].data

def process_data(data):
    time = data["TIME"]
    additional_columns = [data[col] for col in ["PIXEL", "TEMP_FIT", "CHISQ", "NEVENT"]]
    sortid = np.argsort(time)
    sorted_columns = [col[sortid] for col in [time] + additional_columns]
    dtime = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in sorted_columns[0]])
    print(f"data from {dtime[0]} --> {dtime[-1]}")
    return sorted_columns, dtime

def save_pixel_data_to_csv(pixel, time, temp_fit):
    # Create output directory if it doesn't exist
    output_dir = "ghf_check"
    os.makedirs(output_dir, exist_ok=True)

    # Save data for each pixel
    for pixel_ in np.arange(36):
        pixelcut = (pixel == pixel_)
        px_time = time[pixelcut]
        px_temp_fit = temp_fit[pixelcut]
        if len(px_time) == 0:
            print("warning: data is empty for pixel =", pixel_)
            continue

        df = pd.DataFrame({
            'px_time': px_time,
            'px_temp_fit': px_temp_fit
        })
        csv_filename = os.path.join(output_dir, f"pixel_{pixel_:02d}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Data for pixel {pixel_} saved to {csv_filename}")

def plot_ghf(time, dtime, pixel, temp_fit, reverse_axes=False, hk1=None, outfname="mkpi.png", title="test", show=False, paper=False):
    k2mk = 1e3 
    if paper:
        fig, ax1 = plt.subplots(figsize=(9, 7))
    else:
        fig, ax1 = plt.subplots(figsize=(11, 7))        
    plt.subplots_adjust(right=0.8)  # make the right space bigger

    if reverse_axes:
        ax1.set_xlabel('Date')
        ax1.set_xlim(dtime[0], dtime[-1])
        # 単位を指定して変換関数を作成
        unit = 'seconds'  # ここで単位を変更できる（'seconds', 'minutes', 'hours', 'days', 'years'）
        dtime_to_time, time_to_dtime = create_date_time_funcs(dtime[0], unit)        
        ax2 = ax1.secondary_xaxis('top', functions=(dtime_to_time, time_to_dtime)) ###############################追加
        ax2.set_xlabel("Elapsed Time (s) from " + dtime[0].strftime('%Y/%m/%d %H:%M:%S')) ###############################追加
        x_data = dtime
    else:
        ax1.set_xlabel("Elapsed Time (s) from " + str(dtime[0]))
        ax1.set_xlim(time[0], time[-1])
        x_data = time

    ax1.set_ylabel("Effective Temperature (mK)")
#    ax1.grid(alpha=0.2)

    if not paper: 
        ax1.set_title(title)

    # Plot histograms for each pixel
    for pixel_ in np.arange(36):
        pixelcut = (pixel == pixel_)
        px_time = x_data[pixelcut]
        px_temp_fit = temp_fit[pixelcut]
        if len(px_time) == 0:
            print("warning: data is empty for pixel =", pixel_)
            continue

        color = scalarMap.to_rgba(pixel_)
        event_number = len(px_time)
        ax1.errorbar(px_time, px_temp_fit * k2mk, color=color, alpha=0.8, fmt=ishape[pixel_ % 5], label=f"P{pixel_} ({event_number})")
        ax1.errorbar(px_time, px_temp_fit * k2mk, color=color, alpha=0.2, fmt="-", label=None)

    ax1.legend(bbox_to_anchor=(1.1, 1.08), loc='upper left', borderaxespad=0., fontsize=8)

    if hk1 is not None:
        if not os.path.isfile(hk1):
            print("ERROR: hk1 File not found", hk1)
            sys.exit(1)
        hk1data = fits.open(hk1)[4].data
        hk1time = hk1data["TIME"]
        hk1fwpos = hk1data["FWE_FW_POSITION1_CAL"]
        ax3 = ax1.twinx()  # Secondary y-axis
        ax3.set_ylabel("FW Position (degree)")
        if reverse_axes:
            dtime_hk1time = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in hk1time])            
            if paper:
                hk1fwpos_30open = hk1fwpos[ (hk1fwpos > 29) &  (hk1fwpos < 31)] 
                hk1fwpos_150ND = hk1fwpos[ (hk1fwpos > 149) &  (hk1fwpos < 151)] 
                hk1fwpos_330Fe55 = hk1fwpos[(hk1fwpos > 329.36) &  (hk1fwpos < 329.40)] 
#                ax3.plot(dtime_hk1time[(hk1fwpos > 28) &  (hk1fwpos < 32)], hk1fwpos_30open, 'y.', alpha=0.5, label="OPEN")
                ax3.plot(dtime_hk1time[(hk1fwpos > 149) &  (hk1fwpos < 151)][::40], hk1fwpos_150ND[::40], 'o', color="bisque",alpha=0.5, label="FW ND", ms=2)
                ax3.plot(dtime_hk1time[(hk1fwpos > 329.36) &  (hk1fwpos < 329.40)][::40], hk1fwpos_330Fe55[::40], 'o', color="springgreen",alpha=0.5, label="FW Fe55", ms=2)
                ax3.set_ylim(140,340)
            else:
                ax3.plot(dtime_hk1time, hk1fwpos, 'g-', alpha=0.5, label="FW Position")
        else:
            ax3.plot(hk1time, hk1fwpos, 'g-', alpha=0.5, label="FW Position")

#        ax3.legend(loc='upper right', fontsize=8)
        legend_ax3 = ax3.legend(bbox_to_anchor=(0.9, 1.01), loc='lower left', borderaxespad=0., fontsize=9)

    ofname = f"fig_{outfname}"
    plt.savefig(ofname)
    print(f"..... {ofname} is created.")

    if paper:
        # PNG を開いて PDF として保存
        pdfname = ofname.replace(".png",".pdf")
        image = Image.open(ofname)
        image.save(pdfname, "PDF")

        epsname = ofname.replace(".png",".eps")
        plt.savefig(epsname)

    if show:
        plt.show()
    
def main():
    args = parse_arguments()
    if not args.filename or (args.hk1 and not os.path.isfile(args.hk1)):
        print("Usage: resolve_ecal_plot_ghf_with_FWE.py <filename> [--hk1 <hk1file>] [--reverse_axes]")
        print("Example: resolve_ecal_plot_ghf_with_FWE.py xa300065010rsl_000_fe55.ghf --hk1 xa300065010rsl_a0.hk1 --reverse_axes")
        sys.exit(1)
    
    data = open_fits_data(args.filename)
    processed_data, dtime = process_data(data)
    time, pixel, temp_fit, chisq, nevent = processed_data  # data unpack

    # Save pixel data to CSV
    save_pixel_data_to_csv(pixel, time, temp_fit)

    plot_ghf(time, dtime, pixel, temp_fit, reverse_axes=args.reverse_axes, hk1=args.hk1, outfname=f"ql_plotghf_{args.filename.replace('.ghf', '').replace('ghf.gz', '')}.png", title=f"Gain history of {args.filename}", show=args.show, paper=args.paper)

if __name__ == "__main__":
    main()