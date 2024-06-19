#!/usr/bin/env python

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

def get_file_list(input_arg):
    # Check if input_arg is a file list indicator
    print(input_arg, input_arg)
    if input_arg.startswith('@'):
        file_list_path = input_arg[1:]
        with open(file_list_path, 'r') as file:
            file_list = [line.strip() for line in file.readlines()]
    else:
        # Assume it's a comma-separated list of file names or a single file name
        file_list = input_arg.split(',')
    
    return file_list

# Define a function to compute a fast light curve
def fast_lc(tstart, tstop, binsize, x):
    """Compute a fast light curve using a given time series."""
    times = np.arange(tstart, tstop, binsize)[:-1]
    n_bins = len(times)

    # Initialize arrays for the light curve data
    x_lc, y_lc = np.zeros(n_bins), np.zeros(n_bins)
    x_err = np.full(n_bins, 0.5 * binsize)
    y_err = np.zeros(n_bins)

    # Ensure the time array is sorted
    x = np.sort(x)

    # Compute counts for each bin
    for i, t in enumerate(times):
        start, end = np.searchsorted(x, [t, t + binsize])
        count = end - start

        x_lc[i] = t + 0.5 * binsize
        y_lc[i] = count / binsize
        y_err[i] = np.sqrt(count) / binsize

    return x_lc, x_err, y_lc, y_err


def plot_fits_data(gtifiles, evtfiles, title, outfname, \
                      plotflag = False, markers = "o", debug=True, markersize = 1, timebinsize = 100., usetime = False):

    # GTI データを格納する辞書
    gtidic = {}
    index = 0
    objlist = []

    for gtifile in gtifiles:
        with fits.open(gtifile) as hdulist:
            for hdu in hdulist:
                if 'GTI' in hdu.name:
                    print(f"..... {hdu.name} is opened.")
                    if "OBJECT" in hdu.header:
                        obj = hdu.header["OBJECT"]
                    else:
                    	obj ="N/A"                    
                    objlist.append(obj)

                    start = hdu.data["START"]
                    stop = hdu.data["STOP"]
                    # 辞書にデータを格納
                    gtidic[index] = {
                        'file_name': gtifile,
                        'hdu_name': hdu.name,
                        'start': start,
                        'stop': stop
                    }
                    index += 1

    # make lightcurves 
    lcdic = {}
    lcindex = 0
    if evtfiles == None:
        pass
    else:
        for evtfile in evtfiles:
            with fits.open(evtfile) as hdulist:
                for hdu in hdulist:
                    if 'EVENTS' in hdu.name:	
                        time = hdu.data["TIME"]        
                        if 'ITYPE' in hdu.columns.names:	
	                        itype = hdu.data["ITYPE"]        
	                        cutid = np.where(itype<5)[0] # Hp, Mp, Ms, Lp, Ls
                        if 'AC_ITYPE' in hdu.columns.names:	
	                        itype = hdu.data["AC_ITYPE"]        
	                        cutid = np.where(itype==0)[0] # 0:AC, 1:BL, 2:EL, 3:PE
                        time = time[cutid]
                        itype = itype[cutid]
                        x_lc, x_err, y_lc, y_err = fast_lc(time[0], time[-1], timebinsize, time)
                        zcutid = np.where(y_lc > 0)[0]
                        x_lc  = x_lc[zcutid]
                        x_err = x_err[zcutid]
                        y_lc  = y_lc[zcutid]
                        y_err = y_err[zcutid]           
                        dtime_lc = [REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(date_sec)) for date_sec in x_lc]
                        lcdic[lcindex] = {
                        'file_name': evtfile,
                        'datetime': dtime_lc,
                        'y_lc': y_lc,
                        'y_err': y_err
                        }
                        lcindex += 1


    fig, axs = plt.subplots(1, 1, figsize=(12, 7))
    plt.subplots_adjust(right=0.75)  # make the right space bigger
    colors = plt.cm.viridis(np.linspace(0, 1, index))    

    # 凡例用のリストを初期化
    legend_elements = []

    # gtidic の内容を表示
    for idx, data in gtidic.items():
        print(f"Index: {idx}, File: {data['file_name']}, HDU: {data['hdu_name']}, START: {data['start']}, STOP: {data['stop']}")
        pretime = 0
        for i, (s, e) in enumerate(zip(data['start'], data['stop'])): 	      
            if i == 0:
                pretime = e
            print(".... debug :", s, e, e-s, s-pretime)

        yval = 1 - (1/index) * idx
        sdate = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in data['start']])
        edate = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in data['stop']])

        if usetime:
            for s,e in zip(data['start'], data['stop']):
                axs.plot([s, e], [yval,yval], marker='o', ms=2, color=colors[idx],alpha=0.8)
        else:
            for s,e in zip(sdate, edate):
                axs.plot([s, e], [yval,yval], marker='o', ms=2, color=colors[idx],alpha=0.8)

        shortfname = os.path.basename(data['file_name'])
        # 凡例要素を追加
        legend_elements.append(Line2D([0], [0], color=colors[idx], marker=markers, label=shortfname+":"+data['hdu_name']))

    axs.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=6)

    if evtfiles is not None:
        colors = plt.cm.viridis(np.linspace(0, 1, lcindex))    
        ax2 = axs.twinx()  # Create a second y-axis
        for idx, data in lcdic.items():
            print(f"Index: {idx}, File: {data['file_name']}, datetime: {data['datetime']}, y_lc: {data['y_lc']}, y_err: {data['y_err']}")
            shortfname = os.path.basename(data['file_name'])
            ax2.errorbar(data['datetime'], data['y_lc'], yerr=data['y_err'], fmt='.', ms=2, color=colors[idx], label=shortfname)
        # 凡例をプロット
        ax2.legend(bbox_to_anchor=(1.05, 0.5), loc='upper left', borderaxespad=0., fontsize=6)
        ax2.set_ylabel(f"c/s (binsize={timebinsize}s), ITYPE < 5")

    objset = set(objlist)
    plt.title(",".join(objset))
    axs.set_ylabel("a.u.")
    axs.set_xlabel("TIME")
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
        resolve_util_gtiplot.py xa300049010rsl_p0px3000_uf_prevnext_cutclgti.fits PHA 1,1 PI,EPI 1,1 --plot --filters itype==0
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)    
    parser.add_argument("gtifilelist", type=str, help="Comma-separated GTI file names or a file containing GTI file names.")
    parser.add_argument('--plot', '-p', action='store_true', default=False, help='Flag to display plot')
    parser.add_argument("--markers", '-m', type=str, help="marker type", default="o")
    parser.add_argument("--markersize", '-k', type=float, help="marker size", default=1)
    parser.add_argument("--y_cols_scale", '-s', type=str, help="Comma-separated column names for the y-axis",default=None)
    parser.add_argument("--evtfilelist", '-e', type=str, help="Comma-separated evt file names or a file containing evt file names.",default=None)

    args = parser.parse_args()
    gtifiles = get_file_list(args.gtifilelist)
    print(f'gtifiles = {gtifiles}')        
    gtifiles_shortname = [os.path.basename(_) for _ in gtifiles]
    title = ",".join(gtifiles_shortname)

    if args.evtfilelist == None:
        evtfiles = None
    else:
        evtfiles = get_file_list(args.evtfilelist)
    print(f'evtfiles = {evtfiles}')        

    outfname = "gtiplot_" + ("_".join(gtifiles_shortname)).replace(".","_p_") + ".png"

    plot_fits_data(gtifiles, evtfiles, title, outfname, plotflag = args.plot, markers = args.markers, markersize = args.markersize)
