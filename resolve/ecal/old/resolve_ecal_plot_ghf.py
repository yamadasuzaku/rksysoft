#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from astropy.time import Time
import datetime

# Constants
MJD_REFERENCE_DAY = 58484
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')

# Set plot parameters for consistent styling
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

# Maximum number of colors to be used in plots
CMAX = 20
colors = plt.cm.tab20(np.linspace(0, 1, CMAX))

def plot_ghf(ghf, dumpflag = False, plotflag=False, offset1 = 10, offset2 = 2.0):

    # FITSファイルを開く
    hdu = fits.open(ghf)

    # 必要なデータを抽出
    header = hdu[1].header
    obsid = header["OBS_ID"]
    dataobs = header["DATE-OBS"]

    data = hdu[1].data
    pixels = data["PIXEL"]
    times = data["TIME"]
    specs = data["SPECTRUM"]
    fitprofs = data["FITPROF"]
    binmeshs = data["BINMESH"]
    chisqs = data["CHISQ"]
    avgfits = data["AVGFIT"]
    shifts = data["SHIFT"]
    temp_fits = data["TEMP_FIT"]
    nevents = data["NEVENT"]
    exposures = data["exposure"]
    telapses = data["TELAPSE"]

    for pixel in range(36):
        cutid = np.where(pixels == pixel)[0]
        time = times[cutid]
        spec = specs[cutid]
        fitprof = fitprofs[cutid]
        binmesh = binmeshs[cutid]
        chisq = chisqs[cutid]
        avgfit = avgfits[cutid]
        shift = shifts[cutid]
        temp_fit = temp_fits[cutid]
        nevent = nevents[cutid]
        exposure = exposures[cutid]
        telapse = telapses[cutid]

        # プロット作成
        print(f"PIXEL{pixel:02d} (time from {time[0]})")        
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 7))
        plt.subplots_adjust(right=0.7)
        for i in range(len(spec)):
            color = colors[i % len(colors)]  # get color 
            # BINMESH > 0 のデータだけを抽出
            binm = binmesh[i]
            valid_indices = np.where(binm>0)[0]
            valid_binmesh = binmesh[i][valid_indices]
            valid_spec = spec[i][valid_indices]
            valid_fitprof = fitprof[i][valid_indices]
            # 比を計算
            ratio = np.abs(valid_spec / valid_fitprof)
            
            # 上部2/3のプロット
            fitinfo = fr"{time[i]-time[0]:0.1f}s, $\chi=${chisq[i]:0.1f} N={nevent[i]} Exp={exposure[i]:0.1f}"
            print(fitinfo)
            ax1.set_title(f"PIXEL{pixel:02d} time from {time[0]}")
            ax1.errorbar(valid_binmesh, valid_spec + offset1*i, color=color, yerr=np.sqrt(valid_spec), fmt=".", label=fitinfo, alpha=0.7)
            ax1.plot(valid_binmesh, valid_fitprof  + offset1*i, "-", color=color, label=None, alpha=0.7)
            ax1.set_ylabel('Intensity (counts/bin) + offset')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=8)
            # 下部1/3のプロット
            ax2.errorbar(valid_binmesh, ratio * (offset2*(i+1)), yerr=np.sqrt(valid_spec)/valid_fitprof, fmt=".", color=color, label='SPECTRUM / FITPROF', alpha=0.7)
            ax2.set_xlabel('BINMESH')
            ax2.set_ylabel('Ratio + offset')
            ax2.set_yscale('log')    
#            ax2.legend()
        # 図を保存
        plt.suptitle(f"{ghf} OBSID={obsid} {dataobs}")
        outfile=f'ecal_plot_ghf_pixel{pixel:02d}.png'
        plt.savefig(outfile)
        print(f"..... {outfile} is created.")        

        if plotflag:
            plt.show()
        plt.close(fig)

    # FITSファイルを閉じる
    hdu.close()


def main():
    """
    Main function to parse arguments and call the plot_data_6x6 function.
    """
    parser = argparse.ArgumentParser(
      description='This program is to plot pulserecord',
      epilog='''
        Example 1) just dump all fit results 
        resolve_ecal_plot_ghf.py xa300065010rsl_000_fe55.ghf
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('ghf', type=str, help='Input ghf file')
    parser.add_argument('--dumptext', action='store_true', help='Flag to dump x_time and pulse data to NPZ files')
    parser.add_argument('--plot', '-p', action='store_true', help='Flag to plot')

    args = parser.parse_args()
    
    plot_ghf(args.ghf, dumpflag = args.dumptext, plotflag=args.plot)

if __name__ == "__main__":
    main()
