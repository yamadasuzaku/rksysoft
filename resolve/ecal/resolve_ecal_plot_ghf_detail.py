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
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 7}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

# Maximum number of colors to be used in plots
CMAX = 20
colors = plt.cm.tab20(np.linspace(0, 1, CMAX))

def plot_ghf(ghf, dumpflag=False, plotflag=False, detailflag=False, selected_pixels=None, offset1=10, offset2=2.0):

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

    if selected_pixels is None:
        selected_pixels = range(36)

    for pixel in selected_pixels:
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

        if detailflag:
            for i in range(len(spec)):
                fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 7))
                plt.subplots_adjust(right=0.7)
                color = colors[i % len(colors)]
                binm = binmesh[i]
                valid_indices = np.where(binm > 0)[0]
                valid_binmesh = binmesh[i][valid_indices]
                valid_spec = spec[i][valid_indices]
                valid_fitprof = fitprof[i][valid_indices]
                ratio = np.abs(valid_spec / valid_fitprof)

                fitinfo = fr"{time[i] - time[0]:0.1f}s, $\chi=${chisq[i]:0.1f} N={nevent[i]} Exp={exposure[i]:0.1f}"
                ax1.set_title(f"PIXEL{pixel:02d} time from {time[0]}")
                ax1.errorbar(valid_binmesh, valid_spec, color=color, yerr=np.sqrt(valid_spec), fmt=".", label=fitinfo, alpha=0.7)
                ax1.plot(valid_binmesh, valid_fitprof, "-", color=color, label=None, alpha=0.7)
                ax1.set_ylabel('Intensity (counts/bin)')
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=8)

                ax2.errorbar(valid_binmesh, ratio, yerr=np.sqrt(valid_spec) / valid_fitprof, fmt=".", color=color, label='SPECTRUM / FITPROF', alpha=0.7)
                ax2.set_xlabel('BINMESH')
                ax2.set_ylabel('Ratio')
                ax2.set_yscale('log')

                # 詳細情報を図上に表示
                detail_text = (
                    f"Time: {time[i]}\n"
                    f"Chi-Square: {chisq[i]:0.4f}\n"
                    f"Avg Fit: {avgfit[i]:0.4f}\n"
                    f"Shift: {shift[i]:0.3f}\n"
                    f"Temp Fit: {temp_fit[i]:0.6f}\n"
                    f"Event Count: {nevent[i]}\n"
                    f"Exposure: {exposure[i]:0.3f}\n"
                    f"Telapse: {telapse[i]:0.3f}\n"
                )
                plt.gcf().text(0.75, 0.5, detail_text, fontsize=10, verticalalignment='center')

                plt.suptitle(f"{ghf} OBSID={obsid} {dataobs}")
                outfile = f'detail_plot_ghf_pixel{pixel:02d}_spec{i:02d}.png'
                plt.savefig(outfile)
                print(f"..... {outfile} is created.")

                if plotflag:
                    plt.show()
                plt.close(fig)
        else:
            # 全体のプロット作成（現状のスクリプトのまま）
            fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 7))
            plt.subplots_adjust(right=0.7)
            for i in range(len(spec)):
                color = colors[i % len(colors)]
                binm = binmesh[i]
                valid_indices = np.where(binm > 0)[0]
                valid_binmesh = binmesh[i][valid_indices]
                valid_spec = spec[i][valid_indices]
                valid_fitprof = fitprof[i][valid_indices]
                ratio = np.abs(valid_spec / valid_fitprof)

                fitinfo = fr"{time[i] - time[0]:0.1f}s, $\chi=${chisq[i]:0.1f} N={nevent[i]} Exp={exposure[i]:0.1f}"
                ax1.set_title(f"PIXEL{pixel:02d} time from {time[0]}")
                ax1.errorbar(valid_binmesh, valid_spec + offset1 * i, color=color, yerr=np.sqrt(valid_spec), fmt=".", label=fitinfo, alpha=0.7)
                ax1.plot(valid_binmesh, valid_fitprof + offset1 * i, "-", color=color, label=None, alpha=0.7)
                ax1.set_ylabel('Intensity (counts/bin) + offset')
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., handletextpad=0.5, labelspacing=0.5)

                ax2.errorbar(valid_binmesh, ratio * (offset2 * (i + 1)), yerr=np.sqrt(valid_spec) / valid_fitprof, fmt=".", color=color, label='SPECTRUM / FITPROF', alpha=0.7)
                ax2.set_xlabel('BINMESH')
                ax2.set_ylabel('Ratio + offset')
                ax2.set_yscale('log')

            plt.suptitle(f"{ghf} OBSID={obsid} {dataobs}")
            outfile = f'ecal_plot_ghf_pixel{pixel:02d}.png'
            plt.savefig(outfile)
            print(f"..... {outfile} is created.")

            if plotflag:
                plt.show()
            plt.close(fig)

    # FITSファイルを閉じる
    hdu.close()


def parse_pixel_range(pixel_range_str):
    if pixel_range_str is None:
        return None
    return [int(p) for p in pixel_range_str.split(',')]

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
    parser.add_argument('--detail', action='store_true', help='Flag for detailed plot')
    parser.add_argument('--pixels', type=str, help='Comma-separated list of pixel numbers to process')

    args = parser.parse_args()

    selected_pixels = parse_pixel_range(args.pixels)
    plot_ghf(args.ghf, dumpflag=args.dumptext, plotflag=args.plot, detailflag=args.detail, selected_pixels=selected_pixels)


if __name__ == "__main__":
    main()
