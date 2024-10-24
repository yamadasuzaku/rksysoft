#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
import astropy.io.fits
import argparse
import sys
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import os

# Observation Type Information
itypename = [0, 1, 2, 3, 4]
typename = ["Hp", "Mp", "Ms", "Lp", "Ls"]

# ピクセルマッピングのデータを直接定義
pixel_fromdetxdety = [
    [12, 11,  9, 19, 21, 23],
    [14, 13, 10, 20, 22, 24],
    [16, 15, 17, 18, 25, 26],
    [ 8,  7,  0, 35, 33, 34],
    [ 6,  4,  2, 28, 31, 32],
    [ 5,  3,  1, 27, 29, 30]
]

def plot_data(counts, pixel_fromdetxdety, fname, date_obs, itype, pha_range, show=False):
    """
    データをプロットする関数
    """
    xbins = np.linspace(0.5, 6.5, 7)
    ybins = np.linspace(0.5, 6.5, 7)

    total = int(np.sum(counts))
    if total == 0:
        print(f"(plot_data) no data found in itype={itype}: {total}")
        return -1 
    else:
        print(f"(plot_data) data found in itype={itype}: {total}")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    filtermemo = f"PHA{pha_range[0]}-{pha_range[1]}" if pha_range else "no PHA cut"
    fig.suptitle(f"{fname} DATE: {date_obs} TYPE=" + str(typename[itype]) + " (" + filtermemo + ")" + f" #={total}",fontsize=10)

    for ax, norm in zip(axs, [LogNorm(vmin=1, vmax=counts.max()), None]):
        pcm = ax.pcolormesh(xbins, ybins, counts.T, norm=norm, cmap="plasma")
        plt.colorbar(pcm, ax=ax)

        for one_detx in np.arange(1, 7):
            for one_dety in np.arange(1, 7):
                ax.text(one_detx - 0.3, one_dety + 0.3, pixel_fromdetxdety[one_detx-1][one_dety-1],
                        ha="center", va="center", color="0.9", size=8)

                ax.text(one_detx, one_dety, f'{int(counts[one_detx-1][one_dety-1])}',
                        ha="center", va="center", color="k", size=9)

        ax.set_xlabel("DETX")
        ax.set_ylabel("DETY")

    # ファイル名の成形
    pha_suffix = f"_PHA{pha_range[0]}-{pha_range[1]}" if pha_range else ""
    output_filename = f"{typename[itype]}_{os.path.basename(fname).replace('.evt', '')}{pha_suffix}.png"
    
    plt.savefig(output_filename)
    if show:
        plt.show()

def main():
    # argparseのセットアップ
    parser = argparse.ArgumentParser(description='FITSファイルからデータを処理し、プロットします。')
    parser.add_argument('filename', help='入力するFITSファイルの名前')
    parser.add_argument('--pha_min', '-min', type=int, default=None, help='PHAの下限値')
    parser.add_argument('--pha_max', '-max', type=int, default=None, help='PHAの上限値')
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグモード')    
    parser.add_argument('--show', '-s', action='store_true', help='plt.show()を実行するかどうか。defaultはplotしない。')    

    args = parser.parse_args()

    pha_range = (args.pha_min, args.pha_max) if args.pha_min is not None and args.pha_max is not None else None

    # FITSファイルを開く
    with astropy.io.fits.open(args.filename) as fits:
        date_obs = fits[1].header["DATE-OBS"]
        data = fits[1].data

        times, pixel, detx, dety, pha, itype = (
            data["TIME"], data["PIXEL"], data["DETX"], data["DETY"], data["PHA"], data["ITYPE"]
        )

        counts = np.zeros([6, 6])

        print("#, detx, dety, pixel, count)")
        for oneitype in itypename:
            for one_detx in np.arange(1, 7):
                for one_dety in np.arange(1, 7):
                    # PHA範囲のフィルタリング
                    condition = (detx == one_detx) & (dety == one_dety) & (itype == oneitype)
                    if pha_range:
                        condition &= (pha >= pha_range[0]) & (pha <= pha_range[1])

                    cutid = np.where(condition)[0]

                    one_pixel = pixel_fromdetxdety[one_detx-1][one_dety-1]
                    one_count = len(pha[cutid])
                    counts[one_detx-1][one_dety-1] = int(one_count)
                    if args.debug:
                        print(one_detx-1, one_detx, one_dety, one_pixel, one_count)

            plot_data(counts, pixel_fromdetxdety, args.filename, date_obs, oneitype, pha_range, show=args.show)

if __name__ == "__main__":
    main()
