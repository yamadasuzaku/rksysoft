#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
import astropy.io.fits
import argparse
import sys
import os
from matplotlib.colors import LogNorm

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

# pixel から (detx, dety) へのマッピング辞書を作成
pixel_to_detx_dety = {}
for detx_from_zero in range(6):
    for dety_from_zero in range(6):
        pixel = pixel_fromdetxdety[detx_from_zero][dety_from_zero]
        pixel_to_detx_dety[pixel] = (detx_from_zero + 1, dety_from_zero + 1)  # 1-based index


def read_fits_data(filename, pha_range, oneitype = 0):
    with astropy.io.fits.open(filename) as fits:
        data = fits[1].data
        date_obs = fits[1].header["DATE-OBS"]
        exposure = fits[1].header["EXPOSURE"]        
        detx, dety, pha, itype = data["DETX"], data["DETY"], data["PHA"], data["ITYPE"]
        counts = np.zeros([6, 6])
        for one_detx in np.arange(1, 7):
            for one_dety in np.arange(1, 7):
                condition = (detx == one_detx) & (dety == one_dety) & (itype == oneitype)
                if pha_range:
                    condition &= (pha >= pha_range[0]) & (pha <= pha_range[1])
                counts[one_detx-1][one_dety-1] = np.sum(condition)
    return counts, date_obs, exposure

def plot_data(counts, pixel_fromdetxdety, fname, date_obs, itype, pha_range, show=False, exposure=0.):
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
    fig.suptitle(f"{fname}\nDATE: {date_obs} TYPE=" + str(typename[itype]) + " (" + filtermemo + ")" + f" #={total}",fontsize=9)

    if exposure > 0:
        fig.suptitle(f"{fname}\nDATE: {date_obs} TYPE=" + str(typename[itype]) + " (" + filtermemo + ")" + f" #={total}" + \
                                  f" rate={total/exposure:0.2e} c/s (exp={exposure}s)",fontsize=9)
    else:
        fig.suptitle(f"{fname}\nDATE: {date_obs} TYPE=" + str(typename[itype]) + " (" + filtermemo + ")" + f" #={total}",fontsize=9)

    for ax, norm in zip(axs, [LogNorm(vmin=1, vmax=counts.max()), None]):
        pcm = ax.pcolormesh(xbins, ybins, np.abs(counts.T), norm=norm, cmap="plasma")
        cbar = plt.colorbar(pcm, ax=ax, fraction=0.04, shrink=0.8, pad=0.02) # ★ カラーバーを細長く短くする
        if norm == None:
            cbar.set_label("Counts (linear)", fontsize=9)  # カラーバーのタイトルを設定
        else:
            cbar.set_label("Counts (log)", fontsize=9)  # カラーバーのタイトルを設定            
        ax.set_aspect('equal')  # ★ x軸とy軸のスケールを同じにする
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
    output_filename = f"resolve_map2d_{typename[itype]}_{os.path.basename(fname).replace('.evt', '')}{pha_suffix}.png"
    
    plt.savefig(output_filename)
    print(f"{output_filename} is saved.")

    if show:
        plt.show()


    # # plot 1D plot
    # _pixel_list = []
    # _count_list = []
    # for _pixel in np.arange(0, 36):
    #     detx, dety = pixel_to_detx_dety.get(_pixel, (None, None))  # 存在しない場合は (None, None)
    #     _count = counts[detx-1][dety-1]
    #     print(f"Pixel {_pixel} (DETX={detx}, DETY={dety}) : Counts = {_count}")
    #     _pixel_list.append(_pixel)
    #     _count_list.append(_count)





def plot_1d_pixel_counts(counts, fname, date_obs, itype, pha_range, oneitype = 0, show=False):

    filtermemo = f"PHA{pha_range[0]}-{pha_range[1]}" if pha_range else "no PHA cut"

    _pixel_list = []
    _count_list = []
    for _pixel in np.arange(0, 36):
        detx, dety = pixel_to_detx_dety.get(_pixel, (None, None))  # 存在しない場合は (None, None)
        if detx is not None and dety is not None:
            _count = counts[detx-1][dety-1]
        else:
            _count = 0
        _pixel_list.append(_pixel)
        _count_list.append(_count)
    
    # _count_list の大きい順に並べ替え
    sorted_indices = np.argsort(_count_list)[::-1]
    sorted_pixel_list = np.array(_pixel_list)[sorted_indices]
    sorted_count_list = np.array(_count_list)[sorted_indices]
        
    # プロット
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    fig.suptitle(f"input: {fname}\nDATE: {date_obs} TYPE=" + str(typename[itype]) + " (" + filtermemo + ")",fontsize=10)
    
    # axs[0].bar(_pixel_list, _count_list, color='b')
    axs[0].bar(_pixel_list, _count_list, color='b', alpha=0.7)
    for i, v in enumerate(_count_list):
        axs[0].text(i, v * 1.04 + 1, str(int(v)), ha='center', rotation=90)    
    axs[0].set_xlabel("Pixel")
    axs[0].set_ylabel("Counts")
    axs[0].set_title("Counts per Pixel")

    axs[1].bar(np.arange(1, 37), sorted_count_list, color='r', alpha=0.7)
    for i, v in enumerate(sorted_count_list):
        axs[1].text(i + 1, v *1.04 + 1, str(int(v)), ha='center', rotation=90)
    axs[1].set_xticks(np.arange(1, 37))
    axs[1].set_xticklabels(sorted_pixel_list, rotation=90)
    axs[1].set_xlabel("Pixel (sorted by count)")
    axs[1].set_ylabel("Counts")
    axs[1].set_title("Sorted Counts per Pixel")
    
    # axs[1].bar(sorted_pixel_list, sorted_count_list, color='r')
    # axs[1].set_xlabel("Pixel (sorted by count)")
    # axs[1].set_ylabel("Counts")
    # axs[1].set_title("Sorted Counts per Pixel")
    
    plt.tight_layout()

    # ファイル名の成形    
    pha_suffix = f"_PHA{pha_range[0]}-{pha_range[1]}" if pha_range else ""

    output_txtfilename = f"resolve_count1d_{typename[itype]}_{os.path.basename(fname).replace('.evt', '')}{pha_suffix}.txt"
    # テキストファイルに書き出し
    with open(output_txtfilename, "w") as f:
        for _p, _c in zip(sorted_pixel_list, sorted_count_list):
            detx, dety = pixel_to_detx_dety.get(_p, (None, None))  # 存在しない場合は (None, None)
            print(f"Pixel {_p} (DETX={detx}, DETY={dety}) : Counts = {_c}")
            f.write(f"{_p}, {_c}\n")

    # pngファイルに書き出し
    output_filename = f"resolve_count1d_{typename[itype]}_{os.path.basename(fname).replace('.evt', '')}{pha_suffix}.png"
    plt.savefig(output_filename)
    print(f"{output_filename} is saved.")
    plt.savefig(output_filename)
    if show:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='FITSファイルからデータを処理し、プロットします。')
    parser.add_argument('filename1', help='1つ目のFITSファイル')
    parser.add_argument('filename2', nargs='?', default=None, help='2つ目のFITSファイル（オプション）')
    parser.add_argument('--pha_min', '-min', type=int, default=None, help='PHAの下限値')
    parser.add_argument('--pha_max', '-max', type=int, default=None, help='PHAの上限値')
    parser.add_argument('--show', '-s', action='store_true', help='plt.show()を実行するかどうか')
    parser.add_argument('--itypenames', '-y', type=str, help='Comma-separated list of itype', default='0,1,2,3,4')    
    args = parser.parse_args()
    pha_range = (args.pha_min, args.pha_max) if args.pha_min is not None and args.pha_max is not None else None
    itypenames = list(map(int, args.itypenames.split(',')))

    for itype_ in itypenames:
        print(f"----- START ITYPE {itype_}")
        counts1, date_obs1, exposure1 = read_fits_data(args.filename1, pha_range, oneitype = itype_)
        plot_data(counts1, pixel_fromdetxdety, args.filename1, date_obs1, itype_, pha_range, show=args.show, exposure=exposure1)
        plot_1d_pixel_counts(counts1, args.filename1, date_obs1, itype_, pha_range, show=args.show)

        if args.filename2:
            print(f"..... processing {args.filename2}")
            counts2, date_obs2, exposure2 = read_fits_data(args.filename2, pha_range, oneitype = itype_)
            plot_data(counts2, pixel_fromdetxdety, args.filename2, date_obs1, itype_, pha_range, show=args.show, exposure=exposure2)
            plot_1d_pixel_counts(counts2, args.filename2, date_obs2, itype_, pha_range, show=args.show)

            print(f"..... processing {args.filename1} - {args.filename2}")
            counts_diff = counts1 - counts2
            diff_filename = f"{os.path.basename(args.filename1).replace('.evt', '')}_{os.path.basename(args.filename2).replace('.evt', '')}"
            plot_data(counts_diff, pixel_fromdetxdety, diff_filename, date_obs1, itype_, pha_range, show=args.show, exposure=exposure1)
            plot_1d_pixel_counts(counts_diff, diff_filename, date_obs1, itype_, pha_range, show=args.show)

        print(f"----- END ITYPE {itype_}\n")

if __name__ == "__main__":
    main()
