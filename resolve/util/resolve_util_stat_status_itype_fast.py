#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import argparse
import os
import pandas as pd

# Plotting Configuration
plt.rcParams['font.family'] = 'serif'

itypemax = 8  # Hp, Mp, Ms, Lp, Ls, BL, EL, NA
g_typename = ["Hp", "Mp", "Ms", "Lp", "Ls", "BL", "EL", "NA"]
num_bits = 16  # 16ビット全体を解析する
all_pixels = range(36)  # PIXEL values from 0 to 35

def load_status_column(fits_file):
    """FITS ファイルから STATUS、ITYPE、PIXEL コラムを抽出する"""
    with fits.open(fits_file) as hdul:
        status_data = hdul[1].data['STATUS']
        itype_data = hdul[1].data['ITYPE']
        pixel_data = hdul[1].data['PIXEL']
    return np.array(status_data), np.array(itype_data), np.array(pixel_data)

def analyze_status_bits_per_pixel(status_data, itype_data, pixel_data, selected_pixels):
    """PIXELごとにITYPEごとのビットフラグの統計量を計算"""
    # 結果を保存するためのデータフレーム
    bit_counts_per_pixel = {pixel: {itype: {f'S[{i}]': 0 for i in range(num_bits)} for itype in range(itypemax)} for pixel in selected_pixels}
    bit_ratios_per_pixel = {}

    # 各ピクセルに対して計算を実行
    for pixel in selected_pixels:
        pixel_mask = pixel_data == pixel
        total_event = np.sum(pixel_mask)

        for itype in range(itypemax):
            itype_mask = (itype_data == itype) & pixel_mask
            for i in range(num_bits):
                bit_counts_per_pixel[pixel][itype][f'S[{i}]'] = np.sum(status_data[itype_mask, i] > 0)

            # 比率を計算して保存
            bit_ratios_per_pixel[pixel] = {
                itype: {k: v / max(1, total_event) for k, v in bit_counts_per_pixel[pixel][itype].items()}
                for itype in range(itypemax)
            }

    return bit_counts_per_pixel, bit_ratios_per_pixel

def plot_status_statistics_per_pixel(bit_counts_per_pixel, bit_ratios_per_pixel, fits_file, output_dir, show_plot):
    """指定された PIXEL ごとに ITYPE ごとのビットフラグの統計量を可視化し、PNG に保存"""
    os.makedirs(output_dir, exist_ok=True)

    for pixel, bit_counts in bit_counts_per_pixel.items():
        for itype, counts in bit_counts.items():
            ratios = bit_ratios_per_pixel[pixel][itype]
            total = np.sum(list(counts.values()))
            labels = list(counts.keys())
            count_values = list(counts.values())
            ratio_values = list(ratios.values())

            fig, ax = plt.subplots(figsize=(12, 6))

            # 左側のY軸: カウントのプロット
            ax.bar(labels, count_values, alpha=0.6, color='b', label=f'{g_typename[itype]} ({total} flags)')
            ax.set_ylabel('Count', color='b')
            ax.tick_params(axis='y', labelcolor='b')
            for i, v in enumerate(count_values):
                ax.text(i, v + 0.04 * max(count_values), str(v), ha='center', va='bottom', color='b')
            ax.legend(fontsize=8, loc="upper right")

            # 右側のY軸: 比率のプロット
            ax2 = ax.twinx()
            ax2.plot(labels, ratio_values, color='g', marker='o', linestyle='-', label='Ratio', alpha=0.5)
            ax2.set_ylabel('Ratio', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            for i, v in enumerate(ratio_values):
                ax2.text(i + 0.2, v + 0.02 * max(ratio_values), f"{v:.3f}", ha='center', va='bottom', color='g', fontsize=8, alpha=0.5)

            # ファイル情報を表示
            plt.figtext(0.1, 0.97, f"InFile: {os.path.basename(fits_file)}", ha='center', fontsize=8, color='gray')
            plt.figtext(0.1, 0.95, f"Pixel: {pixel} - ITYPE {itype} ({g_typename[itype]})", ha='center', fontsize=8, color='gray')
            plt.suptitle(f'STATUS Bit - Pixel {pixel} - ITYPE {itype} - {os.path.basename(fits_file)}', fontsize=12)

            # 各PIXEL・ITYPEごとにPNGファイルを保存
            output_png = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(fits_file))[0]}_pixel_{pixel}_itype_{itype}_status_plot.png")
            plt.tight_layout()
            plt.savefig(output_png)
            print(f"Plot saved as {output_png}")

            if show_plot:
                plt.show()

            plt.close(fig)

def main(fits_file, output_dir, show_plot, plotpixels):
    if not os.path.exists(fits_file):
        print(f"Error: {fits_file} not found.")
        return

    # プロット対象の PIXEL をリスト形式で取得
    selected_pixels = list(map(int, plotpixels.split(',')))

    status_data, itype_data, pixel_data = load_status_column(fits_file)
    bit_counts_per_pixel, bit_ratios_per_pixel = analyze_status_bits_per_pixel(status_data, itype_data, pixel_data, selected_pixels)

    # 各PIXELおよびITYPEのビットフラグのカウントと比率を表示
    for pixel in selected_pixels:
        print(f"\nPixel {pixel} - STATUS Bit Flag Counts and Ratios by ITYPE:")
        for itype in range(itypemax):
            print(f"\n  ITYPE {itype} ({g_typename[itype]}) - Bit Flag Counts:")
            for flag, count in bit_counts_per_pixel[pixel][itype].items():
                print(f"    {flag}: {count}")

            print(f"\n  ITYPE {itype} ({g_typename[itype]}) - Bit Flag Ratios:")
            for flag, ratio in bit_ratios_per_pixel[pixel][itype].items():
                print(f"    {flag}: {ratio:.4f}")

    # PIXELごとの分布をプロット
    plot_status_statistics_per_pixel(bit_counts_per_pixel, bit_ratios_per_pixel, fits_file, output_dir, show_plot)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze STATUS column by ITYPE and PIXEL in a FITS file.')
    parser.add_argument('fits_file', help='Path to the FITS file')
    parser.add_argument('--output_dir', '-o', default='output', help='Directory to save the output plots')
    parser.add_argument('--show', '-s', action='store_true', help='Show the plot')
    parser.add_argument('--plotpixels', '-p', type=str, help='プロットするピクセルのカンマ区切りリスト', default=','.join(map(str, range(36))))
    args = parser.parse_args()

    main(args.fits_file, args.output_dir, args.show, args.plotpixels)
