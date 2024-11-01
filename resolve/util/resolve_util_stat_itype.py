#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import argparse
import os

# Plotting Configuration
plt.rcParams['font.family'] = 'serif'

itypemax = 8  # Hp, Mp, Ms, Lp, Ls, BL, EL, NA
g_typename = ["Hp", "Mp", "Ms", "Lp", "Ls", "BL", "EL", "NA"]
all_pixels = range(36)  # PIXEL values from 0 to 35

def load_itype_column(fits_file):
    """FITS ファイルから ITYPE と PIXEL コラムを抽出する"""
    with fits.open(fits_file) as hdul:
        itype_data = hdul[1].data['ITYPE']
        pixel_data = hdul[1].data['PIXEL']
    return itype_data, pixel_data

def analyze_itype_distribution_per_pixel(itype_data, pixel_data, selected_pixels):
    """指定された PIXEL ごとに ITYPE の出現頻度を集計し、分布を計算する"""
    counts_per_pixel = {pixel: np.zeros(itypemax, dtype=int) for pixel in selected_pixels}
    ratios_per_pixel = {}

    # PIXELごとにITYPEの出現回数をカウント
    for itype, pixel in zip(itype_data, pixel_data):
        if pixel in counts_per_pixel:
            counts_per_pixel[pixel][itype] += 1

    # 出現比率を計算
    for pixel in selected_pixels:
        total_events = np.sum(counts_per_pixel[pixel])
        ratios_per_pixel[pixel] = counts_per_pixel[pixel] / max(1, total_events)

    return counts_per_pixel, ratios_per_pixel

def plot_itype_distribution_per_pixel(counts_per_pixel, ratios_per_pixel, fits_file, output_dir, show_plot):
    """指定された PIXEL ごとに ITYPE の統計分布をヒストグラムと円グラフで可視化する"""
    os.makedirs(output_dir, exist_ok=True)

    for pixel in counts_per_pixel.keys():
        counts = counts_per_pixel[pixel]
        ratios = ratios_per_pixel[pixel]
        total = np.sum(counts)

        labels = [f'{g_typename[i]}' for i in range(itypemax)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

        # ヒストグラム (カウント)
        ax1.bar(labels, counts, color='b', alpha=0.7)
        ax1.set_ylabel('Count')
        ax1.set_title(f'Pixel {pixel} - Count Distribution')
        for i, v in enumerate(counts):
            ax1.text(i, v + 0.01 * max(counts), str(v), ha='center', va='bottom')

        # 円グラフ (比率)
        ax2.pie(ratios, labels=labels, autopct='%1.1f%%', startangle=0, colors=plt.cm.tab20.colors)
        ax2.set_title(f'Pixel {pixel} - Ratio Distribution')

        # ファイル情報を表示
        plt.figtext(0.15, 0.97, f"Input File: {os.path.basename(fits_file)}", ha='center', fontsize=8, color='gray')
        plt.figtext(0.15, 0.95, f"Total Events (Pixel {pixel}) = {total}", ha='center', fontsize=8, color='gray')

        plt.suptitle(f'ITYPE Stat. - Pixel {pixel} - {os.path.basename(fits_file)}', fontsize=10)
        plt.tight_layout()

        # 各PIXELごとにPNGファイルを保存
        output_png = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(fits_file))[0]}_pixel_{pixel}_itype_distribution.png")
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

    itype_data, pixel_data = load_itype_column(fits_file)
    counts_per_pixel, ratios_per_pixel = analyze_itype_distribution_per_pixel(itype_data, pixel_data, selected_pixels)

    # 統計情報の表示
    for pixel in selected_pixels:
        counts = counts_per_pixel[pixel]
        ratios = ratios_per_pixel[pixel]
        total = np.sum(counts)
        print(f"Pixel {pixel} - ITYPE Count Distribution:")
        for i in range(itypemax):
            print(f"  ITYPE {i} ({g_typename[i]}): {counts[i]} ({ratios[i]:.2%})")
        print(f"  Total Events: {total}\n")

    # 選択された PIXEL の分布をプロット
    plot_itype_distribution_per_pixel(counts_per_pixel, ratios_per_pixel, fits_file, output_dir, show_plot)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze ITYPE column distribution per PIXEL in a FITS file.')
    parser.add_argument('fits_file', help='Path to the FITS file')
    parser.add_argument('--output_dir', '-o', default='output', help='Directory to save the output plots')
    parser.add_argument('--show', '-s', action='store_true', help='Show the plot')
    parser.add_argument('--plotpixels', '-p', type=str, help='プロットするピクセルのカンマ区切りリスト', default=','.join(map(str, range(36))))
    args = parser.parse_args()

    main(args.fits_file, args.output_dir, args.show, args.plotpixels)
