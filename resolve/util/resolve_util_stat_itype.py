import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import argparse
import os

# Plotting Configuration
plt.rcParams['font.family'] = 'serif'

itypemax = 8  # Hp, Mp, Ms, Lp, Ls, BL, EL, NA
g_typename = ["Hp", "Mp", "Ms", "Lp", "Ls", "BL", "EL", "NA"]

def load_itype_column(fits_file):
    """FITS ファイルから ITYPE コラムを抽出する"""
    with fits.open(fits_file) as hdul:
        itype_data = hdul[1].data['ITYPE']
    return itype_data

def analyze_itype_distribution(itype_data):
    """ITYPE の出現頻度を集計し、分布を計算する"""
    counts = np.zeros(itypemax, dtype=int)  # ITYPE 0～7 のカウント
    total_events = len(itype_data)

    # ITYPEごとに出現回数をカウント
    for itype in itype_data:
        counts[itype] += 1

    # 出現比率の計算
    ratios = counts / max(1, total_events)

    return counts, ratios, total_events

def plot_itype_distribution(counts, ratios, fits_file, output_png, show_plot, total):
    """ITYPE の統計分布をヒストグラムと円グラフで可視化する"""
    labels = [f'{g_typename[i]}' for i in range(itypemax)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # ヒストグラム (カウント)
    ax1.bar(labels, counts, color='b', alpha=0.7)
    ax1.set_ylabel('Count')
    ax1.set_title('Count Distribution')
    for i, v in enumerate(counts):
        ax1.text(i, v + 0.03 * max(counts), str(v), ha='center', va='bottom')

    # 円グラフ (比率)
    ax2.pie(ratios, labels=labels, autopct='%1.1f%%', startangle=0, colors=plt.cm.tab20.colors)
    ax2.set_title('Ratio Distribution')

    # ファイル情報を表示
    plt.figtext(0.15, 0.97, f"Input File: {os.path.basename(fits_file)}", ha='center', fontsize=8, color='gray')
    plt.figtext(0.15, 0.95, f"Total Events = {total}", ha='center', fontsize=8, color='gray')

    plt.suptitle(f'ITYPE Stat. - {os.path.basename(fits_file)}', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"Plot saved as {output_png}")


    if show_plot:
        plt.show()

def main(fits_file, show_plot):
    if not os.path.exists(fits_file):
        print(f"Error: {fits_file} not found.")
        return

    itype_data = load_itype_column(fits_file)
    counts, ratios, total_events = analyze_itype_distribution(itype_data)

    # 統計情報の表示
    print("ITYPE Count Distribution:")
    for i in range(itypemax):
        print(f"ITYPE {i}: {counts[i]} ({ratios[i]:.2%})")

    # PNG ファイル名を入力ファイル名に基づいて生成
    output_png = f"{os.path.splitext(os.path.basename(fits_file))[0]}_itype_distribution.png"
    plot_itype_distribution(counts, ratios, fits_file, output_png, show_plot, total_events)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze ITYPE column distribution in a FITS file.')
    parser.add_argument('fits_file', help='Path to the FITS file')
    parser.add_argument('--show', '-s', action='store_true', help='Show the plot')
    args = parser.parse_args()

    main(args.fits_file, args.show)
