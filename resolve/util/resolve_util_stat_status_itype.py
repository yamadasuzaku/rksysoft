import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import argparse
import os

# Plotting Configuration
plt.rcParams['font.family'] = 'serif'

itypemax = 8  # Hp, Mp, Ms, Lp, Ls, BL, EL, NA
g_typename = ["Hp", "Mp", "Ms", "Lp", "Ls", "BL", "EL", "NA"]

def load_status_column(fits_file):
    """FITS ファイルから STATUS と ITYPE コラムを抽出する"""
    with fits.open(fits_file) as hdul:
        status_data = hdul[1].data['STATUS']
        itype_data = hdul[1].data['ITYPE']
    return status_data, itype_data

def analyze_status_bits(status_data, itype_data):
    """全体および ITYPE ごとのビットフラグの統計量を計算"""
    num_bits = 16  # 16ビット全体を解析する
    total_event = len(itype_data)

    # 全体および各 ITYPE ごとのビットカウント初期化
    bit_counts = {itype: {f'S[{i}]': 0 for i in range(num_bits)} for itype in range(itypemax)}

    # 各イベントのビットフラグを解析
    for status, itype in zip(status_data, itype_data):
        for i in range(num_bits):
            if status[i]:  # ビットが立っている場合
                bit_counts[itype][f'S[{i}]'] += 1

    # 比率の計算
    bit_ratios = {
        itype: {k: v / max(1, total_event) for k, v in bits.items()}
        for itype, bits in bit_counts.items()
    }

    return bit_counts, bit_ratios

def plot_status_statistics(bit_counts, bit_ratios, fits_file, itype_data, output_png, show_plot, total, itypemax_for_plot=5):
    """ITYPEごとのビットフラグの統計量を可視化し、PNG に保存"""
    num_bits = 16
    fig, axes = plt.subplots(itypemax_for_plot, 1, figsize=(12, 8), sharex=True)

    for itype in range(itypemax_for_plot):
        cutid=np.where(itype_data==itype)[0]
        itype_num = len(itype_data[cutid])

        labels = list(bit_counts[itype].keys())
        counts = list(bit_counts[itype].values())
        ratios = list(bit_ratios[itype].values())

        ax = axes[itype]

        # 左側のY軸: カウントのプロット
        ax.bar(labels, counts, alpha=0.6, color='b', label=f'{g_typename[itype]} ({itype_num} events, {np.sum(counts)} flags)')
        ax.set_ylabel('Count', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        for i, v in enumerate(counts):
            ax.text(i, v + 0.04 * max(counts), str(v), ha='center', va='bottom', color='b')
        ax.legend(fontsize=8, loc="upper right")

        # 右側のY軸: 比率のプロット
        ax2 = ax.twinx()
        ax2.plot(labels, ratios, color='g', marker='o', linestyle='-', label='Ratio', alpha=0.5)
        ax2.set_ylabel('Ratio', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        for i, v in enumerate(ratios):
            ax2.text(i+0.2, v + 0.2 * max(ratios), f"{v:.3f}", ha='center', va='bottom', color='g', fontsize=8, alpha=0.5)

#        ax.set_title(f'ITYPE {itype} - Bit Flag Analysis')

    plt.figtext(0.15, 0.97, f"Input File: {os.path.basename(fits_file)}", ha='center', fontsize=8, color='gray')
    plt.figtext(0.15, 0.95, f"Total Events = {total}", ha='center', fontsize=8, color='gray')
    plt.suptitle(f'STATUS Bit - {os.path.basename(fits_file)}', fontsize=12)
#    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"Plot saved as {output_png}")

    if show_plot:
        plt.show()

def main(fits_file, show_plot):
    if not os.path.exists(fits_file):
        print(f"Error: {fits_file} not found.")
        return

    status_data, itype_data = load_status_column(fits_file)
    bit_counts, bit_ratios = analyze_status_bits(status_data, itype_data)

    # 各 ITYPE のビットフラグのカウントと比率を表示
    for itype in range(itypemax):
        print(f"\nITYPE {itype} - Bit Flag Counts:")
        for flag, count in bit_counts[itype].items():
            print(f"{flag}: {count}")

        print(f"\nITYPE {itype} - Bit Flag Ratios:")
        for flag, ratio in bit_ratios[itype].items():
            print(f"{flag}: {ratio:.4f}")

    # PNG ファイル名を入力ファイル名に基づいて生成
    output_png = f"{os.path.splitext(os.path.basename(fits_file))[0]}_status_itype_plot.png"
    total_event=len(itype_data)
    plot_status_statistics(bit_counts, bit_ratios, fits_file, itype_data, output_png, show_plot, total_event)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze STATUS column by ITYPE in a FITS file.')
    parser.add_argument('fits_file', help='Path to the FITS file')
    parser.add_argument('--show', '-s', action='store_true', help='Show the plot')
    args = parser.parse_args()

    main(args.fits_file, args.show)
