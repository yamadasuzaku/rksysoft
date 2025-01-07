#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse

# プロットのパラメータを設定します
params = {
    'xtick.labelsize': 14,  # x軸目盛りのフォントサイズ
    'ytick.labelsize': 14,  # y軸目盛りのフォントサイズ
    'legend.fontsize': 12,   # 凡例のフォントサイズ
    'axes.labelsize': 14  # xlabel, ylabel のフォントサイズを変更
}
plt.rcParams['font.family'] = 'serif'  # フォントファミリを設定します
plt.rcParams.update(params)


def parse_qdp_file(file_path, debug=False):
    """
    Parses a QDP file and extracts ene, ene_err, data, data_err as numpy arrays.
    
    Parameters:
        file_path (str): Path to the QDP file.
        debug (bool): If True, print debug information.

    Returns:
        tuple: (ene, ene_err, data, data_err) as numpy arrays.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Skip the first 3 lines
    data_lines = lines[3:]

    # Parse the numerical data
    ene, ene_err, data, data_err = [], [], [], []
    for line in data_lines:
        if line.strip():  # Ignore empty lines
            values = line.split()
            if len(values) == 4:
                ene.append(float(values[0]))
                ene_err.append(float(values[1]))
                data.append(float(values[2]))
                data_err.append(float(values[3]))

    if debug:
        print("Parsed Data:")
        print("ene:", ene)
        print("ene_err:", ene_err)
        print("data:", data)
        print("data_err:", data_err)

    return np.array(ene), np.array(ene_err), np.array(data), np.array(data_err)

def plot_data(ene, ene_err, data, data_err, output_prefix, xlim=None, ylim=None, debug=False):
    """
    Plots the data using matplotlib and saves the plot as .png and .eps files.

    Parameters:
        ene (np.array): Energy values.
        ene_err (np.array): Energy error values.
        data (np.array): Data values.
        data_err (np.array): Data error values.
        output_prefix (str): Prefix for the output filenames.
        xlim (tuple): x-axis limits as (xmin, xmax).
        ylim (tuple): y-axis limits as (ymin, ymax).
        debug (bool): If True, print debug information.
    """
    plt.figure(figsize=(8, 6))
    plt.errorbar(ene, data, xerr=ene_err, yerr=data_err, fmt='.', ecolor='red', capsize=0, label='Data')
    plt.xlabel('Energy')
    plt.ylabel('Data')
    plt.title('Energy vs Data')
    plt.grid(True)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.legend()

    # Save the plot
    png_filename = f"{output_prefix}.png"
    eps_filename = f"{output_prefix}.eps"
    plt.savefig(png_filename, format='png')
    plt.savefig(eps_filename, format='eps')

    if debug:
        print(f"Plot saved as {png_filename} and {eps_filename}")

    plt.show()

def plot_bothdata(ene1, ene_err1, data1, data_err1, ene2, ene_err2, data2, data_err2, output_prefix, xlim=None, ylim=None, debug=False):
    """
    Plots the data using matplotlib and saves the plot as .png and .eps files.

    Parameters:
        ene (np.array): Energy values.
        ene_err (np.array): Energy error values.
        data (np.array): Data values.
        data_err (np.array): Data error values.
        output_prefix (str): Prefix for the output filenames.
        xlim (tuple): x-axis limits as (xmin, xmax).
        ylim (tuple): y-axis limits as (ymin, ymax).
        debug (bool): If True, print debug information.
    """
    # 両方のプロットを線形スケールで作成します
    fig = plt.figure(figsize=(12, 5))

    # メインのプロット領域
    ax1 = fig.add_axes([0.1, 0.12, 0.88, 0.8])
    # インセットプロット領域
    ax2 = fig.add_axes([0.58, 0.57, 0.39, 0.32])

    errorbar = ax1.errorbar(ene1, data1, xerr=ene_err1, yerr=data_err1, fmt='.', color='k', label='Resolve', ms=1)
    for line in errorbar[2]: # errorbar[2] は誤差棒の Line2D オブジェクト
        line.set_alpha(0.3)  # 誤差棒の透明度を0.3に設

    corfactor = 12
    errorbar = ax1.errorbar(ene2, data2/corfactor, xerr=ene_err2, yerr=data_err2/corfactor, fmt='.', color='skyblue', ms=1, label=f"Xtend (1/{corfactor})")
    for line in errorbar[2]: # errorbar[2] は誤差棒の Line2D オブジェクト
        line.set_alpha(0.3)  # 誤差棒の透明度を0.3に設

    ax1.set_xlabel('Energy (keV)')
    ax1.set_ylabel('Counts/s/keV')
#    ax1.set_title('Energy vs Data')
#    ax1.grid(True,alpha=0.2)

    if xlim:
        ax1.set_xlim(xlim)
    if ylim:
        ax1.set_ylim(ylim)
    ax1.legend(loc="lower left")

    errorbar = ax2.errorbar(ene1, data1, xerr=ene_err1, yerr=data_err1, fmt='.', color='k', label='Resolve', ms=1)
    for line in errorbar[2]: # errorbar[2] は誤差棒の Line2D オブジェクト
        line.set_alpha(0.3)  # 誤差棒の透明度を0.3に設
    errorbar = ax2.errorbar(ene1, data1, xerr=ene_err1, yerr=data_err1, fmt='-', color='k', alpha=0.1)

#    corfactor = 12
#    errorbar = ax2.errorbar(ene2, data2/corfactor, xerr=ene_err2, yerr=data_err2/corfactor, fmt='.', color='skyblue', ms=1, label=f"Xtend (1/{corfactor})")
    # for line in errorbar[2]: # errorbar[2] は誤差棒の Line2D オブジェクト
    #     line.set_alpha(0.3)  # 誤差棒の透明度を0.3に設
    ax2.set_xlim(6.3,7.2)
    ax2.set_ylim(1.3,3.1)
    ax2.grid(True,alpha=0.05, axis='x', linestyle="--")
    # 不要な枠線を削除します
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    ax2.text(6.395, 1.45, r"Fe I", rotation=70, color="salmon", fontsize=10)
    ax2.text(6.700, 2.55, r"Fe XXV", rotation=70, color="salmon", fontsize=10)
    ax2.text(6.952, 2.35, r"Fe XXVI", rotation=70, color="salmon", fontsize=10)

    # Save the plot
    png_filename = f"{output_prefix}.png"
    eps_filename = f"{output_prefix}.eps"
    plt.savefig(png_filename, format='png')
    plt.savefig(eps_filename, format='eps')

    if debug:
        print(f"Plot saved as {png_filename} and {eps_filename}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Parse and plot QDP file data.")
    parser.add_argument('file1', type=str, help="Path to the QDP file. (Resolve)")
    parser.add_argument('file2', type=str, help="Path to the QDP file. (Xend)")
    parser.add_argument('--output', type=str, default="spec_resolve_xtend_cygx1", help="Output prefix for saved files.")
    parser.add_argument('--xlim', type=float, nargs=2, help="X-axis limits as 'xmin xmax'.", default=[4,10])
    parser.add_argument('--ylim', type=float, nargs=2, help="Y-axis limits as 'ymin ymax'.", default=[0,3.4])
    parser.add_argument('--debug', action='store_true', help="Enable debug mode.")

    args = parser.parse_args()
    debug = args.debug 

    # Parse the QDP file of Resolve
    ene1, ene_err1, data1, data_err1 = parse_qdp_file(args.file1, debug=args.debug)
    # Plot the data
    if debug: 
        plot_data(ene1, ene_err1, data1, data_err1, args.output, xlim=args.xlim, ylim=args.ylim, debug=args.debug)

    # Parse the QDP file of Xtend 
    ene2, ene_err2, data2, data_err2 = parse_qdp_file(args.file2, debug=args.debug)
    # Plot the data
    if debug:
        plot_data(ene2, ene_err2, data2, data_err2, args.output, xlim=args.xlim, ylim=args.ylim, debug=args.debug)

    # Plot both data
    plot_bothdata(ene1, ene_err1, data1, data_err1, ene2, ene_err2, data2, data_err2, args.output, xlim=args.xlim, ylim=args.ylim, debug=args.debug)



if __name__ == "__main__":
    main()
