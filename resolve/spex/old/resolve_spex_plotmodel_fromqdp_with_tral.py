#!/usr/bin/env python

import matplotlib.pyplot as plt
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)
import numpy as np
import pandas as pd
import argparse
import os

def read_data(file_path, emin, emax):
    data = np.loadtxt(file_path, skiprows=1)
    xval = data[:, 0]
    xerr_1 = data[:, 1]
    xerr_2 = data[:, 2]
    model = data[:, 3]
    ecut = np.where((xval > emin) & (xval <= emax))[0]
    xval = xval[ecut]
    xerr_1 = xerr_1[ecut]
    xerr_2 = xerr_2[ecut]
    xerr = [xerr_1, xerr_2]
    model = model[ecut]
    return xval, xerr, model

def read_secondary_data(file_path):
    # テキストファイルを読み込む
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header_index = 0
    data_lines = lines[header_index + 2:]  # 2行分のヘッダーを飛ばす

    # データフレームに変換するための準備
    data = []
    for line in data_lines:
        if line.strip():  # 空行をスキップ
            parts = line.split()
            if len(parts) >= 14:  # 必要な列が存在するか確認
                try:
                    energy_keV = float(parts[10])
                    ew_keV = float(parts[13])
                    data.append((energy_keV, ew_keV))
                except ValueError:
                    continue  # 数値に変換できない場合はスキップ

    # データフレームを作成
    df = pd.DataFrame(data, columns=['Energy keV', 'EW keV'])
    return df

def plot_data(qdpfile, secondary_file, ylin, emin, emax, y1, y2, output_name, plotflag, marker_size):
    x, xerr, model = read_data(qdpfile, emin, emax)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    if ylin:
        ax1.set_yscale('linear')
    else:
        ax1.set_yscale('log')

    if y1 is not None:
        ax1.set_ylim(y1, y2)
        
    ax1.errorbar(x, model, fmt='-', label='model', capsize=0, color='red', alpha=0.9)
    ax1.set_ylabel(r'Photons/m$^2$/s/keV')
    ax1.set_xlabel('Energy (keV)')
    ax1.set_title(qdpfile)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if secondary_file:
        df = read_secondary_data(secondary_file)
        ax2 = ax1.twinx()
        ax2.scatter(df['Energy keV'], 1e3 * df['EW keV'], color='blue', alpha=0.8, s=marker_size, label="EW")
        ax2.set_ylabel('EW (eV)')
        ax2.legend(loc="lower right")
        print(f"Secondary data plotted from {secondary_file}")

    plt.savefig(output_name)
    print(f"Output file {output_name} is created.")
    if plotflag:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot spectra from a model QDP file.')
    parser.add_argument('qdpfile', help='The name of the QDP file to process.')
    parser.add_argument('-s', '--secondary_file', type=str, default=None, help='Secondary data file for scatter plot')
    parser.add_argument('-l', '--ylin', action='store_true', help='Plot yscale in linear scale (log in default)')
    parser.add_argument('-m', '--emin', type=float, help='emin', default=1.5)
    parser.add_argument('-x', '--emax', type=float, help='emax', default=10.0)
    parser.add_argument('--y1', type=float, help='y1', default=None)
    parser.add_argument('--y2', type=float, help='y2', default=None)
    parser.add_argument('--output', type=str, help='Output file name', default='spex_fitmodel.png')
    parser.add_argument('--plot', '-p', action='store_true', default=False, help='Flag to display plot')
    parser.add_argument('--marker_size', type=float, help='Size of the scatter plot markers', default=6)

    args = parser.parse_args()

    plot_data(args.qdpfile, args.secondary_file, args.ylin, args.emin, args.emax, args.y1, args.y2, args.output, args.plot, args.marker_size)
