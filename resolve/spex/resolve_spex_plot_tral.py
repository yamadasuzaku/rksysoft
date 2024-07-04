#!/usr/bin/env python 

import pandas as pd
import matplotlib.pyplot as plt
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)
import argparse
import os

def main(file_path, output_name, show_plot, marker_size):
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

    # プロットする
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Energy keV'], df['EW keV'], color='blue', alpha=0.8, s=marker_size)
    plt.title(f'Energy keV vs. EW keV : {file_path}')
    plt.xlabel('Energy keV')
    plt.ylabel('EW keV')
    plt.grid(True)

    # グラフを保存する
    plt.savefig(output_name)
    print(f"..... save as {output_name}")

    # プロットを表示するかどうかをフラグで調整
    if show_plot:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Energy keV vs. EW keV from a data file")
    parser.add_argument("file_path", type=str, help="Path to the input data file")
    parser.add_argument("--output_name", type=str, default=None, help="Name of the output image file (default: same as input with .png extension)")
    parser.add_argument("--show_plot", action="store_true", help="Flag to display the plot")
    parser.add_argument("--marker_size", type=float, default=5, help="Size of the plot markers (default: 5)")

    args = parser.parse_args()

    # 出力ファイル名を設定
    if args.output_name is None:
        base_name = os.path.splitext(os.path.basename(args.file_path))[0]
        output_name = base_name + ".png"
    else:
        output_name = args.output_name

    main(args.file_path, output_name, args.show_plot, args.marker_size)
