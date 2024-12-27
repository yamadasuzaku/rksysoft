#!/usr/bin/env python

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm

# グローバル設定
plt.rcParams['font.family'] = 'serif'

def parse_file(filename, verbose=0):
    """
    ファイルを解析してデータを辞書形式に変換する関数。

    Parameters:
        filename (str): 処理するファイル名。
        verbose (int): 詳細情報の出力レベル。

    Returns:
        dict: GRIDごとのデータフレーム辞書。
    """
    data = {}
    current_grid = None
    columns = None

    with open(filename, 'r') as f:
        for line in f:
            if verbose >= 2:
                print(f"行: {line.strip()}")
            # GRID_DELIMITを検出
            if line.startswith('########################### GRID_DELIMIT'):
                current_grid = re.search(r'grid\d+', line).group()
                data[current_grid] = []
                if verbose >= 1:
                    print(f"新しいグリッド検出: {current_grid}")
            elif line.startswith('#depth'):
                # ヘッダー行を取得
                columns = line.strip('#').strip().split('\t')
            elif current_grid and columns:
                # データ行を追加
                values = line.strip().split('\t')
                data[current_grid].append([float(v) for v in values])

    return {grid_name: pd.DataFrame(grid_data, columns=columns) for grid_name, grid_data in data.items()}

def parse_grid_file(file_path):
    """
    Parses a tab-separated grid file (e.g., cygx1_lhs_hden10dr8zone1.mygrid) file and returns a list of dictionaries.

    Parameters:
    file_path (str): Path to the file to be parsed.

    Returns:
    list[dict]: A list of dictionaries where each dictionary represents a row in the file.
    """
    parsed_data = []
    column_names = [
        "Index", "Failure?", "Warnings?", "Exit_code", "#rank", "#seq",
        "XI", "grid_parameter_string"]

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Skip comments and empty lines
                if line.startswith("#") or not line.strip():
                    continue
                
                # Split the line by tabs
                columns = line.strip('#').strip().split('\t')
                
                # Ensure the line has the correct number of columns
                if len(columns) < len(column_names):
                    raise ValueError(f"Invalid format in line: {line}")
                
                # Map the columns to the column names
                row_data = {col: value for col, value in zip(column_names, columns)}
                # Convert numerical fields to their proper types
                row_data["Index"] = int(row_data["Index"])
                row_data["XI"] = float(row_data["XI"])
                
                parsed_data.append(row_data)
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except ValueError as ve:
        print(f"Error parsing file: {ve}")

    return parsed_data


def preprocess_data(grids, verbose=0):
    """
    データの前処理を行う関数。

    Parameters:
        grids (dict): GRIDごとのデータフレーム辞書。
        verbose (int): 詳細情報の出力レベル。

    Returns:
        dict: 前処理後のデータフレーム辞書。
    """
    for grid_name, df in grids.items():
        columns_to_drop = []
        for column in df.columns[1:]:  # depth以外の列
            if (df[column] <= 0).all():  # 全て非正値の場合
                if verbose >= 1:
                    print(f"警告: {grid_name} の列 '{column}' は全て非正値のため除外されました。")
                columns_to_drop.append(column)
        df.drop(columns=columns_to_drop, inplace=True)
        for column in df.columns[1:]:
            df[column] = df[column].apply(lambda x: x if x > 0 else None)
        grids[grid_name] = df.dropna()
    return grids

def plot_data(grids, output_prefix, xi_list_from_file, debug=False, verbose=0):
    """
    データをプロットしてPNGファイルを生成する関数。

    Parameters:
        grids (dict): GRIDごとのデータフレーム辞書。
        output_prefix (str): 出力ファイルの接頭辞。
        debug (bool): プロットを表示するかどうか。
        verbose (int): 詳細情報の出力レベル。
        xi_list_from_file : xi obtained from grid file
    """
    num_colors = 27
    usercmap = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=num_colors)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=usercmap)

    for grid_name, df in grids.items():
        gridint = int(grid_name.replace("grid", ""))
        if df.empty or len(df.columns) <= 1:
            if verbose >= 1:
                print(f"警告: {grid_name} にはプロット可能なデータがありません。スキップします。")
            continue

        plt.figure(figsize=(12, 7))
        for i, column in enumerate(df.columns[1:]):  # depth以外の列をプロット
            if verbose >= 2:
                print(f"プロット列: {column}")
            if column == "Fe":
                ionnum = 0
            elif column == "Fe+":
                ionnum = 1
            else:
                ionnum = int(column.replace("Fe", "").replace("+", ""))
            c = scalarMap.to_rgba(ionnum)
            plt.plot(
                df['depth'], 
                df[column], 
                label=f"{column} zones={len(df['depth']):02d} mean={np.mean(df[column]):.6f}", 
                color=c
            )

        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-30, 10)
        plt.xlim(1e5, 1e14)
        plt.xlabel('Depth [cm]')
        plt.ylabel('Probability')
        plt.title(f'Ionic Probabilities for {grid_name}, XI={xi_list_from_file[gridint]}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=8)

        plt.tight_layout(rect=[0, 0.05, 0.95, 0.95])
        output_file = f"{output_prefix}_{grid_name}.png"
        plt.savefig(output_file)
        if verbose >= 1:
            print(f"保存: {output_file}")
        if debug:
            plt.show()
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="GRIDデータの解析とプロット")
    parser.add_argument('input_file', help="解析するファイル名")
    parser.add_argument('gridfile', help="グリッドのファイル名")

    parser.add_argument('--output_prefix', '-o', default="grid_plot", help="出力ファイル名の接頭辞")
    parser.add_argument('--debug', '-d', action='store_true', help="デバッグモード（プロットを表示）")
    parser.add_argument('--verbose', '-v', type=int, default=0, help="詳細出力レベル（0: 無し, 1: 基本, 2: 詳細）")

    args = parser.parse_args()

    # データ解析
    grids = parse_file(args.input_file, verbose=args.verbose)

    # gird ファイル解析
    simgrids = parse_grid_file(args.gridfile)
    xi_list_from_file = []
    for row in simgrids:
        xi_list_from_file.append(row["XI"])

    # データ前処理
    grids = preprocess_data(grids, verbose=args.verbose)

    # プロット
    plot_data(grids, args.output_prefix, xi_list_from_file, debug=args.debug, verbose=args.verbose)

if __name__ == "__main__":
    main()
