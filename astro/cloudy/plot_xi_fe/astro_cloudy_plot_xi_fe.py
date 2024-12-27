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
    ファイルを解析してデータを辞書形式に変換する関数

    Parameters:
        filename (str): 処理するファイル名
        verbose (int): 詳細情報の出力レベル（0: 非表示, 1: 基本, 2: 詳細）

    Returns:
        dict: GRIDごとのデータフレーム辞書
    """
    data = {}
    current_grid = None
    columns = None

    if verbose >= 1:
        print(f"ファイル '{filename}' を解析中...")

    with open(filename, 'r') as f:
        for line in f:
            if verbose >= 2:
                print(f"行内容: {line.strip()}")
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

    # 各GRIDのデータをDataFrameに変換
    grids = {grid_name: pd.DataFrame(grid_data, columns=columns)
             for grid_name, grid_data in data.items()}

    return grids


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


def extract_acol_array(grids, acol="depth"):
    """
    grids に含まれる "acol" 列をすべて結合して NumPy 配列として返す。

    Parameters:
        grids (dict): GRIDごとのデータフレーム辞書

    Returns:
        np.ndarray: 結合された "acol" 列の値
    """
    acol_values = []
    for grid_name, df in grids.items():
        if "depth" in df:
            acol_values.append(df[acol].values)  # "acol" 列を NumPy 配列として取得
        else:
            print(f"警告: {grid_name} に {acol} 列が見つかりません。")
    
    # リスト内の NumPy 配列を結合して1つの配列にする
    return np.concatenate(acol_values)

def plot_ionic_probabilities(iongrids, output_file, xi_list_from_file, verbose=0, show=False):
    """
    GRIDデータをプロットする関数

    Parameters:
        iongrids (dict): IONのGRIDごとのデータフレーム辞書
        output_file (str): 出力画像ファイル名
        xi_list (list) : list of xi 
        verbose (int): 詳細情報の出力レベル
    """
    num_colors = 27
    usercmap = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=num_colors)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=usercmap)

    plt.figure(figsize=(12, 7))

    grid0_df = iongrids.get('grid000000000')
    if grid0_df is None:
        print("警告: 'grid000000000' のデータが見つかりませんでした。")
        return

    for i, column in enumerate(grid0_df.columns[1:]):  # depth以外の列をプロット
        if column == "Fe":
            ionnum = 0
        elif column == "Fe+":
            ionnum = 1
        else:
            ionnum = int(column.replace("Fe", "").replace("+", ""))

        color = scalarMap.to_rgba(ionnum)
        xi_list = []
        ion_list = []

        for grid_name, df in iongrids.items():
            if df.empty or len(df.columns) <= 1:
                if verbose >= 1:
                    print(f"警告: {grid_name} にはプロット可能なデータがありません。スキップします。")
                continue
            if column not in df:
                if verbose >= 2:
                    print(f"列 '{column}' が見つかりません: {grid_name}")
                continue

            gridint = int(grid_name.replace("grid", ""))
            xi_list.append(xi_list_from_file[gridint])
            ion_list.append(df[column].values[0])

        plt.plot(xi_list, ion_list, "-", label=f"{column}", color=color)

    plt.xlabel(r'Ionization Parameter $\xi$')
    plt.ylabel('Probability')
    plt.title('Relative population of the ionization stages of iron as a function of the ionization parameter')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])
    plt.savefig(output_file)
    if verbose >= 1:
        print(f"プロットを '{output_file}' に保存しました。")
    if show:
        plt.show()
    plt.close()


def plot_basic(xi_list_from_file, te_list, output_file, verbose=0, lastcut=1, show=False):
    """
    GRIDデータをプロットする関数

    Parameters:
        iongrids (dict): IONのGRIDごとのデータフレーム辞書
        output_file (str): 出力画像ファイル名
        xi_list (list) : list of xi 
        verbose (int): 詳細情報の出力レベル
    """

    plt.figure(figsize=(12, 7))
    plt.plot(xi_list_from_file[:-lastcut], te_list, "-", label="Te")
    plt.xlabel(r'Ionization Parameter $\xi$')
    plt.ylabel('Electron Temperature [K]')
    plt.title('Electron Temperature as a function of the ionization parameter')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])
    plt.yscale("log")
    plt.grid(alpha=0.4)
    plt.savefig("te_" + output_file)
    if verbose >= 1:
        print(f"プロットを '{"te_" + output_file}' に保存しました。")
    if show:
        plt.show()
    plt.close()

def main():
    # 引数の設定
    parser = argparse.ArgumentParser(description="GRIDファイルを解析してプロットするツール")
    parser.add_argument('ionfile', help="イオンのファイル名")
    parser.add_argument('gridfile', help="グリッドのファイル名")
    parser.add_argument('ovrfile', help="ovr のファイル名")

    parser.add_argument('--output', '-o', default="xi_fe.png", help="出力PNGファイル名")
    parser.add_argument('--debug', '-d', action='store_true', help="デバッグ情報を表示")
    parser.add_argument('--verbose', '-v', type=int, default=0, help="詳細出力レベル (0: 無し, 1: 基本, 2: 詳細)")
    parser.add_argument('--show', '-s', action='store_true', help="プロットを表示するか")

    args = parser.parse_args()

    # イオンのファイル解析
    iongrids = parse_file(args.ionfile, verbose=args.verbose)

    # gird ファイル解析
    simgrids = parse_grid_file(args.gridfile)
    xi_list_from_file = []
    for row in simgrids:
        xi_list_from_file.append(row["XI"])

    # ovrのファイル解析
    ovrgrids = parse_file(args.ovrfile, verbose=args.verbose)
    te_list = extract_acol_array(ovrgrids, acol="Te")

    # プロット
    plot_ionic_probabilities(iongrids, args.output, xi_list_from_file, verbose=args.verbose, show=args.show)

    # プロット
    plot_basic(xi_list_from_file, te_list, args.output, verbose=args.verbose, show=args.show)

if __name__ == "__main__":
    main()
