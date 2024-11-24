import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# Plotting Configuration
plt.rcParams['font.family'] = 'serif'

# ファイル名
filename = 'cygx1_lhs_nh23.fe'

# GRID_DELIMITを検出しデータを格納
data = {}
current_grid = None
columns = None
with open(filename, 'r') as f:
    for line in f:
        # GRID_DELIMITを検出
        if line.startswith('########################### GRID_DELIMIT'):
            current_grid = re.search(r'grid\d+', line).group()
            data[current_grid] = []
        elif line.startswith('#depth'):
            # ヘッダー行を取得し、列名として設定
            columns = line.strip('#').strip().split('\t')  # '#'を除去して分割
        elif current_grid and columns:
            # データ行を追加
            values = line.strip().split('\t')
            data[current_grid].append([float(v) for v in values])

# 各GRIDのデータをDataFrameに変換
grids = {}
for grid_name, grid_data in data.items():
    grids[grid_name] = pd.DataFrame(grid_data, columns=columns)

# データ前処理: 全て非正値の列を削除
for grid_name, df in grids.items():
    columns_to_drop = []
    for column in df.columns[1:]:  # depth以外の列
        if (df[column] <= 0).all():  # 全て非正値の場合
            print(f"警告: {grid_name} の列 '{column}' は全て非正値のため除外されました。")
            columns_to_drop.append(column)
    # 非正値のみの列を削除
    df.drop(columns=columns_to_drop, inplace=True)
    # 非正値を含む行を削除
    for column in df.columns[1:]:
        df[column] = df[column].apply(lambda x: x if x > 0 else None)
    grids[grid_name] = df.dropna()

# 可視化
for grid_name, df in grids.items():
    gridint = int(grid_name.replace("grid",""))
    if df.empty or len(df.columns) <= 1:
        print(f"警告: {grid_name} にはプロット可能なデータがありません。スキップします。")
        continue
    plt.figure(figsize=(12, 7))
    for column in df.columns[1:]:  # depth以外の列をプロット
        plt.plot(df['depth'], df[column], label=f"{column} mean={np.mean(df[column]):2.6f}")
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-30,10)
    plt.xlim(1e5,1e14)
    plt.xlabel('Depth')
    plt.ylabel('Probability')
    plt.title(f'Ionic Probabilities for {grid_name}, XI={gridint*0.2:2.2f}')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.tight_layout()
    plt.savefig(f"{grid_name}.png")
    plt.show()
    plt.close()