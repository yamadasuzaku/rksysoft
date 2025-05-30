import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm

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
num = 27 # max of color 
usercmap = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=0, vmax=num)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=usercmap)

for grid_name, df in grids.items():
    gridint = int(grid_name.replace("grid",""))
    if df.empty or len(df.columns) <= 1:
        print(f"警告: {grid_name} にはプロット可能なデータがありません。スキップします。")
        continue
    plt.figure(figsize=(12, 7))
    for i, column in enumerate(df.columns[1:]):  # depth以外の列をプロット
        print(i, column)
        if column == "Fe":
            ionnum = 0
        elif column == "Fe+":
            ionnum = 1        
        else:
            ionnum = int(column.replace("Fe","").replace("+",""))
        c = scalarMap.to_rgba(ionnum)
        plt.plot(df['depth'], df[column], label=f"{column} zones={len(df['depth']):02d} mean={np.mean(df[column]):2.6f}", color=c)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-30,10)
    plt.xlim(1e5,1e14)
    plt.xlabel('Depth')
    plt.ylabel('Probability')
    plt.title(f'Ionic Probabilities for {grid_name}, XI={gridint*0.2:2.2f}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=8)

    # Adjust layout to minimize gaps
    plt.tight_layout(rect=[0, 0.01, 0.8, 0.90])
    plt.savefig(f"{grid_name}.png")
    plt.show()
    plt.close()