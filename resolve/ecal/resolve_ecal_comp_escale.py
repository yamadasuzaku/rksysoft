#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
params = {'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

import glob
import os

# ファイルのパスを取得
fitpoly_files = sorted(glob.glob('fitpoly_p*.csv'))
pixel_files = sorted(glob.glob('pixel_*.csv'))

# ファイル名からインデックスを抽出する関数
def extract_index(filename, prefix):
    return int(filename[len(prefix):-4])

# ファイルのペアごとに処理
for pixel_file in pixel_files:
    # 対応する fitpoly ファイルを推測
    pixel_index = extract_index(pixel_file, 'pixel_')
    fitpoly_file = f'fitpoly_p{pixel_index:02d}.csv'
    
    # 対応する fitpoly ファイルが存在する場合のみ処理
    if os.path.exists(fitpoly_file):
        print(f"Processing {fitpoly_file} and {pixel_file}")
        
        # データを読み込む
        fitpoly_df = pd.read_csv(fitpoly_file)
        pixel_df = pd.read_csv(pixel_file)
        
        # median でスケール
        median_px_temp_fit = pixel_df['px_temp_fit'].median()
        fitpoly_df['Ratio'] *= median_px_temp_fit
        
        # TIME と px_time でソート
        fitpoly_df = fitpoly_df.sort_values(by='TIME')
        pixel_df = pixel_df.sort_values(by='px_time')
        
        # プロットを作成
        plt.figure(figsize=(8, 6))
        plt.plot(fitpoly_df['TIME'], fitpoly_df['Ratio'], "o", label=f'Ratios Scaled by {median_px_temp_fit:.5f}')
        plt.plot(pixel_df['px_time'], pixel_df['px_temp_fit'], "o", label='TEMP_FIT from GHF')
        
        # プロットの装飾
        plt.xlabel('Time (sec)')
        plt.ylabel('TEMP_FIT')
        plt.title(f'Plot of {fitpoly_file} and {pixel_file}')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        # png ファイルとして保存
        output_file = f'plot_{fitpoly_file[:-4]}.png'
        plt.savefig(output_file)
        plt.close()
        
        print(f"Saved plot as {output_file}")
    else:
        print(f"Skipping {pixel_file} as {fitpoly_file} does not exist")
