#!/usr/bin/env python 

import argparse
from astropy.io import fits
import numpy as np
import os

# コマンドライン引数の設定
def parse_args():
    parser = argparse.ArgumentParser(description='Generate a new column with differences in a FITS file.')
    parser.add_argument('fits_file', type=str, help='Input FITS file name')
    parser.add_argument('--cname', '-c', type=str, default='TIME', help='Column name to calculate differences (default: TIME)')
    return parser.parse_args()

# FITSファイルの差分を計算し、新しい列として追加
def calculate_diff_column(fits_file, cname):
    # FITSファイルを開く
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        # 指定したコラムから差分を計算
        column_data = data[cname]
        diff_data = np.diff(column_data, prepend=column_data[0])  # 最初の行はプレースホルダー
        
        # 新しいカラムを作成
        new_col_name = f"{cname}_DIFF"
        cols = hdul[1].columns
        new_col = fits.ColDefs([fits.Column(name=new_col_name, format='E', array=diff_data)])
        new_hdu = fits.BinTableHDU.from_columns(cols + new_col)
        
        # 新しいファイル名を作成
        output_filename = f"{cname}_DIFF_{os.path.basename(fits_file)}"
        new_hdu.writeto(output_filename, overwrite=True)
        print(f'FITS file with {new_col_name} created: {output_filename}')

# メイン関数
if __name__ == '__main__':
    args = parse_args()
    calculate_diff_column(args.fits_file, args.cname)
