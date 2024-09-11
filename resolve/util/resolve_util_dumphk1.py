#!/usr/bin/env python

import argparse
from astropy.io import fits

def get_column_data_and_types(fits_file, hdu_number):
    with fits.open(fits_file) as hdul:
        # 指定された HDU を取得
        hdu = hdul[hdu_number]
        
        # テーブルデータかどうか確認
        if isinstance(hdu, fits.BinTableHDU) or isinstance(hdu, fits.TableHDU):
            # columns 属性を取得して、データ型の set を作成
            column_types = {col.dtype for col in hdu.columns}
            print(f"Columns: {hdu.columns.names}")
            print(f"Data types in the columns (as set): {column_types}")
            
            # 各列のデータの set を表示
            for col in hdu.columns.names:
                data = hdu.data[col]
                try:
                    # 単純な型（数値や文字列）であればそのまま set に変換
                    column_data_set = set(data)
                except TypeError:
                    # numpy.ndarray などの場合、tuple に変換してから set にする
                    column_data_set = set(tuple(row) for row in data)
                
                print(f"Column: {col}, Data (as set): {column_data_set}")
        else:
            print(f"HDU {hdu_number} does not contain a table.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract columns, data types, and data from a FITS file.")
    parser.add_argument("fits_file", help="Path to the FITS file")
    parser.add_argument("--hdu", type=int, default=10, help="HDU number (default is 1)")
    
    args = parser.parse_args()
    
    get_column_data_and_types(args.fits_file, args.hdu)
