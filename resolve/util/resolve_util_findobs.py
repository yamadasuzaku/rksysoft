#!/usr/bin/env python

import pandas as pd
import argparse
import os

def search_csv(search_term, search_by, output_columns, no_header):
    # 環境変数からCSVファイルのパスを取得
    file_path = os.getenv('XRISM_CSV_FILE_PATH')
    if not file_path:
        raise ValueError("Environment variable 'XRISM_CSV_FILE_PATH' is not set.")
    
    # CSVファイルの読み込み
    df = pd.read_csv(file_path,dtype={'observation_id': str})
    
    # 検索条件に基づいてデータをフィルタリング
    if search_by == 'observation_id':
        result = df[df['observation_id'] == search_term]
    elif search_by == 'object_name':
        result = df[df['object_name'].str.contains(search_term, case=False, na=False)]
    else:
        raise ValueError("Invalid search_by value. Use 'observation_id' or 'object_name'.")
    
    # 出力する項目のフィルタリング
    if output_columns:
        result = result[output_columns.split(",")]
    
    # 結果を標準出力
    if not result.empty:
        print(result.to_string(index=False, header=not no_header))
    else:
        print("No matching records found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Search CSV by observation_id or object_name.',
        epilog='Examples of usage:\n'
               '  resolve_util_findobs.py CYGNUS_X-1 object_name --output_columns observation_id\n'
               '  resolve_util_findobs.py 300049010 observation_id --output_columns object_name\n'
               '  resolve_util_findobs.py 300049010 observation_id --output_columns center_ra,center_dec',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('search_term', type=str, help='Search term for observation_id or object_name.')
    parser.add_argument('search_by', type=str, choices=['observation_id', 'object_name'], help='Search by observation_id or object_name.')
    parser.add_argument('--output_columns', '-o', type=str, help='Comma-separated list of columns to output.', default=None)
    parser.add_argument('--no_header', '-n', action='store_true', help='Do not print the header (column names).')
 
    
    args = parser.parse_args()
    
    search_csv(args.search_term, args.search_by, args.output_columns, args.no_header)

