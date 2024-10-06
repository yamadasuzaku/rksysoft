#!/usr/bin/env python

import requests
import argparse
import os

# argparse設定
def get_args():
    parser = argparse.ArgumentParser(description="Download a file from a specified URL and save it locally.")
    parser.add_argument("url", help="URL of the file to download")
    parser.add_argument("--output", help="Output filename (default is derived from the URL)", default=None)
    return parser.parse_args()

# ファイルをダウンロードして保存する関数
def download_file(url, output_filename):
    try:
        # ファイルをダウンロード
        response = requests.get(url)

        # ダウンロードが成功したか確認
        if response.status_code == 200:
            # ファイルに保存
            with open(output_filename, "wb") as file:
                file.write(response.content)
            print(f"ファイルが '{output_filename}' として保存されました。")
        else:
            print("ファイルのダウンロードに失敗しました。ステータスコード:", response.status_code)
    except requests.exceptions.RequestException as e:
        print(f"エラーが発生しました: {e}")

def main():
    # 引数を取得
    args = get_args()

    # 出力ファイル名を設定
    if args.output:
        output_filename = args.output
    else:
        # URLからファイル名を取得
        output_filename = os.path.basename(args.url)

    # ファイルをダウンロード
    download_file(args.url, output_filename)

if __name__ == "__main__":
    main()
