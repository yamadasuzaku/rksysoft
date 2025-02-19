#!/usr/bin/env python

import csv
import subprocess
import os
import argparse

def main():
    # argparse の設定
    parser = argparse.ArgumentParser(description="Decrypt files based on a CSV file.")
    parser.add_argument("--filename", "-f", type=str, default="data.csv", help="CSV file containing the decryption information.")
    args = parser.parse_args()

    # 指定された CSV ファイルを開く
    filename = args.filename
    if not os.path.exists(filename):
        print(f"Error: The specified file '{filename}' does not exist.")
        return

    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # ヘッダー行をスキップ

        for row in reader:
            if len(row) == 4:
                obsid, name, key, svpath = row

                # ディレクトリ作成
                if not os.path.exists(name):
                    os.makedirs(name)

                # カレントディレクトリを変更
                os.chdir(name)

                # decrypt_data.pl コマンドを構築
                bash_command = f'decrypt_data.pl -d ./ -p {key} -r'
                print("CMD: ", bash_command)

                # コマンドを実行
                try:
                    result = subprocess.run("pwd; date; ls -l", shell=True, check=True, capture_output=True, text=True)
                    print(f"Start for {name} ({obsid}): {result.stdout}")

                    result = subprocess.run(bash_command, shell=True, check=True, capture_output=True, text=True)
                    print(f"End   for {name} ({obsid}): {result.stdout}")

                except subprocess.CalledProcessError as e:
                    print(f"Error for {name} ({obsid}): {e.stderr}")

                # 元のディレクトリに戻る
                os.chdir('..')

if __name__ == "__main__":
    main()
