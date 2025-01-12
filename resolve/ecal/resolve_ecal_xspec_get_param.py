#!/usr/bin/env python 

import argparse
import csv

def extract_value_with_split(filename, search_text, npara=1):
    """
    指定された文字列を検索し、その次の数値を抽出する関数。

    :param filename: 読み込むファイルの名前
    :param search_text: 検索する文字列
    :param npara: 抽出するパラメータの数 (デフォルトは1)
    :return: 抽出された数値（該当しない場合はNone）
    """
    with open(filename, 'r') as file:
        for line in file:
            # 検索文字列が行に含まれている場合
            if search_text in line:
                parts = line.split()  # 行を空白で分割
                try:
                    # 検索文字列の最後の要素の次のインデックスを取得
                    search_index = parts.index(search_text.split()[-1]) + 1
                    # 指定された数の数値を抽出
                    extracted_values = [float(parts[search_index + i]) for i in range(npara)]
                    return extracted_values if npara > 1 else extracted_values[0]
                except (ValueError, IndexError):
                    # エラー発生時は None を返す
                    return None
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract values from a file based on search text.")
    parser.add_argument("filename", type=str, help="Name of the file to process.")
    parser.add_argument('--plotstr', '-p', type=str, help='pixel for output file', default='12')    

    args = parser.parse_args()

    filename = args.filename
    outputtext = filename.replace(".fitlog", "_fitresult.csv")

    results = []

    # Sigma
    search_text = '#   1    1   gsmooth    Sig_6keV   keV'
    value = extract_value_with_split(filename, search_text)
    if value is not None:
        search_text = '#     1'
        value_e = extract_value_with_split(filename, search_text, npara=2)
        if value_e is not None:
            sigma = value
            sigma_ne = value - value_e[0]
            sigma_pe = value_e[1] - value
            results.append(["Sigma", sigma, sigma_ne, sigma_pe])

    # Const
    search_text = '#  27   10   constant   factor'
    value = extract_value_with_split(filename, search_text)
    if value is not None:
        search_text = '#    27'
        value_e = extract_value_with_split(filename, search_text, npara=2)
        if value_e is not None:
            const = value
            const_ne = value - value_e[0]
            const_pe = value_e[1] - value
            results.append(["Const", const, const_ne, const_pe])

    # Gain
    search_text = '#   2     1    gain     offset'
    value = extract_value_with_split(filename, search_text)
    if value is not None:
        search_text = '#     2'
        value_e = extract_value_with_split(filename, search_text, npara=2)
        if value_e is not None:
            gain = value
            gain_ne = value - value_e[0]
            gain_pe = value_e[1] - value
            results.append(["Gain", gain, gain_ne, gain_pe])

    # Write results to CSV
    with open(outputtext, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Parameter", "Value", "Negative Error", "Positive Error"])
        csvwriter.writerows(results)

    print(f"フィット結果が {outputtext} に保存されました。")
