#!/bin/bash

usage() {
    echo "Usage: $0 <file_list> [lower_limit] [upper_limit]"
    exit 1
}

# 引数の数をチェック
if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    usage
fi

# 引数の設定
file_list=$1
lower_limit=${2:-0}
upper_limit=${3:-35}

# f.list が存在するかチェック
if [ ! -f "$file_list" ]; then
    echo "Error: File '$file_list' not found!"
    exit 1
fi

# resolve_ana_pixel_mklc_branch.py の PATH チェック
if ! command -v resolve_ana_pixel_mklc_branch.py &> /dev/null; then
    echo "Error: resolve_ana_pixel_mklc_branch.py could not be found in your PATH"
    exit 1
fi

# 進行状況の表示とファイル生成
for pixel in $(seq $lower_limit $upper_limit)
do
    ppixel=$(printf "%02d" $pixel)
    echo "Processing pixel: $pixel (zero-padded: $ppixel)"
    resolve_ana_pixel_mklc_branch.py "$file_list" -l -g -p "${pixel}" -u -o "mklc_branch_pixel${ppixel}"
done

echo "Processing completed."
