#!/bin/bash

# スクリプトに渡された引数を取得
input_file="$1"

# 引数が渡されているか確認
if [ -z "$input_file" ]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi

# コマンドを実行
resolve_util_fplot.py "$input_file" TIME HE_TANK2,JT_SHLD2,CAMC_CT0,ADRC_CT_MON_FLUC --hdu 6 -p -m .
