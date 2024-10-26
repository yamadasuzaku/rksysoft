#!/bin/bash

# スクリプトに渡された引数を取得
input_file="$1"

# 引数が渡されているか確認
if [ -z "$input_file" ]; then
  echo "(sigle file) Usage: $0 <input_file>"
  echo "(more than two file) input comma-separeted way"
  echo "resolve_hk_plot_temptrend.sh xa300049010rsl_a0.hk1,xa300041010rsl_a0.hk1"    
  exit 1
fi

# コマンドを実行
#resolve_util_fplot.py "$input_file" TIME HE_TANK2,JT_SHLD2,CAMC_CT0,ADRC_CT_MON_FLUC --hdu 6 -p -m .
resolve_util_fplot.py $input_file TIME 7,6,6,6,6 XBOXA_TEMP3_CAL,HE_TANK2,JT_SHLD2,CAMC_CT1,ADRC_CT_MON_FLUC 7,6,6,6,6  -m . -s linear,linear,linear,log,log
