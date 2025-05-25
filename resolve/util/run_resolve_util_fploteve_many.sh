#!/bin/bash

# 引数のチェック
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <event_file>"
    exit 1
fi

# 引数からイベントファイルを取得
EVENT_FILE="$1"
#EVENT_FILE=addcluster_xa000114000rsl_p0px1000_uf_noBL_prevnext_cutclgti.evt

# ファイルが存在するかチェック
if [ ! -f "$EVENT_FILE" ]; then
    echo "Error: File '$EVENT_FILE' not found!"
    exit 1
fi

# pixel 0 から 35 までループ
for pixel in {0..35}; do
for itype in {0..5}; do
    pixel_str=__pixel$(printf "%02d" $pixel)_itype$(printf "%01d" $itype)__  # 2桁ゼロパディング
    echo "Processing Pixel $pixel..."
    resolve_util_fploteve.py $EVENT_FILE DERIV_MAX 1,1,1,1,1,1,1 LO_RES_PH,PHA,EPI,RISE_TIME,ITYPE,PIXEL 1,1,1,1,1,1,1 --filters "PIXEL==${pixel},ITYPE==${itype}" -o ${pixel_str} -c
    echo "Completed Pixel $pixel."
done
done 
echo "All pixels processed successfully!"
