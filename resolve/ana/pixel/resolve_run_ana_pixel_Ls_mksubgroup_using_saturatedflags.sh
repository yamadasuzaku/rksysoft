#!/bin/sh

# スクリプトに渡された引数を取得
input_file="$1"

# 引数が渡されているか確認
if [ -z "$input_file" ]; then
  echo "Usage: $0 evtfile_with_PREV_INTERVAL"
  exit 1
fi

for i in `seq 0 35`
do
echo ".............. start"  
echo PIXEL = $i 
resolve_ana_pixel_Ls_mksubgroup_using_saturatedflags.py $input_file LO_RES_PH NEXT_INTERVAL -f "PIXEL==${i}"
echo ".............. finish"  
done
