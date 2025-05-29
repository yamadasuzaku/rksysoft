#!/bin/bash

# 引数のチェック
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <event_file>"
    exit 1
fi

# 引数からイベントファイルを取得
EVENT_FILE="$1"

# ファイルが存在するかチェック
if [ ! -f "$EVENT_FILE" ]; then
    echo "Error: File '$EVENT_FILE' not found!"
    exit 1
fi

#
## Hp, Mp, Ms
#for pixel in {0..35}; do
#for itype in 0 1 2; do
#    pixel_str=__pixel$(printf "%02d" $pixel)_itype$(printf "%01d" $itype)__  # 2桁ゼロパディング
#    echo "Processing Pixel $pixel..."
#./resolve_util_fploteve.py $EVENT_FILE DERIV_MAX 1,1,1,1,1,1,1,1 LO_RES_PH,PHA,EPI,RISE_TIME,TICK_SHIFT,ITYPE,PIXEL 1,1,1,1,1,1,1 --filters "PIXEL==${pixel},ITYPE==${itype}" -o ${pixel_str} -c --y_ranges '{"PHA": (0, 65535)}'
#    echo "Completed Pixel $pixel."
#done
#done 
#
## Lp 
#for pixel in {0..35}; do
#for itype in 3; do
#    pixel_str=__pixel$(printf "%02d" $pixel)_itype$(printf "%01d" $itype)__  # 2桁ゼロパディング
#    echo "Processing Pixel $pixel..."
#./resolve_util_fploteve.py $EVENT_FILE DERIV_MAX 1,1,1,1,1,1,1,1 LO_RES_PH,PHA,EPI,RISE_TIME,ITYPE,NEXT_INTERVAL,PIXEL 1,1,1,1,1,1,1,1 --filters "PIXEL==${pixel},ITYPE==${itype}" -o ${pixel_str} -c
#    echo "Completed Pixel $pixel."
#done
#done 
#
## Ls
#for pixel in {0..35}; do
#for itype in 4; do
#    pixel_str=__pixel$(printf "%02d" $pixel)_itype$(printf "%01d" $itype)__  # 2桁ゼロパディング
#    echo "Processing Pixel $pixel..."
#./resolve_util_fploteve.py $EVENT_FILE DERIV_MAX 1,1,1,1,1,1,1 LO_RES_PH,PHA,EPI,RISE_TIME,ITYPE,PIXEL 1,1,1,1,1,1,1 --filters "PIXEL==${pixel},ITYPE==${itype}" -o ${pixel_str} -c
#    echo "Completed Pixel $pixel."
#done
#done 
#

##########################
# large EPI 
##########################

# Hp, Mp, Ms
for pixel in {0..35}; do
for itype in 0 1 2; do
    pixel_str=__epiwide_pixel$(printf "%02d" $pixel)_itype$(printf "%01d" $itype)__  # 2桁ゼロパディング
    echo "Processing Pixel $pixel..."
./resolve_util_fploteve.py $EVENT_FILE DERIV_MAX 1,1,1,1,1,1,1,1 LO_RES_PH,PHA,EPI,RISE_TIME,TICK_SHIFT,ITYPE,PIXEL 1,1,1,1,1,1,1 --filters "PIXEL==${pixel},ITYPE==${itype}" -o ${pixel_str} -c --y_ranges '{"PHA": (0, 65535),"EPI":(0,50000)}'
    echo "Completed Pixel $pixel."
done
done 

# Lp 
for pixel in {0..35}; do
for itype in 3; do
    pixel_str=__epiwide_pixel$(printf "%02d" $pixel)_itype$(printf "%01d" $itype)__  # 2桁ゼロパディング
    echo "Processing Pixel $pixel..."
./resolve_util_fploteve.py $EVENT_FILE DERIV_MAX 1,1,1,1,1,1,1,1 LO_RES_PH,PHA,EPI,RISE_TIME,NEXT_INTERVAL,ITYPE,PIXEL 1,1,1,1,1,1,1,1 --filters "PIXEL==${pixel},ITYPE==${itype}" -o ${pixel_str} -c --y_ranges '{"PHA": (0, 16383),"EPI":(0,50000)}'
    echo "Completed Pixel $pixel."
done
done 

# Ls
for pixel in {0..35}; do
for itype in 4; do
    pixel_str=__epiwide_pixel$(printf "%02d" $pixel)_itype$(printf "%01d" $itype)__  # 2桁ゼロパディング
    echo "Processing Pixel $pixel..."
./resolve_util_fploteve.py $EVENT_FILE DERIV_MAX 1,1,1,1,1,1,1,1 LO_RES_PH,PHA,EPI,RISE_TIME,PREV_INTERVAL,ITYPE,PIXEL 1,1,1,1,1,1,1 --filters "PIXEL==${pixel},ITYPE==${itype}" -o ${pixel_str} -c --y_ranges '{"PHA": (0, 65535),"EPI":(0,50000)}'
    echo "Completed Pixel $pixel."
done
done 
#

echo "All pixels processed successfully!"
