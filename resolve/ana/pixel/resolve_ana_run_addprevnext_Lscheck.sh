#!/bin/bash

# 引数が指定されているか確認
if [ $# -eq 0 ]; then
    echo "Error: No input file specified."
    echo "Usage: $0 <input_file_uf.evt>"
    exit 1
fi

# 引数を受け取る
input_file=$1

# ファイル名の一部を抽出
base_name=$(basename "$input_file" .evt)
base_top=$(basename "$input_file" _uf.evt)

# コマンドを順に実行し、進行状況を表示
echo "Running resolve_tool_addcol_prev_next_interval.py..."
resolve_tool_addcol_prev_next_interval.py "${base_name}.evt" -o "${base_name}_prevnext.evt"
echo "Completed: resolve_tool_addcol_prev_next_interval.py"

echo "Running resolve_util_ftmgtime.sh..."
resolve_util_ftmgtime.sh "${base_top}_cl.evt"
echo "Completed: resolve_util_ftmgtime.sh"

echo "Running resolve_util_ftselect.sh..."
resolve_util_ftselect.sh "${base_name}_prevnext.evt" "gtifilter(\"${base_top}_cl.gti\")" cutclgti
echo "Completed: resolve_util_ftselect.sh"

echo "Running resolve_run_ana_pixel_Ls_mksubgroup.sh..."
resolve_run_ana_pixel_Ls_mksubgroup.sh "${base_name}_prevnext_cutclgti.fits"
echo "Completed: resolve_run_ana_pixel_Ls_mksubgroup.sh"

echo "All tasks completed."
