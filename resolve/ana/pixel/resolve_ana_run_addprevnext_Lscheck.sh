#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Error: No input file specified."
    echo "Usage: $0 <input_file_uf.evt>"
    exit 1
fi

input_file=$1
base_top=$(basename "$input_file" _uf.evt)

echo "Running resolve_util_ftselect.sh to remove BL"
resolve_util_ftselect.sh ${input_file} "ITYPE<5" noBL
echo "Completed: resolve_util_ftselect.sh to remove BL"

base_name=$(basename "$input_file" .evt)_noBL

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
resolve_run_ana_pixel_Ls_mksubgroup.sh "${base_name}_prevnext_cutclgti.evt"
echo "Completed: resolve_run_ana_pixel_Ls_mksubgroup.sh"

echo "All tasks completed."
