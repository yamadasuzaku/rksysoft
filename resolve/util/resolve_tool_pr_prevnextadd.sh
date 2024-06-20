#!/bin/bash

# 引数が2つ指定されているか確認
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <infile1> <infile2>"
  echo ex) resolve_tool_pr_prevnextadd.sh xa000114000rsl_p0px1000_uf.evt xa000114000rsl_a0pxpr_uf.evt
  exit 1
fi

# ファイル名を引数から取得
infile1=$1
infile2=$2
outfile1="${infile1%_uf.evt}_uf_prevnext.evt"
outfile2="${infile2%_uf.evt}_uf_fillprenext.evt"

# 1) Create a new file with PREV/NEXT INTERVAL
if [ -e "$outfile1" ]; then
  echo "$outfile1 already exists. Skipping creation."
else
  echo "Creating $outfile1 with PREV/NEXT INTERVAL."
  resolve_tool_addcol_prev_next_interval.py "$infile1" -o "$outfile1"
fi

# 2) Create a new target FITS file with PREV/NEXT INTERVAL
if [ -e "$outfile2" ]; then
  echo "$outfile2 already exists. Skipping creation."
else
  echo "Creating $outfile2 with PREV/NEXT INTERVAL."
  resolve_tool_map_prevnextinterval.py "$outfile1" "$infile2" -o "$outfile2"
fi

echo "Script execution completed."
