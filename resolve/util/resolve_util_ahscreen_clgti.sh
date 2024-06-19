#!/bin/bash

# 引数が2つ指定されているか確認
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <infile> <gtifile>"
  exit 1
fi

# ファイル名を引数から取得
infile=$1
outfile="${infile%_uf.evt}_uf_clgti.evt"
gtifile=$2

# 出力ファイルが存在するか確認
if [ -e "$outfile" ]; then
  echo "$outfile already exists. Exiting without execution."
  exit 0
else
  echo "$outfile does not exist. Proceeding with execution."
fi

# ahscreen コマンドの実行
echo "Running ahscreen with infile=$infile, outfile=$outfile, gtifile=$gtifile"
ahscreen infile="$infile" outfile="$outfile" gtifile="${gtifile}[GTI]" expr=NONE label=GTI mergegti=AND debug=yes chatter=3

echo "ahscreen execution completed."
