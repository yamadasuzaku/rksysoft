#!/bin/sh

. "$(dirname "$0")/resolve_env.sh"

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Error: Invalid number of arguments."
    cat << EOF
Usage: resolve_util_ftselect.sh eventfile expr [output_suffix]

Description:
  This script is used to perform selection on an event file using the ftselect command.

Arguments:
  eventfile        Path to the event file.
  expr             Selection expression.
  output_suffix    (Optional) Suffix for the output file name.
                   If provided, the output file name will be eventfile_\${output_suffix}.fits.
                   If not provided, the default suffix is '_cut'.

Example:
  resolve_util_ftselect.sh test_pxpr.evt "gtifilter(\"xa300065010rsl_px5000_exp.gti\")"
  resolve_util_ftselect.sh xa300065010rsl_a0pxpr_uf.evt "FLAG_CLIPPED==b0&&gtifilter(\"xa300065010rsl_px1000_exp.gti\")" clip0gtipx1000
EOF
    exit 1
fi

obs="$1"
expr="$2"
outfile_suffix="_cut"

# 3番目の引数があれば、出力ファイルのサフィックスに設定
if [ $# -eq 3 ]; then
    outfile_suffix="_$3"
fi

# 拡張子とファイル名の本体を分けて取得
extention="${obs##*.}"
basename="${obs%.*}"

# 出力ファイル名の生成
outfile="${basename}${outfile_suffix}.${extention}"

# デバッグ用の情報表示
echo "Selected event file: $obs"
echo "Selection expression: $expr"
echo "Output file: $outfile"
echo "Extension: $extention"


echo "which ftselect: $(command -v ftselect)"
echo "HEADAS=$HEADAS"
echo "PATH=$PATH"
echo "DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"
echo "DYLD_FALLBACK_LIBRARY_PATH=$DYLD_FALLBACK_LIBRARY_PATH"

# ftselectコマンドの実行
ftselect \
    "infile=$obs" \
    "outfile=$outfile" \
    "expr=$expr" \
    "clobber=yes" \
    "chatter=5"

echo "Output written to $outfile"
