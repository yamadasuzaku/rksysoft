#!/bin/bash

# 色の定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 関数：使い方の説明を表示
usage() {
    cat << EOF
${YELLOW}Usage:${NC} $0 infile outfile expr [chatter] [clobber]

${YELLOW}Description:${NC}
  This script is used to perform selection on an event file using the ftselect command. 
  The difference from resolve_util_ftselect.sh is only it can accept the outfile name. 

${YELLOW}Arguments:${NC}
  ${GREEN}infile${NC}       Path to the input event file.
  ${GREEN}outfile${NC}      Path to the output event file.
  ${GREEN}expr${NC}         Selection expression. Should include conditions for PIXEL, QUICK_DOUBLE, SLOPE_DIFFER, ITYPE.
  ${GREEN}chatter${NC}      (Optional) Level of verbosity for the command output (default: 5).
  ${GREEN}clobber${NC}      (Optional) Whether to overwrite the output file if it exists (default: yes).

${YELLOW}Example:${NC}
  $0 xa000114000rsl_a0pxpr_uf.evt xa000114000rsl_a0pxpr_uf_cut.evt "(PIXEL==5)&&(QUICK_DOUBLE==b01)&&(SLOPE_DIFFER==b01)&&(ITYPE==3)"
  $0 xa000114000rsl_a0pxpr_uf.evt xa000114000rsl_a0pxpr_uf_cut.evt "(PIXEL==5)&&(QUICK_DOUBLE==b01)&&(SLOPE_DIFFER==b01)&&(ITYPE==3)" 5 no
EOF
    exit 1
}

# 引数の数を確認
if [ $# -lt 3 ]; then
    echo -e "${RED}[ERROR]${NC} Invalid number of arguments."
    usage
fi

# 引数の取得
infile="$1"
outfile="$2"
expr="$3"
chatter="${4:-5}"    # chatterのデフォルト値は5
clobber="${5:-yes}"  # clobberのデフォルト値はyes

# 入力されたパラメータを表示
echo -e "${BLUE}[INFO]${NC} Input file: ${YELLOW}$infile${NC}"
echo -e "${BLUE}[INFO]${NC} Output file: ${YELLOW}$outfile${NC}"
echo -e "${BLUE}[INFO]${NC} Selection expression: ${YELLOW}$expr${NC}"
echo -e "${BLUE}[INFO]${NC} Chatter level: ${YELLOW}$chatter${NC}"
echo -e "${BLUE}[INFO]${NC} Clobber: ${YELLOW}$clobber${NC}"

# ftselect コマンドの実行
echo -e "${GREEN}[RUNNING]${NC} Executing ftselect..."
ftselect \
    "infile=$infile" \
    "outfile=$outfile" \
    "expr=$expr" \
    "chatter=$chatter" \
    "clobber=$clobber"

# 結果の表示
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS]${NC} Output written to ${YELLOW}$outfile${NC}"
else
    echo -e "${RED}[ERROR]${NC} Failed to execute ftselect. Check the parameters and try again."
fi
