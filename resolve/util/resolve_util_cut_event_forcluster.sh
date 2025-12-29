#!/bin/bash
. "$(dirname "$0")/heasoft_env.sh"

# 色の定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 関数：使い方の説明を表示
usage() {
    cat << EOF
${YELLOW}Usage:${NC} $0 infile

${YELLOW}Description:${NC}
  This script is to split files using IMEMBER

${YELLOW}Arguments:${NC}
  ${GREEN}infile${NC}       Path to the input event file.

${YELLOW}Example:${NC}
  $0 event.fits
EOF
    exit 1
}

# 引数の数を確認
if [ $# -lt 1 ]; then
    echo -e "${RED}[ERROR]${NC} Invalid number of arguments."
    usage
fi

# 引数の取得
infile="$1"

# 入力されたパラメータを表示
echo -e "${BLUE}[INFO]${NC} Input file: ${YELLOW}$infile${NC}"

echo -e "${GREEN}[RUN]${NC} resolve_util_ftselect_split_file.py FLAG: QD=${YELLOW}$qd${NC} SD=${YELLOW}$sd${NC} "

ofileim0=im0_${infile}
ofileim1=im1_${infile}
ofileim1a=imabove1_${infile}
ofileim2a=imabove2_${infile}

ftselect infile=${infile} outfile=${ofileim0} expr="IMEMBER==0" chatter=5 clobber=yes
ftselect infile=${infile} outfile=${ofileim1} expr="IMEMBER==1" chatter=5 clobber=yes
ftselect infile=${infile} outfile=${ofileim1a} expr="IMEMBER>0" chatter=5 clobber=yes
ftselect infile=${infile} outfile=${ofileim2a} expr="IMEMBER>1" chatter=5 clobber=yes
