#!/bin/bash

# 色の定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# スクリプトの使用方法を表示し、引数が3つ(uf, pruf, clevt)指定されているか確認
if [ "$#" -ne 3 ]; then
  echo -e "${RED}[ERROR]${NC} Usage: $0 <infile1_ufevt> <infile2_pruf> <infile3_clevt>"
  echo -e "${RED}[ERROR]${NC} e.g. $0 xa000114000rsl_p0px1000_uf.evt xa000114000rsl_a0pxpr_uf.evt xa000114000rsl_p0px1000_cl.evt"
  exit 1
fi

# 引数からファイル名を取得
infile1=$1
infile2=$2
infile3=$3

# スクリプトの開始を通知
echo -e "${GREEN}[START]${NC} ================================"
echo -e "${GREEN}[INFO]${NC} Processing files: $infile1, $infile2, $infile3"
echo -e "---------------------------------------\n"

# 引数のベース名を抽出し表示
echo -e "${BLUE}[INFO]${NC} Checking parameters..."

base_name1=$(basename "$infile1" .evt)
base_top1=$(basename "$infile1" _uf.evt)
echo -e "Base name 1: ${YELLOW}$base_name1${NC}, Top name 1: ${YELLOW}$base_top1${NC}"

base_name2=$(basename "$infile2" .evt)
base_top2=$(basename "$infile2" _uf.evt)
echo -e "Base name 2: ${YELLOW}$base_name2${NC}, Top name 2: ${YELLOW}$base_top2${NC}"

base_name3=$(basename "$infile3" .evt)
base_top3=$(basename "$infile3" _cl.evt)
echo -e "Base name 3: ${YELLOW}$base_name3${NC}, Top name 3: ${YELLOW}$base_top3${NC}"
echo -e "---------------------------------------\n"

# (1) prev/next interval を uf.evt で計算し、pruf.evt に map する。
echo -e "${BLUE}[INFO]${NC} Running resolve_tool_pr_prevnextadd.sh..."
resolve_tool_pr_prevnextadd.sh $infile1 $infile2 
echo -e "${GREEN}[DONE]${NC} resolve_tool_pr_prevnextadd.sh"

# 出力ファイルのパスを生成
outfile1="${infile1%_uf.evt}_uf_prevnext.evt"
outfile2="${infile2%_uf.evt}_uf_fillprenext.evt"

# 出力ファイルの存在確認
if [ -e "$outfile1" ]; then
  echo -e "${GREEN}[OK]${NC} $outfile1 is ready"
else
  echo -e "${RED}[ERROR]${NC} $outfile1 is not ready"
  exit 1
fi

if [ -e "$outfile2" ]; then
  echo -e "${GREEN}[OK]${NC} $outfile2 is ready"
else
  echo -e "${RED}[ERROR]${NC} $outfile2 is not ready"
  exit 1
fi

# (2) cl.evt の GTI ファイルを生成する。
echo -e "---------------------------------------"
echo -e "${BLUE}[INFO]${NC} Running resolve_util_ftmgtime.sh..."
resolve_util_ftmgtime.sh $infile3
echo -e "${GREEN}[DONE]${NC} resolve_util_ftmgtime.sh\n"

# GTI ファイルの生成と確認
outgti="${infile3%.evt}.gti"
if [ -e "$outgti" ]; then
  echo -e "${GREEN}[OK]${NC} $outgti is ready"
else
  echo -e "${RED}[ERROR]${NC} $outgti is not ready"
  exit 1
fi

# GTIフィルタリング処理
echo -e "---------------------------------------"
echo -e "${BLUE}[INFO]${NC} Running resolve_util_ftselect.sh..."

outfile_suffix=cutclgti

# (3-a) uf.evtの出力ファイルを cl.gti でフィルタリング
resolve_util_ftselect.sh $outfile1 "gtifilter(\"${outgti}\")" $outfile_suffix
outfile1_cutcl="${outfile1%.evt}_${outfile_suffix}.evt"

# フィルタリング結果の確認
if [ -e "$outfile1_cutcl" ]; then
  echo -e "${GREEN}[OK]${NC} $outfile1_cutcl is ready"
else
  echo -e "${RED}[ERROR]${NC} $outfile1_cutcl is not ready"
  exit 1
fi

# (3-b) pr.evtの出力ファイルを cl.gti でフィルタリング
resolve_util_ftselect.sh $outfile2 "gtifilter(\"${outgti}\")" $outfile_suffix
outfile2_cutcl="${outfile2%.evt}_${outfile_suffix}.evt"

# フィルタリング結果の確認
if [ -e "$outfile2_cutcl" ]; then
  echo -e "${GREEN}[OK]${NC} $outfile2_cutcl is ready"
else
  echo -e "${RED}[ERROR]${NC} $outfile2_cutcl is not ready"
  exit 1
fi

echo -e "${GREEN}[DONE]${NC} resolve_util_ftselect.sh"
echo -e "${GREEN}[END]${NC} ================================\n"
