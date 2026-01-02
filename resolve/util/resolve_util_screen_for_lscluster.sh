#!/usr/bin/env bash
set -Eeuo pipefail

. "$(dirname "$0")/heasoft_env.sh"

# -------------------------
# Color / Logging
# -------------------------
GREEN=$(printf '\033[32m')
RED=$(printf '\033[31m')
CYAN=$(printf '\033[36m')
RESET=$(printf '\033[0m')

log_info()  { printf "${CYAN}[INFO]${RESET} %s\n" "$*"; }
log_ok()    { printf "${GREEN}[OK]${RESET} %s\n" "$*"; }
log_warn()  { printf "${RED}[WARN]${RESET} %s\n" "$*"; }
die()       { printf "${RED}[ERROR]${RESET} %s\n" "$*"; exit 1; }

# -------------------------
# Utils
# -------------------------
check_program_in_path() {
  local prog="$1"
  command -v "$prog" >/dev/null 2>&1 || die "$prog not found in \$PATH"
  log_ok "$prog is found in \$PATH"
}

usage() {
  cat <<EOF
Usage:
  $(basename "$0") <input_ufevt.evt> <cl.evt>

Example:
  $(basename "$0") small_large_xa000154000rsl_p0px1000_uf_noBL_prevnext_cutclgti.evt xa000154000rsl_p0px1000_cl.evt
EOF
  exit 1
}

require_file() {
  local f="$1"
  [[ -f "$f" ]] || die "Input file does not exist: $f"
}

# フィルタ文字列を「1行化」し、さらに PIL 誤解釈を避けるため「空白を全削除」まで行う
normalize_filter_expr() {
  # stdin -> stdout
  tr '\n' ' ' \
    | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//' \
    | tr -d ' '
}

# -------------------------
# HEASOFT tasks wrappers
# -------------------------
run_xselect_make_clgti() {
  local infile_evt="$1"
  local outfile_evt="$2"
  local gtifile="$3"


  log_info "Running xselect to create clgti event: $outfile_evt"
  rm -f "$outfile_evt"

  # 注意：xselect は対話コマンド。最小限の所作だけを書く。
  xselect <<EOF
xsel
no

read event
./
${infile_evt}

filter time file ${gtifile}

show filter

extract event
save events ${outfile_evt}
no
exit
no
EOF

  [[ -f "$outfile_evt" ]] || die "xselect failed to create: $outfile_evt"
  log_ok "Completed: xselect -> $outfile_evt"
}

run_ftcopy_with_filter() {
  local base_evt="$1"   # already GTI-merged event file
  local outfile_evt="$2"
  local filter_raw="$3" # multi-line human readable filter

  local filter_clean
  filter_clean="$(printf '%s' "$filter_raw" | normalize_filter_expr)"

  log_info "ftcopy -> $outfile_evt"
  # PILの誤解釈を避けるため、infile= を含む文字列を 1 引数として渡す
  ftcopy \
    "infile=${base_evt}[EVENTS][${filter_clean}]" \
    "outfile=${outfile_evt}" \
    copyall=yes clobber=yes history=yes chatter=5

  [[ -f "$outfile_evt" ]] || die "ftcopy failed to create: $outfile_evt"
  log_ok "Created: $outfile_evt"
}

# ★追加：cl.evt から GTI を生成して、中身があることまでチェック
run_make_gti_from_clevt() {
  local clevt="$1"

  log_info "Generating GTI from cl.evt: $clevt"
  require_file "$clevt"

  # cl.evt の GTI ファイルを生成する。
  resolve_util_ftmgtime.py "$clevt"

  # GTI ファイルの生成と確認
  local outgti="${clevt%.evt}.gti"
  if [[ -s "$outgti" ]]; then
    log_ok "$outgti is ready (non-empty)"
  else
    if [[ -e "$outgti" ]]; then
      die "$outgti exists but is EMPTY"
    else
      die "$outgti is not ready"
    fi
  fi
}

# -------------------------
# Main
# -------------------------
main() {
  log_info "[start] resolve_util_screen_for_lscluster.sh"

  # Required programs
  check_program_in_path "ftlist"
  check_program_in_path "ftcopy"
  check_program_in_path "xselect"
  check_program_in_path "resolve_util_ftmgtime.py"

  [[ $# -eq 2 ]] || usage

  local ufevt="$1"
  local clevt="$2"
  require_file "$ufevt"
  require_file "$clevt"

  # ★追加：clevt から gti 生成＆中身チェック
  run_make_gti_from_clevt "$clevt"

  local gtifile="${clevt%.evt}.gti"
  local ufclgtievt="${ufevt%.evt}_clgti.evt"
  local base_top
  base_top="$(basename "$ufevt" .evt)"

  # 1) xselectでベースイベント作成（GTI整形）
  run_xselect_make_clgti "$ufevt" "$ufclgtievt" "$gtifile"

  log_info "Input check: $ufclgtievt"
  ftlist "$ufclgtievt" H

  # -------------------------
  # Filters (human readable)
  # ここに “条件” を増やしていけばよい
  # -------------------------
  local FILTER_STDCUT='
(ITYPE < 5) &&
((SLOPE_DIFFER == b0 || PI > 22000)) &&
(QUICK_DOUBLE == b0) &&
(STATUS[2] == b0) &&
(STATUS[3] == b0) &&
(STATUS[4] == b0) &&
(STATUS[6] == b0) &&
(PI >= 600) &&
(RISE_TIME < 127) &&
(PIXEL != 12) &&
(TICK_SHIFT > -8 && TICK_SHIFT < 7) &&
(
  (
    ((RISE_TIME + 0.00075 * DERIV_MAX) > 46) &&
    ((RISE_TIME + 0.00075 * DERIV_MAX) < 58) &&
    (ITYPE < 4)
  ) ||
  (ITYPE == 4)
)
'

  local FILTER_CLUSTERCUT='
(ITYPE < 5) &&
(ICLUSTERL == 0) &&
(ICLUSTERS == 0)
'

  local FILTER_CLUSTERCUTSTDCUT="
(
${FILTER_STDCUT}
)
&&
(
${FILTER_CLUSTERCUT}
)
"

local FILTER_NOT_CLUSTERCUT="
(ITYPE < 5) &&
(
  (ICLUSTERL != 0) ||
  (ICLUSTERS != 0)
)
"

  # -------------------------------
  # Outputs table: name <-> filter
  # -------------------------------
  declare -a OUT_FILES=(
    "${ufevt%.evt}_clgti_stdcut.evt"
    "${ufevt%.evt}_clgti_clustercut.evt"
    "${ufevt%.evt}_clgti_clustercutstdcut.evt"
    "${ufevt%.evt}_clgti_NOT_clustercut.evt"
  )

  declare -a OUT_FILTERS=(
    "$FILTER_STDCUT"
    "$FILTER_CLUSTERCUT"
    "$FILTER_CLUSTERCUTSTDCUT"
    "$FILTER_NOT_CLUSTERCUT"
  )

  # 実行（まとめて生成）
  local i
  for i in "${!OUT_FILES[@]}"; do
    local out="${OUT_FILES[$i]}"
    local fil="${OUT_FILTERS[$i]}"

    log_info ">>>>>>>>>>>> Generating output: $out >>>>>>>>>>>>>>"
    run_ftcopy_with_filter "$ufclgtievt" "$out" "$fil"
    ftlist "$out" H
  done

  log_ok "[done] all outputs generated"
}

main "$@"
