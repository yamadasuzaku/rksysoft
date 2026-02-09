#!/usr/bin/env bash
#
# run_lscheck_pipeline.sh
#
# 通常モード:
#   ./run_lscheck_pipeline.sh xa000154000rsl_p0px1000_uf.evt [lscheck_dir_name] [CAL_ROOT_DIR] [EVE_ARGS] [CLUSTERCUT_ARGS]
#
# Summary のみ再生成モード:
#   ./run_lscheck_pipeline.sh --summary-only xa000154000rsl_p0px1000_uf.evt [lscheck_dir_name] [CAL_ROOT_DIR]
#
# 期待する初期位置: event_uf ディレクトリ
#
# DEBUG_LSCHECK=1 を環境変数にセットすると内部状態を色々出力します。
#

set -euo pipefail

DEBUG="${DEBUG_LSCHECK:-0}"

debug_echo() {
    if [[ "$DEBUG" == "1" ]]; then
        echo "[DEBUG] $*" >&2
    fi
}

# ========================================
# ヘルプ表示
# ========================================

print_help() {
    cat <<EOF
run_lscheck_pipeline.sh : LS-cluster 解析 + mklc_branch + summary HTML のワンコマンド実行ラッパ

Usage:
  フルパイプライン:
    $(basename "$0") <uf_event_file> [lscheck_dir_name] [CAL_ROOT_DIR] [EVE_ARGS] [CLUSTERCUT_ARGS]

  Summary のみ再生成:
    $(basename "$0") --summary-only <uf_event_file> [lscheck_dir_name] [CAL_ROOT_DIR]

引数:
  uf_event_file
      event_uf ディレクトリにある UF イベントファイル名
      例: xa000154000rsl_p0px1000_uf.evt

  lscheck_dir_name (任意)
      CAL_ROOT_DIR の配下に作成する lscheck 用ディレクトリ名。
      省略時は "lscheck_YYYYMMDD" (実行日の年月日) となる。

  CAL_ROOT_DIR (任意)
      LS 解析結果をまとめて置くルートディレクトリ。
      省略時は、event_uf から見て "../../.." (天体名ディレクトリ) が使われる想定。

  EVE_ARGS / CLUSTERCUT_ARGS (任意; フルパイプライン時のみ有効)
      resolve_ana_pixel_mklc_branch.py に渡すオプション。
      ここで与えなければ、環境変数 EVE_ARGS / CLUSTERCUT_ARGS、
      それも無ければスクリプト内のデフォルト値が使われる。

モード:
  --summary-only
      既に作成済みの LSCHECK ディレクトリの成果物を使って、
      summary_*.html だけを再生成するモード。
      クラスタ検出や mklc_branch の再実行は行わない。

環境変数:
  DEBUG_LSCHECK=1
      内部のデバッグ出力を有効にする。

  EVE_ARGS, CLUSTERCUT_ARGS
      mklc_branch 実行用のオプションを上書き指定できる。
      例:
        export EVE_ARGS="-t 2048 -odir output_eve -g -rmax 0.5 -yscaleing log -u --exclude-pixels 12"

必要なコマンド:
  resolve_ftools_cluster_pileline.sh
  resolve_util_screen_for_lscluster.sh
  resolve_ana_pixel_mklc_branch.py
  ftlist  (NAXIS2 や FITS ヘッダ参照に使用)

期待されるディレクトリ構造:
  (OBS ID)/resolve/event_uf/ で本スクリプトを実行することを想定。
  event_cl/ は (OBS ID)/resolve/event_cl/ に存在している必要がある。

EOF
}

# ========================================
# ヘルパー関数
# ========================================

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command '${cmd}' not found in PATH." >&2
        exit 1
    fi
}

# Get number of events from NAXIS2.
# If the second argument is 'no_cal', count only events with PIXEL!=12.
# If ftlist is not available, returns -1.
get_naxis2() {
    local file="$1"
    local mode="${2:-}"   # "" (default) or "no_cal"

    debug_echo "get_naxis2: start for file=${file}, mode=${mode}"

    if ! command -v ftlist >/dev/null 2>&1; then
        debug_echo "get_naxis2: ftlist not found in PATH"
        echo "-1"
        return
    fi
    if [[ ! -f "$file" ]]; then
        debug_echo "get_naxis2: file not found: ${file}"
        echo "-1"
        return
    fi

    local target="${file}+1"
    if [[ "$mode" == "no_cal" ]]; then
        # Exclude calibration pixel (PIXEL==12)
        target="${file}+1[PIXEL!=12]"
    fi

    debug_echo "get_naxis2: running 'ftlist ${target} K'"
    local naxis2
    naxis2=$(ftlist "${target}" K 2>/dev/null | awk '/NAXIS2/ {print $3; exit}' || true)

    if [[ -z "$naxis2" ]]; then
        debug_echo "get_naxis2: NAXIS2 not found in header"
        echo "-1"
    else
        debug_echo "get_naxis2: NAXIS2=${naxis2}"
        echo "$naxis2"
    fi
}

# FITS ヘッダから keyword を取得
get_header_keyword() {
    local file="$1"
    local key="$2"
    debug_echo "get_header_keyword: start for file=${file}, key=${key}"

    if ! command -v ftlist >/dev/null 2>&1; then
        debug_echo "get_header_keyword: ftlist not found in PATH"
        echo ""
        return
    fi
    if [[ ! -f "$file" ]]; then
        debug_echo "get_header_keyword: file not found: ${file}"
        echo ""
        return
    fi

    debug_echo "get_header_keyword: running 'ftlist ${file}+1 K | grep ^${key} ='"
    local line rhs
    line=$(ftlist "${file}+1" K 2>/dev/null | grep -E "^${key}[ ]*=" | head -n1 || true)
    debug_echo "get_header_keyword: raw line='${line}'"
    if [[ -z "$line" ]]; then
        debug_echo "get_header_keyword: keyword ${key} not found"
        echo ""
        return
    fi

    rhs="${line#*=}"
    rhs="${rhs%%/*}"
    rhs="$(echo "$rhs" | sed "s/^ *//;s/ *$//")"
    rhs="$(echo "$rhs" | sed "s/^'//;s/'$//")"

    debug_echo "get_header_keyword: parsed value='${rhs}'"
    echo "$rhs"
}

# ========================================
# Summary HTML generator
# ========================================
# The following global variables are expected to be set beforehand:
#   UF_EVT_BASENAME, CL_EVT_BASENAME, UF_NO_SUFFIX
#   LSCHECK_DIR, ASSESS_DIR, MKLC_DIR, CAL_ROOT_DIR
#   BRANCH_PREFIX, EVE_EVT_BASENAME, CLUSTERCUT_EVT_BASENAME
#   EVE_ARGS_USED, CLUSTERCUT_ARGS_USED
#   UF_EVT_ABS, MKRATE_DIR, EVE_EVT_STDCUT_BASENAME
# ========================================
generate_summary_html() {
    echo "[Summary] Generating HTML summary report ..."
    debug_echo "Summary: EVE_EVT=${ASSESS_DIR}/${EVE_EVT_BASENAME}"
    debug_echo "Summary: CLUSTERCUT_EVT=${ASSESS_DIR}/${CLUSTERCUT_EVT_BASENAME}"

    if [[ ! -f "${ASSESS_DIR}/${EVE_EVT_BASENAME}" ]]; then
        echo "Error in summary: ${ASSESS_DIR}/${EVE_EVT_BASENAME} not found." >&2
        exit 1
    fi
    if [[ ! -f "${ASSESS_DIR}/${CLUSTERCUT_EVT_BASENAME}" ]]; then
        echo "Error in summary: ${ASSESS_DIR}/${CLUSTERCUT_EVT_BASENAME} not found." >&2
        exit 1
    fi

    local N_ALL N_CL HAVE_STATS CLUST_FRAC PCT PCT_INT STATE_TEXT
    N_ALL=$(get_naxis2 "${ASSESS_DIR}/${EVE_EVT_BASENAME}" no_cal)
    N_CL=$(get_naxis2 "${ASSESS_DIR}/${CLUSTERCUT_EVT_BASENAME}" no_cal)
    debug_echo "Summary: N_ALL=${N_ALL}, N_CL=${N_CL}"

    HAVE_STATS=0
    CLUST_FRAC=""
    PCT=""
    STATE_TEXT=""

    if [[ "$N_ALL" =~ ^[0-9]+$ && "$N_CL" =~ ^[0-9]+$ && "$N_ALL" -gt 0 ]]; then
        HAVE_STATS=1
        CLUST_FRAC=$(awk 'BEGIN {printf "%.6f\n", 1 - '"$N_CL"'/'"$N_ALL"'}')
        PCT=$(awk 'BEGIN {printf "%.1f\n", '"$CLUST_FRAC"'*100}')
        PCT_INT=${PCT%.*}
        debug_echo "Summary: CLUST_FRAC=${CLUST_FRAC}, PCT=${PCT}, PCT_INT=${PCT_INT}"

        if [[ "$PCT_INT" -lt 5 ]]; then
            STATE_TEXT="Only about ${PCT}% of all events are rejected as LS-cluster candidates. The impact of LS-cluster pseudo events appears small, and the observation can be regarded as a \"quiet\" state."
        elif [[ "$PCT_INT" -lt 15 ]]; then
            STATE_TEXT="About ${PCT}% of all events are removed by the LS-cluster selection, indicating a moderate level of cluster activity. Spectral and timing analyses should take the contribution from LS clusters into account."
        else
            STATE_TEXT="Approximately ${PCT}% of all events are removed by the LS-cluster selection, which is a large fraction. Pseudo events associated with LS clusters may strongly affect this observation, and systematic uncertainties related to LS clusters should be carefully assessed in the scientific analysis."
        fi
    fi

    debug_echo "Summary: getting header keywords from ${UF_EVT_ABS}"

    local OBS_ID_HDR OBJECT_HDR DATEOBS_HDR PIXID_HDR PIXID_FROM_NAME
    OBS_ID_HDR="$(get_header_keyword "$UF_EVT_ABS" "OBS_ID")"
    OBJECT_HDR="$(get_header_keyword "$UF_EVT_ABS" "OBJECT")"
    DATEOBS_HDR="$(get_header_keyword "$UF_EVT_ABS" "DATE-OBS")"
    PROCVER_HDR="$(get_header_keyword "$UF_EVT_ABS" "PROCVER")"

    debug_echo "Summary: OBS_ID=${OBS_ID_HDR}, OBJECT=${OBJECT_HDR}, DATE-OBS=${DATEOBS_HDR}, PROCVER=${PROCVER_HDR}"

    PIXID_FROM_NAME="$(echo "$UF_NO_SUFFIX" | sed -n 's/.*px\([0-9][0-9]*\).*/\1/p')"
    debug_echo "Summary: PIXID_FROM_NAME=${PIXID_FROM_NAME}"

    # ---- Relative PNG paths (from the HTML file) ----
    local SMALL_PIXELMAP_REL SMALL_OVERLAY_REL LARGE_PIXELMAP_REL LARGE_OVERLAY_REL
    local DIAG_SUMMARY_REL EVE_BIAS_REL EVE_DIAG_REL CL_BIAS_REL CL_DIAG_REL
    local EVE_EVT_STDCUT_SUM_PNG_REL EVE_EVT_STDCUT_SUM_PIXMAPS_PNG_REL
    local EVE_EVT_SUM_PNG_REL EVE_EVT_SUM_PIXMAPS_PNG_REL

    SMALL_PIXELMAP_REL="fig_cluster/small_cluster_size_hist_all_pixels_pixelmap.png"
    SMALL_OVERLAY_REL="fig_cluster/small_cluster_size_hist_overlay_all_pixels.png"
    LARGE_PIXELMAP_REL="fig_cluster/large_cluster_size_hist_all_pixels_pixelmap.png"
    LARGE_OVERLAY_REL="fig_cluster/large_cluster_size_hist_overlay_all_pixels.png"

    DIAG_SUMMARY_REL="diagnostic_plots/summary_discarded_per_pixel_by_type.png"

    EVE_BIAS_REL="assess_bratios/mklc_branch/output_eve/mklc_lp_ls_bias_profilelik.png"
    EVE_DIAG_REL="assess_bratios/mklc_branch/output_eve/mklc_lp_ls_diag_basic_${BRANCH_PREFIX}.png"

    CL_BIAS_REL="assess_bratios/mklc_branch/output_clustercut/mklc_lp_ls_bias_profilelik.png"
    CL_DIAG_REL="assess_bratios/mklc_branch/output_clustercut/mklc_lp_ls_diag_basic_${BRANCH_PREFIX}_clustercut.png"

    EVE_EVT_SUM_PIXMAPS_PNG_REL="assess_bratios/calc_lscluster_rates/${EVE_EVT_SUM_PIXMAPS_PNG}"
    EVE_EVT_SUM_PNG_REL="assess_bratios/calc_lscluster_rates/${EVE_EVT_SUM_PNG}"
    EVE_EVT_STDCUT_SUM_PIXMAPS_PNG_REL="assess_bratios/calc_lscluster_rates/${EVE_EVT_STDCUT_SUM_PIXMAPS_PNG}"
    EVE_EVT_STDCUT_SUM_PNG_REL="assess_bratios/calc_lscluster_rates/${EVE_EVT_STDCUT_SUM_PNG}"

    debug_echo "Summary PNGs:"
    debug_echo "  small pixelmap : ${SMALL_PIXELMAP_REL}"
    debug_echo "  small overlay  : ${SMALL_OVERLAY_REL}"
    debug_echo "  large pixelmap : ${LARGE_PIXELMAP_REL}"
    debug_echo "  large overlay  : ${LARGE_OVERLAY_REL}"
    debug_echo "  diag summary   : ${DIAG_SUMMARY_REL}"
    debug_echo "  eve bias       : ${EVE_BIAS_REL}"
    debug_echo "  eve diag       : ${EVE_DIAG_REL}"
    debug_echo "  cl bias        : ${CL_BIAS_REL}"
    debug_echo "  cl diag        : ${CL_DIAG_REL}"
    debug_echo "  sum pmap       : ${EVE_EVT_SUM_PIXMAPS_PNG_REL}"
    debug_echo "  sum png        : ${EVE_EVT_SUM_PNG_REL}"
    debug_echo "  sum pmap std   : ${EVE_EVT_STDCUT_SUM_PIXMAPS_PNG_REL}"
    debug_echo "  sum png std    : ${EVE_EVT_STDCUT_SUM_PNG_REL}"

    local SUMMARY_HTML
    SUMMARY_HTML="${LSCHECK_DIR}/summary_${UF_NO_SUFFIX}.html"
    debug_echo "Summary: writing HTML to ${SUMMARY_HTML}"

    # ---- HTML part 1 (header and basic statistics) ----
    {
        echo '<!DOCTYPE html>'
        echo '<html lang="en">'
        echo '<head>'
        echo '  <meta charset="UTF-8">'
        echo "  <title>LS cluster summary for ${UF_NO_SUFFIX}</title>"
        echo '</head>'
        echo '<body>'
        echo "  <h1>LS cluster summary: ${UF_NO_SUFFIX}</h1>"

        echo '  <h2>Observation metadata (from FITS header)</h2>'
        echo '  <ul>'
        echo "    <li>OBS_ID : ${OBS_ID_HDR}</li>"
        echo "    <li>OBJECT : ${OBJECT_HDR}</li>"
        echo "    <li>DATE-OBS : ${DATEOBS_HDR}</li>"
        echo "    <li>PROCVER (header) : ${PROCVER_HDR}</li>"
        echo "    <li>PIXID (from filename) : ${PIXID_FROM_NAME}</li>"
        echo '  </ul>'

        echo '  <h2>Pipeline overview</h2>'
        echo '  <ul>'
        echo "    <li>UF event file: ${UF_EVT_BASENAME}</li>"
        echo "    <li>CL event file: ${CL_EVT_BASENAME}</li>"
        echo "    <li>Calibration root directory (CAL_ROOT_DIR): ${CAL_ROOT_DIR}</li>"
        echo "    <li>lscheck directory: ${LSCHECK_DIR}</li>"
        echo "    <li>assess_bratios directory: ${ASSESS_DIR}</li>"
        echo "    <li>mklc_branch directory: ${MKLC_DIR}</li>"
        echo '  </ul>'

        echo '  <h2>mklc_branch settings</h2>'
        echo '  <p><strong>eve.list:</strong> resolve_ana_pixel_mklc_branch.py eve.list '"$EVE_ARGS_USED"'</p>'
        echo '  <p><strong>clustercut.list:</strong> resolve_ana_pixel_mklc_branch.py clustercut.list '"$CLUSTERCUT_ARGS_USED"'</p>'
    } > "$SUMMARY_HTML"

    # Statistics section
    if [[ "$HAVE_STATS" -eq 1 ]]; then
        {
            echo '  <h2>LS-cluster statistics excluding cal pixel</h2>'
            echo "  <p>Total number of events in the LS-cluster candidate file (N_all): ${N_ALL}</p>"
            echo "  <p>Number of events after the LS-cluster cut (N_clustercut): ${N_CL}</p>"
            echo "  <p>Fraction of events rejected as LS clusters: ${PCT}%</p>"
            echo "  <p><strong>Interpretation:</strong> ${STATE_TEXT}</p>"
        } >> "$SUMMARY_HTML"
    else
        {
            echo '  <h2>LS-cluster statistics</h2>'
            echo '  <p>Automatic event counting failed because ftlist was not found in PATH or NAXIS2 could not be obtained from the FITS header.</p>'
        } >> "$SUMMARY_HTML"
    fi

    # ---- PNG embedding section ----
    {
        echo '  <h2>Cluster-detection diagnostics</h2>'

        echo '  <h3>Small clusters</h3>'
        if [[ -f "${LSCHECK_DIR}/${SMALL_PIXELMAP_REL}" ]]; then
            echo "  <p><img src=\"${SMALL_PIXELMAP_REL}\" alt=\"Pixel map of small LS-cluster counts for all pixels\" style=\"max-width:48%;height:auto;margin:4px;\"></p>"
        else
            echo "  <p>(Pixel map for small clusters not found: ${SMALL_PIXELMAP_REL})</p>"
        fi
        if [[ -f "${LSCHECK_DIR}/${SMALL_OVERLAY_REL}" ]]; then
            echo "  <p><img src=\"${SMALL_OVERLAY_REL}\" alt=\"Overlay of size distributions for small LS clusters over all pixels\" style=\"max-width:48%;height:auto;margin:4px;\"></p>"
        else
            echo "  <p>(Overlay histogram for small clusters not found: ${SMALL_OVERLAY_REL})</p>"
        fi

        echo '  <h3>Large clusters</h3>'
        if [[ -f "${LSCHECK_DIR}/${LARGE_PIXELMAP_REL}" ]]; then
            echo "  <p><img src=\"${LARGE_PIXELMAP_REL}\" alt=\"Pixel map of large LS-cluster counts for all pixels\" style=\"max-width:48%;height:auto;margin:4px;\"></p>"
        else
            echo "  <p>(Pixel map for large clusters not found: ${LARGE_PIXELMAP_REL})</p>"
        fi
        if [[ -f "${LSCHECK_DIR}/${LARGE_OVERLAY_REL}" ]]; then
            echo "  <p><img src=\"${LARGE_OVERLAY_REL}\" alt=\"Overlay of size distributions for large LS clusters over all pixels\" style=\"max-width:48%;height:auto;margin:4px;\"></p>"
        else
            echo "  <p>(Overlay histogram for large clusters not found: ${LARGE_OVERLAY_REL})</p>"
        fi

        echo '  <h2>Diagnostic summary plot</h2>'
        if [[ -f "${LSCHECK_DIR}/${DIAG_SUMMARY_REL}" ]]; then
            echo "  <p><img src=\"${DIAG_SUMMARY_REL}\" alt=\"Summary of discarded events per pixel, broken down by event type\" style=\"max-width:60%;height:auto;margin:4px;\"></p>"
        else
            echo "  <p>(Diagnostic summary PNG not found: ${DIAG_SUMMARY_REL})</p>"
        fi

        echo '  <h2>assess_bratios: output_eve (before LS-cluster cut)</h2>'
        if [[ -f "${LSCHECK_DIR}/${EVE_BIAS_REL}" ]]; then
            echo "  <p><img src=\"${EVE_BIAS_REL}\" alt=\"Profile-likelihood scan of LP/LS branch ratios (eve sample)\" style=\"max-width:48%;height:auto;margin:4px;\"></p>"
        else
            echo "  <p>(Profile-likelihood plot for eve sample not found: ${EVE_BIAS_REL})</p>"
        fi
        if [[ -f "${LSCHECK_DIR}/${EVE_DIAG_REL}" ]]; then
            echo "  <p><img src=\"${EVE_DIAG_REL}\" alt=\"Basic diagnostic plots for LP/LS branch ratios (eve sample)\" style=\"max-width:48%;height:auto;margin:4px;\"></p>"
        else
            echo "  <p>(Diagnostic plot for eve sample not found: ${EVE_DIAG_REL})</p>"
        fi

        echo '  <h2>assess_bratios: output_clustercut (after LS-cluster cut)</h2>'
        if [[ -f "${LSCHECK_DIR}/${CL_BIAS_REL}" ]]; then
            echo "  <p><img src=\"${CL_BIAS_REL}\" alt=\"Profile-likelihood scan of LP/LS branch ratios (clustercut sample)\" style=\"max-width:48%;height:auto;margin:4px;\"></p>"
        else
            echo "  <p>(Profile-likelihood plot for clustercut sample not found: ${CL_BIAS_REL})</p>"
        fi
        if [[ -f "${LSCHECK_DIR}/${CL_DIAG_REL}" ]]; then
            echo "  <p><img src=\"${CL_DIAG_REL}\" alt=\"Basic diagnostic plots for LP/LS branch ratios (clustercut sample)\" style=\"max-width:48%;height:auto;margin:4px;\"></p>"
        else
            echo "  <p>(Diagnostic plot for clustercut sample not found: ${CL_DIAG_REL})</p>"
        fi

        echo '  <h2>assess_bratios: calc_lscluster_rates pixel maps (all events)</h2>'
        if [[ -f "${LSCHECK_DIR}/${EVE_EVT_SUM_PIXMAPS_PNG_REL}" ]]; then
            echo "  <p><img src=\"${EVE_EVT_SUM_PIXMAPS_PNG_REL}\" alt=\"Per-pixel count rates and LS-cluster fractions (all events, pixel maps)\" style=\"max-width:48%;height:auto;margin:4px;\"></p>"
        else
            echo "  <p>(calc_lscluster_rates pixel maps (all events) not found: ${EVE_EVT_SUM_PIXMAPS_PNG_REL})</p>"
        fi

        echo '  <h2>assess_bratios: calc_lscluster_rates summary (all events)</h2>'
        if [[ -f "${LSCHECK_DIR}/${EVE_EVT_SUM_PNG_REL}" ]]; then
            echo "  <p><img src=\"${EVE_EVT_SUM_PNG_REL}\" alt=\"Summary of per-pixel count rates and LS-cluster fractions (all events)\" style=\"max-width:48%;height:auto;margin:4px;\"></p>"
        else
            echo "  <p>(calc_lscluster_rates summary (all events) not found: ${EVE_EVT_SUM_PNG_REL})</p>"
        fi

        echo '  <h2>assess_bratios: calc_lscluster_rates pixel maps (stdcut)</h2>'
        if [[ -f "${LSCHECK_DIR}/${EVE_EVT_STDCUT_SUM_PIXMAPS_PNG_REL}" ]]; then
            echo "  <p><img src=\"${EVE_EVT_STDCUT_SUM_PIXMAPS_PNG_REL}\" alt=\"Per-pixel count rates and LS-cluster fractions (standard event selection, pixel maps)\" style=\"max-width:48%;height:auto;margin:4px;\"></p>"
        else
            echo "  <p>(calc_lscluster_rates pixel maps (stdcut) not found: ${EVE_EVT_STDCUT_SUM_PIXMAPS_PNG_REL})</p>"
        fi

        echo '  <h2>assess_bratios: calc_lscluster_rates summary (stdcut)</h2>'
        if [[ -f "${LSCHECK_DIR}/${EVE_EVT_STDCUT_SUM_PNG_REL}" ]]; then
            echo "  <p><img src=\"${EVE_EVT_STDCUT_SUM_PNG_REL}\" alt=\"Summary of per-pixel count rates and LS-cluster fractions (standard event selection)\" style=\"max-width:48%;height:auto;margin:4px;\"></p>"
        else
            echo "  <p>(calc_lscluster_rates summary (stdcut) not found: ${EVE_EVT_STDCUT_SUM_PNG_REL})</p>"
        fi

    } >> "$SUMMARY_HTML"

    # ---- Textual interpretation ----
    {
        echo '  <h2>Quick interpretation of this observation</h2>'
        echo '  <p>'
        echo '    The key indicator here is the fraction of events removed by the LS-cluster selection,'
        echo '    computed as the ratio between the number of events in the LS-cluster candidate file'
        echo '    (clgti event file) and the number of events that remain after applying the LS-cluster cut.'
        echo '  </p>'

        echo '  <h2>How to read these results and possible next steps</h2>'
        echo '  <p>'
        echo '    For more detailed diagnostics, please refer to the plots and text summaries'
        echo '    generated under the <code>output_eve</code> and <code>output_clustercut</code> directories.'
        echo '  </p>'
        echo '</body>'
        echo '</html>'
    } >> "$SUMMARY_HTML"

    echo
    echo "Summary HTML report generated at:"
    echo "  ${SUMMARY_HTML}"
}


# ========================================
# メイン処理
# ========================================

SUMMARY_ONLY=0

# ここで -h / --help を処理
case "${1:-}" in
    -h|--help)
        print_help
        exit 0
        ;;
    --summary-only)
        SUMMARY_ONLY=1
        shift
        ;;
esac

if [[ $# -lt 1 ]]; then
    echo "Usage:" >&2
    echo "  Full pipeline : $0 <uf_event_file> [lscheck_dir_name] [CAL_ROOT_DIR] [EVE_ARGS] [CLUSTERCUT_ARGS]" >&2
    echo "  Summary only  : $0 --summary-only <uf_event_file> [lscheck_dir_name] [CAL_ROOT_DIR]" >&2
    exit 1
fi

require_cmd resolve_ftools_cluster_pileline.sh
require_cmd resolve_util_screen_for_lscluster.sh
require_cmd resolve_ana_pixel_mklc_branch.py

UF_EVT_BASENAME="$1"
if [[ ! -f "$UF_EVT_BASENAME" ]]; then
    echo "Error: uf event file '$UF_EVT_BASENAME' not found in current directory." >&2
    exit 1
fi

if [[ $# -ge 2 && -n "${2:-}" ]]; then
    LSCHECK_DIR_NAME="$2"
else
    LSCHECK_DIR_NAME="lscheck_$(date +%Y%m%d)"
fi

CURRENT_DIR="$(pwd)"
RESOLVE_DIR="$(cd .. && pwd)"
OBSID_DIR="$(cd ../.. && pwd)"
DEFAULT_CAL_ROOT_DIR="$(cd ../../.. && pwd)"

if [[ $# -ge 3 && -n "${3:-}" ]]; then
    CAL_ROOT_DIR="$(cd "$3" && pwd)"
else
    CAL_ROOT_DIR="$DEFAULT_CAL_ROOT_DIR"
fi

EVENT_CL_DIR="${RESOLVE_DIR}/event_cl"
EVENT_UF_DIR="${RESOLVE_DIR}/event_uf"

if [[ ! -d "$EVENT_CL_DIR" ]]; then
    echo "Error: event_cl directory not found at '$EVENT_CL_DIR'." >&2
    exit 1
fi

UF_NO_SUFFIX="${UF_EVT_BASENAME%_uf.evt}"
CL_EVT_BASENAME="${UF_NO_SUFFIX}_cl.evt"
CL_EVT_PATH="${EVENT_CL_DIR}/${CL_EVT_BASENAME}"

if [[ ! -f "$CL_EVT_PATH" ]]; then
    echo "Error: corresponding cl event file '$CL_EVT_PATH' not found." >&2
    exit 1
fi

UF_EVT_ABS="${EVENT_UF_DIR}/${UF_EVT_BASENAME}"

# EVE / CLUSTERCUT の引数 (full モードのときのみ意味があるが、summary でも表示用に使う)
EVE_ARGS_OVERRIDE="${4:-}"
CLUSTERCUT_ARGS_OVERRIDE="${5:-}"

if [[ -n "$EVE_ARGS_OVERRIDE" ]]; then
    EVE_ARGS_USED="$EVE_ARGS_OVERRIDE"
elif [[ -n "${EVE_ARGS:-}" ]]; then
    EVE_ARGS_USED="$EVE_ARGS"
else
    EVE_ARGS_USED="-t 2048 -odir output_eve -g -rmax 0.5 -yscaleing log -u --exclude-pixels 12"
fi

if [[ -n "$CLUSTERCUT_ARGS_OVERRIDE" ]]; then
    CLUSTERCUT_ARGS_USED="$CLUSTERCUT_ARGS_OVERRIDE"
elif [[ -n "${CLUSTERCUT_ARGS:-}" ]]; then
    CLUSTERCUT_ARGS_USED="$CLUSTERCUT_ARGS"
else
    CLUSTERCUT_ARGS_USED="-t 2048 -odir output_clustercut -g -rmax 0.5 -yscaleing log -u --exclude-pixels 12"
fi

echo "Mode             : $([[ $SUMMARY_ONLY -eq 1 ]] && echo 'SUMMARY ONLY' || echo 'FULL PIPELINE')"
echo "Current directory: $CURRENT_DIR"
echo "Resolve directory: $RESOLVE_DIR"
echo "OBSID directory  : $OBSID_DIR"
echo "CAL_ROOT_DIR     : $CAL_ROOT_DIR"
echo "Using UF event   : $UF_EVT_BASENAME"
echo "Using CL event   : $CL_EVT_BASENAME"
echo "lscheck dir name : $LSCHECK_DIR_NAME"
echo "EVE_ARGS used        : $EVE_ARGS_USED"
echo "CLUSTERCUT_ARGS used : $CLUSTERCUT_ARGS_USED"
echo

LSCHECK_DIR="${CAL_ROOT_DIR}/${LSCHECK_DIR_NAME}"
ASSESS_DIR="${LSCHECK_DIR}/assess_bratios"
MKLC_DIR="${ASSESS_DIR}/mklc_branch"
MKRATE_DIR="${ASSESS_DIR}/calc_lscluster_rates"

UF_NO_EXT="${UF_EVT_BASENAME%.evt}"
SMALL_LARGE_BASE="small_large_${UF_NO_EXT}_noBL_prevnext_cutclgti.evt"
BRANCH_PREFIX="small_large_${UF_NO_EXT}_noBL_prevnext_cutclgti_clgti"
EVE_EVT_BASENAME="${BRANCH_PREFIX}.evt"
CLUSTERCUT_EVT_BASENAME="${BRANCH_PREFIX}_clustercut.evt"
EVE_EVT_STDCUT_BASENAME="${BRANCH_PREFIX}_stdcut.evt"

EVE_EVT_SUM_PIXMAPS_PNG="${BRANCH_PREFIX}_lsrates_summary_pixmaps.png"
EVE_EVT_SUM_PNG="${BRANCH_PREFIX}_lsrates_summary.png"
EVE_EVT_STDCUT_SUM_PIXMAPS_PNG="${BRANCH_PREFIX}_stdcut_lsrates_summary_pixmaps.png"
EVE_EVT_STDCUT_SUM_PNG="${BRANCH_PREFIX}_stdcut_lsrates_summary.png"

# -------------------------
# FULL PIPELINE モード
# -------------------------
if [[ "$SUMMARY_ONLY" -eq 0 ]]; then
    echo "[1/5] Creating lscheck directory: $LSCHECK_DIR"
    mkdir -p "$LSCHECK_DIR"

    CL_EVT_ABS="${CL_EVT_PATH}"

    echo "[2/5] Creating symbolic links to UF/CL event files in $LSCHECK_DIR"
    cd "$LSCHECK_DIR"

    if [[ ! -e "$UF_EVT_BASENAME" ]]; then
        ln -s "$UF_EVT_ABS" .
    fi
    if [[ ! -e "$CL_EVT_BASENAME" ]]; then
        ln -s "$CL_EVT_ABS" .
    fi

    echo "Files in lscheck:"
    ls
    echo

    echo "[3/5] Running resolve_ftools_cluster_pileline.sh ..."
#    resolve_ftools_cluster_pileline.sh "$UF_EVT_BASENAME"

    echo "After clustering pipeline:"
    ls
    echo

    SMALL_LARGE_PATH="${LSCHECK_DIR}/${SMALL_LARGE_BASE}"
    if [[ ! -f "$SMALL_LARGE_PATH" ]]; then
        echo "Error: expected small_large file '$SMALL_LARGE_PATH' not found after clustering pipeline." >&2
        exit 1
    fi

    echo "[4/5] Creating assess_bratios directory and running resolve_util_screen_for_lscluster.sh ..."
    mkdir -p "$ASSESS_DIR"
    cd "$ASSESS_DIR"

    if [[ ! -e "$SMALL_LARGE_BASE" ]]; then
        ln -s "$SMALL_LARGE_PATH" .
    fi
    if [[ ! -e "$CL_EVT_BASENAME" ]]; then
        ln -s "${LSCHECK_DIR}/${CL_EVT_BASENAME}" .
    fi

#    resolve_util_screen_for_lscluster.sh "$SMALL_LARGE_BASE" "$CL_EVT_BASENAME"

    echo "After resolve_util_screen_for_lscluster.sh:"
    ls
    echo

    if [[ ! -f "$EVE_EVT_BASENAME" ]]; then
        echo "Error: '$EVE_EVT_BASENAME' not found in assess_bratios." >&2
        exit 1
    fi
    if [[ ! -f "$CLUSTERCUT_EVT_BASENAME" ]]; then
        echo "Error: '$CLUSTERCUT_EVT_BASENAME' not found in assess_bratios." >&2
        exit 1
    fi

    echo "[5/5] Creating mklc_branch directory and preparing mklc_branch analysis ..."
    mkdir -p "$MKLC_DIR"
    cd "$MKLC_DIR"

    if [[ ! -e "$EVE_EVT_BASENAME" ]]; then
        ln -s "${ASSESS_DIR}/${EVE_EVT_BASENAME}" .
    fi
    if [[ ! -e "$CLUSTERCUT_EVT_BASENAME" ]]; then
        ln -s "${ASSESS_DIR}/${CLUSTERCUT_EVT_BASENAME}" .
    fi

    echo "$EVE_EVT_BASENAME"        > eve.list
    echo "$CLUSTERCUT_EVT_BASENAME" > clustercut.list

    cat > run_mklc_branch.sh << 'EOF'
#!/bin/sh
: "${EVE_ARGS:="-t 2048 -odir output_eve -g -rmax 0.5 -yscaleing log -u --exclude-pixels 12"}"
: "${CLUSTERCUT_ARGS:="-t 2048 -odir output_clustercut -g -rmax 0.5 -yscaleing log -u --exclude-pixels 12"}"

echo "[run_mklc_branch] EVE_ARGS=${EVE_ARGS}"
echo "[run_mklc_branch] CLUSTERCUT_ARGS=${CLUSTERCUT_ARGS}"

resolve_ana_pixel_mklc_branch.py eve.list        $EVE_ARGS
resolve_ana_pixel_mklc_branch.py clustercut.list $CLUSTERCUT_ARGS
EOF

    chmod +x run_mklc_branch.sh

    echo "mklc_branch directory contents (before running mklc):"
    ls
    echo

    echo "Running run_mklc_branch.sh ..."
    if [[ -n "$EVE_ARGS_OVERRIDE" && -n "$CLUSTERCUT_ARGS_OVERRIDE" ]]; then
        EVE_ARGS="$EVE_ARGS_OVERRIDE" CLUSTERCUT_ARGS="$CLUSTERCUT_ARGS_OVERRIDE" ./run_mklc_branch.sh
    elif [[ -n "$EVE_ARGS_OVERRIDE" ]]; then
        EVE_ARGS="$EVE_ARGS_OVERRIDE" ./run_mklc_branch.sh
    elif [[ -n "$CLUSTERCUT_ARGS_OVERRIDE" ]]; then
        CLUSTERCUT_ARGS="$CLUSTERCUT_ARGS_OVERRIDE" ./run_mklc_branch.sh
    else
#        ./run_mklc_branch.sh
        echo
    fi

    echo
    echo "mklc_branch directory contents (after running mklc):"
    ls
    echo

    ###### added 
    cd ..

    mkdir -p "$MKRATE_DIR"
    cd "$MKRATE_DIR"

    if [[ ! -e "$EVE_EVT_BASENAME" ]]; then
        ln -s "${ASSESS_DIR}/${EVE_EVT_BASENAME}" .
    fi
    if [[ ! -e "$EVE_EVT_STDCUT_BASENAME" ]]; then
        ln -s "${ASSESS_DIR}/${EVE_EVT_STDCUT_BASENAME}" .
    fi

    echo "Running resolve_ftools_cluster_calc_lscluster_rates.py ..."
    resolve_ftools_cluster_calc_lscluster_rates.py "$EVE_EVT_BASENAME"
    resolve_ftools_cluster_calc_lscluster_rates.py "$EVE_EVT_STDCUT_BASENAME"
    ls
    echo
    ########

else
    # -------------------------
    # SUMMARY ONLY モード
    # -------------------------
    echo "[Summary-only mode] Skipping pipeline steps, using existing products."
    echo "  LSCHECK_DIR: ${LSCHECK_DIR}"
    echo "  ASSESS_DIR : ${ASSESS_DIR}"
    echo "  MKLC_DIR   : ${MKLC_DIR}"
    echo

    # 最低限の存在チェック
    if [[ ! -d "$LSCHECK_DIR" ]]; then
        echo "Error: LSCHECK_DIR '${LSCHECK_DIR}' does not exist. Please specify correct lscheck_dir_name and CAL_ROOT_DIR." >&2
        exit 1
    fi
    if [[ ! -d "$ASSESS_DIR" ]]; then
        echo "Error: ASSESS_DIR '${ASSESS_DIR}' does not exist. Pipeline might not have been run yet." >&2
        exit 1
    fi
    if [[ ! -d "$MKLC_DIR" ]]; then
        echo "Warning: MKLC_DIR '${MKLC_DIR}' does not exist. mklc_branch may not have been run, but summary will try using assess_bratios only." >&2
    fi
fi

# 最後に Summary HTML を生成
generate_summary_html

if [[ "$SUMMARY_ONLY" -eq 0 ]]; then
    echo
    echo "All pipeline steps completed successfully (including summary)."
else
    echo
    echo "Summary-only regeneration completed successfully."
fi
