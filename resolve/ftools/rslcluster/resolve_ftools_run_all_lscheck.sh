#!/usr/bin/env bash
#
# run_all_lscheck.sh
#
# 使い方:
#   ./run_all_lscheck.sh targets.txt
#
# 環境変数:
#   CAL_ROOT_DIR       : 天体ディレクトリが並んでいるルート
#                        省略時は、このスクリプトのあるディレクトリをルートとみなす。
#
#   LSCHECK_ROOT_DIR   : lscheck_* ディレクトリをまとめて置くルート
#                        省略時は、「CAL_ROOT_DIR の 1 つ上のディレクトリ」配下に
#                        lscheck_results/ を作成して使用する。
#                        例: CAL_ROOT_DIR=/.../cal_database →
#                            LSCHECK_ROOT_DIR=/.../lscheck_results
#
#   SKIP_EXISTING=1    : 既に lscheck_* ディレクトリがあればスキップ（デフォルト）
#                        0 にすると強制的に再実行する。
#
#   MODE=full or summary
#      full    : フルパイプライン (デフォルト)
#      summary : --summary-only モードで summary_*.html のみ再生成
#

set -euo pipefail

LIST_FILE="${1:-}"
if [[ -z "$LIST_FILE" ]]; then
    echo "Usage: $0 <target_list_file>" >&2
    exit 1
fi
if [[ ! -f "$LIST_FILE" ]]; then
    echo "Error: target list file '$LIST_FILE' not found." >&2
    exit 1
fi

# スクリプトが置いてあるディレクトリ
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# CAL_ROOT_DIR（データのルート）
CAL_ROOT_DIR="${CAL_ROOT_DIR:-$SCRIPT_DIR}"

# ★ LSCHECK_ROOT_DIR（解析生成物のルート） ★
# デフォルトは CAL_ROOT_DIR の 1 つ上に lscheck_results/ を作る
if [[ -z "${LSCHECK_ROOT_DIR:-}" ]]; then
    PARENT_DIR="$(cd "${CAL_ROOT_DIR}/.." && pwd)"
    LSCHECK_ROOT_DIR="${PARENT_DIR}/lscheck_results"
fi

SKIP_EXISTING="${SKIP_EXISTING:-1}"
MODE="${MODE:-full}"

# ルートディレクトリの作成（無ければ）
mkdir -p "${LSCHECK_ROOT_DIR}"

echo "CAL_ROOT_DIR     : $CAL_ROOT_DIR"
echo "LSCHECK_ROOT_DIR : $LSCHECK_ROOT_DIR"
echo "TARGET LIST      : $LIST_FILE"
echo "SKIP_EXISTING    : $SKIP_EXISTING"
echo "MODE             : $MODE"
echo

while read -r OBJ OBSID FILTER; do
    # 空行・コメント行をスキップ
    if [[ -z "${OBJ:-}" ]] || [[ "${OBJ:0:1}" == "#" ]]; then
        continue
    fi

    echo "========================================"
    echo "OBJECT = $OBJ, OBSID = $OBSID, FILTER = $FILTER"
    echo "========================================"

    EVENT_UF_DIR="${CAL_ROOT_DIR}/${OBJ}/${OBSID}/resolve/event_uf"
    if [[ ! -d "$EVENT_UF_DIR" ]]; then
        echo "  [WARN] event_uf directory not found: $EVENT_UF_DIR"
        continue
    fi

    UF_EVT="xa${OBSID}rsl_p0px${FILTER}_uf.evt"
    if [[ ! -f "${EVENT_UF_DIR}/${UF_EVT}" ]]; then
        echo "  [WARN] UF event file not found: ${EVENT_UF_DIR}/${UF_EVT}"
        continue
    fi

    # lscheck_* ディレクトリ名（名前自体は今まで通り）
    LSCHECK_DIR_NAME="lscheck_${OBJ}_${OBSID}_px${FILTER}"

    # ★ 生成物を置く実際のパスは LSCHECK_ROOT_DIR の下にする ★
    LSCHECK_DIR_PATH="${LSCHECK_ROOT_DIR}/${LSCHECK_DIR_NAME}"

    if [[ "$SKIP_EXISTING" -eq 1 && -d "${LSCHECK_DIR_PATH}" ]]; then
        echo "  [INFO] ${LSCHECK_DIR_PATH} already exists -> skip (SKIP_EXISTING=1)"
        continue
    fi

    echo "  event_uf dir : $EVENT_UF_DIR"
    echo "  UF_EVT       : $UF_EVT"
    echo "  LSCHECK_DIR  : ${LSCHECK_DIR_PATH}"

    (
        cd "$EVENT_UF_DIR"

        if [[ "$MODE" == "summary" ]]; then
            echo "  -> run summary-only mode"
            # 第2引数: LSCHECK_DIR_NAME
            # 第3引数: 生成物ルート (LSCHECK_ROOT_DIR) を渡す
            run_resolve_ftools_cluster_pileline.sh --summary-only "$UF_EVT" "$LSCHECK_DIR_NAME" "$LSCHECK_ROOT_DIR"
        else
            echo "  -> run full pipeline"
            run_resolve_ftools_cluster_pileline.sh "$UF_EVT" "$LSCHECK_DIR_NAME" "$LSCHECK_ROOT_DIR"
        fi
    )
    echo
done < "$LIST_FILE"

echo "All requested targets processed."
