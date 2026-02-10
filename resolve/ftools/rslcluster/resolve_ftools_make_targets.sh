#!/usr/bin/env bash
#
# make_targets.sh
#
# 目的:
#   CAL_ROOT_DIR 配下の
#     OBJECT/OBSID/resolve/event_uf/xa${OBSID}rsl_p0pxXXXX_uf.evt
#   を探索して
#     OBJECT_NAME   OBSID      FILTER
#   形式の targets.txt を自動生成する。
#
# 仕様:
#   - FILTER は p0pxXXXX の XXXX 部分
#   - ただし XXXX が 1000 / 3000 のものだけを残し、
#     0000, 5000 など天体観測ではないものは除外する。
#
# 使い方:
#   ./make_targets.sh                 # カレントディレクトリを CAL_ROOT_DIR とみなす
#   ./make_targets.sh /path/to/caldb  # CAL_ROOT_DIR を明示指定
#
#   出力ファイル: CAL_ROOT_DIR/targets.txt
#

set -euo pipefail

CAL_ROOT_DIR="${1:-$(pwd)}"
OUTPUT_FILE="${CAL_ROOT_DIR}/targets.txt"

echo "CAL_ROOT_DIR : ${CAL_ROOT_DIR}"
echo "OUTPUT_FILE  : ${OUTPUT_FILE}"
echo

if [[ ! -d "${CAL_ROOT_DIR}" ]]; then
    echo "Error: CAL_ROOT_DIR='${CAL_ROOT_DIR}' is not a directory." >&2
    exit 1
fi

# マッチしないグロブを空にする
shopt -s nullglob

TMP_FILE="$(mktemp)"
trap 'rm -f "${TMP_FILE}"' EXIT

# オブジェクトディレクトリ (= 天体名ディレクトリ)
for obj_dir in "${CAL_ROOT_DIR}"/*/; do
    [[ -d "$obj_dir" ]] || continue

    obj_name="$(basename "${obj_dir}")"

    # OBSID ディレクトリ
    for obsid_dir in "${obj_dir}"/*/; do
        [[ -d "$obsid_dir" ]] || continue

        obsid="$(basename "${obsid_dir}")"

        event_uf_dir="${obsid_dir}/resolve/event_uf"
        if [[ ! -d "${event_uf_dir}" ]]; then
            continue
        fi

        # 例: xa300013010rsl_p0px1000_uf.evt
        for evt in "${event_uf_dir}"/xa*rsl_p0px*_uf.evt; do
            [[ -f "$evt" ]] || continue

            fname="$(basename "$evt")"

            # ファイル名から OBSID 抜き出し
            obsid_from_name="$(echo "$fname" | sed -E 's/^xa([0-9]{9})rsl.*/\1/')" || obsid_from_name=""

            # ファイル名から FILTER (1000, 3000, 0000, 5000, ...) 抜き出し
            filter="$(echo "$fname" | sed -E 's/^xa[0-9]{9}rsl_p0px([0-9]+)_uf\.evt/\1/')" || filter=""

            if [[ -z "$obsid_from_name" ]] || [[ -z "$filter" ]]; then
                echo "[WARN] could not parse OBSID/FILTER from '$fname' (in $event_uf_dir)" >&2
                continue
            fi

            # ★ ここで 1000 / 3000 以外は捨てる ★
            if [[ "$filter" != "1000" && "$filter" != "3000" ]]; then
                # 0000, 5000 など天体観測ではないモードはスキップ
                continue
            fi

            printf "%-15s %-9s %s\n" "$obj_name" "$obsid_from_name" "$filter" >> "${TMP_FILE}"
        done
    done
done

# ソート + 重複排除してヘッダを付ける
{
    echo "# OBJECT_NAME   OBSID      FILTER"
    sort -u "${TMP_FILE}"
} > "${OUTPUT_FILE}"

echo "Done."
echo "Generated ${OUTPUT_FILE}"
