#!/usr/bin/env bash
#
# resolve_ftools_make_targets.sh (make_targets.sh)
#
# Purpose:
#   Scan CAL_ROOT_DIR for
#     OBJECT/OBSID/resolve/event_uf/xa${OBSID}rsl_p0pxXXXX_uf.evt
#   and corresponding
#     OBJECT/OBSID/resolve/event_cl/xa${OBSID}rsl_p0pxXXXX_cl.evt
#
#   For each (OBJECT, OBSID):
#     - Consider only filters 1000 and 3000
#     - Require that BOTH uf.evt and cl.evt exist
#     - If both 1000 and 3000 exist, choose the one with larger uf.evt size
#     - Output at most one line:
#         OBJECT_NAME   OBSID      FILTER
#
# Usage:
#   ./resolve_ftools_make_targets.sh
#   ./resolve_ftools_make_targets.sh /path/to/cal_database
#
# Output:
#   CAL_ROOT_DIR/targets.txt
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

# just in case, though we no longer use globs that need this
shopt -s nullglob

TMP_FILE="$(mktemp)"
trap 'rm -f "${TMP_FILE}"' EXIT

# helper: get file size in bytes (portable)
get_size() {
    # usage: get_size <path>
    wc -c < "$1" 2>/dev/null || echo 0
}

# loop over object directories
for obj_dir in "${CAL_ROOT_DIR}"/*/; do
    [[ -d "$obj_dir" ]] || continue

    obj_name="$(basename "${obj_dir}")"

    # loop over OBSID directories
    for obsid_dir in "${obj_dir}"/*/; do
        [[ -d "$obsid_dir" ]] || continue

        obsid="$(basename "${obsid_dir}")"

        event_uf_dir="${obsid_dir}/resolve/event_uf"
        event_cl_dir="${obsid_dir}/resolve/event_cl"

        if [[ ! -d "${event_uf_dir}" ]]; then
            # no event_uf directory -> skip this OBSID
            continue
        fi
        if [[ ! -d "${event_cl_dir}" ]]; then
            # no event_cl directory -> skip this OBSID
            echo "[WARN] ${obj_name} ${obsid}: event_cl directory not found, skip" >&2
            continue
        fi

        # candidates: p0px1000, p0px3000
        best_filter=""
        best_size=0

        for filter in 1000 3000; do
            uf_evt="${event_uf_dir}/xa${obsid}rsl_p0px${filter}_uf.evt"
            cl_evt="${event_cl_dir}/xa${obsid}rsl_p0px${filter}_cl.evt"

            if [[ ! -f "$uf_evt" ]]; then
                # uf not present for this filter
                continue
            fi
            if [[ ! -f "$cl_evt" ]]; then
                # cl not present for this filter
                echo "[WARN] ${obj_name} ${obsid} px${filter}: cl.evt not found, skip this filter" >&2
                continue
            fi

            size="$(get_size "$uf_evt")"
            if [[ "$size" -gt "$best_size" ]]; then
                best_size="$size"
                best_filter="$filter"
            fi
        done

        if [[ -z "$best_filter" ]]; then
            # no valid filter (no uf+cl pair)
            echo "[WARN] ${obj_name} ${obsid}: no valid (uf+cl) pair for px1000/px3000, skip" >&2
            continue
        fi

        # one line per OBJECT+OBSID (the best filter)
        printf "%-15s %-9s %s\n" "$obj_name" "$obsid" "$best_filter" >> "${TMP_FILE}"
    done
done

{
    echo "# OBJECT_NAME   OBSID      FILTER"
    sort -u "${TMP_FILE}"
} > "${OUTPUT_FILE}"

echo "Done."
echo "Generated ${OUTPUT_FILE}"
