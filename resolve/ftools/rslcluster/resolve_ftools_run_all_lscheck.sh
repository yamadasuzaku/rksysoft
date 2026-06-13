#!/usr/bin/env bash
#
# run_all_lscheck.sh
#
# Usage:
#   ./run_all_lscheck.sh [--interval_limit INTERVAL_LIMIT] targets.txt
#
# Input:
#   targets.txt
#
# Output:
#   targets_valid.txt    : entries with both uf.evt and cl.evt present
#   targets_missing.txt  : entries with missing uf.evt and/or cl.evt
#
# Environment variables:
#   CAL_ROOT_DIR   : root directory containing object directories
#                    default: directory where this script exists
#   SKIP_EXISTING  : 1 (default) skip if lscheck dir exists
#                    0 force re-run
#   MODE           : full | summary
#   INTERVAL_LIMIT : optional value passed as --interval_limit to the clustering pipeline
#

set -euo pipefail

INTERVAL_LIMIT="${INTERVAL_LIMIT:-}"

while [[ $# -gt 0 ]]; do
    case "${1:-}" in
        --interval_limit)
            if [[ $# -lt 2 ]]; then
                echo "Error: --interval_limit requires an integer value." >&2
                echo "Usage: $0 [--interval_limit INTERVAL_LIMIT] <target_list_file>" >&2
                exit 1
            fi
            INTERVAL_LIMIT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--interval_limit INTERVAL_LIMIT] <target_list_file>"
            echo "Environment: CAL_ROOT_DIR, SKIP_EXISTING, MODE=full|summary, INTERVAL_LIMIT"
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Error: unknown option '$1'." >&2
            echo "Usage: $0 [--interval_limit INTERVAL_LIMIT] <target_list_file>" >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

LIST_FILE="${1:-}"
if [[ -z "$LIST_FILE" ]]; then
    echo "Usage: $0 [--interval_limit INTERVAL_LIMIT] <target_list_file>" >&2
    exit 1
fi
if [[ $# -gt 1 ]]; then
    echo "Error: too many positional arguments: $*" >&2
    echo "Usage: $0 [--interval_limit INTERVAL_LIMIT] <target_list_file>" >&2
    exit 1
fi
if [[ -n "$INTERVAL_LIMIT" && ! "$INTERVAL_LIMIT" =~ ^[0-9]+$ ]]; then
    echo "Error: --interval_limit must be a non-negative integer: '$INTERVAL_LIMIT'" >&2
    exit 1
fi
if [[ ! -f "$LIST_FILE" ]]; then
    echo "Error: target list file '$LIST_FILE' not found." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CAL_ROOT_DIR="${CAL_ROOT_DIR:-$SCRIPT_DIR}"

SKIP_EXISTING="${SKIP_EXISTING:-1}"
MODE="${MODE:-full}"

VALID_LIST="targets_valid.txt"
MISSING_LIST="targets_missing.txt"

# initialize output files
: > "$VALID_LIST"
: > "$MISSING_LIST"

echo "CAL_ROOT_DIR  : $CAL_ROOT_DIR"
echo "TARGET LIST   : $LIST_FILE"
echo "VALID LIST    : $VALID_LIST"
echo "MISSING LIST  : $MISSING_LIST"
echo "SKIP_EXISTING : $SKIP_EXISTING"
echo "MODE          : $MODE"
if [[ -n "$INTERVAL_LIMIT" ]]; then
    echo "INTERVAL_LIMIT: $INTERVAL_LIMIT"
else
    echo "INTERVAL_LIMIT: detector default"
fi
echo

while read -r OBJ OBSID FILTER; do
    # skip empty or comment lines
    if [[ -z "${OBJ:-}" ]] || [[ "${OBJ:0:1}" == "#" ]]; then
        continue
    fi

    EVENT_UF_DIR="${CAL_ROOT_DIR}/${OBJ}/${OBSID}/resolve/event_uf"
    EVENT_CL_DIR="${CAL_ROOT_DIR}/${OBJ}/${OBSID}/resolve/event_cl"

    UF_EVT="xa${OBSID}rsl_p0px${FILTER}_uf.evt"
    CL_EVT="xa${OBSID}rsl_p0px${FILTER}_cl.evt"

    missing_reason=()

    if [[ ! -d "$EVENT_UF_DIR" ]]; then
        missing_reason+=("event_uf_dir_missing")
    elif [[ ! -f "$EVENT_UF_DIR/$UF_EVT" ]]; then
        missing_reason+=("uf_evt_missing")
    fi

    if [[ ! -d "$EVENT_CL_DIR" ]]; then
        missing_reason+=("event_cl_dir_missing")
    elif [[ ! -f "$EVENT_CL_DIR/$CL_EVT" ]]; then
        missing_reason+=("cl_evt_missing")
    fi

    if [[ ${#missing_reason[@]} -ne 0 ]]; then
        echo "[WARN] ${OBJ} ${OBSID} px${FILTER} : ${missing_reason[*]}"
        printf "%-15s %-9s %s  # %s\n" \
            "$OBJ" "$OBSID" "$FILTER" "$(IFS=','; echo "${missing_reason[*]}")" \
            >> "$MISSING_LIST"
        continue
    fi

    # valid target
    printf "%-15s %-9s %s\n" "$OBJ" "$OBSID" "$FILTER" >> "$VALID_LIST"

    echo "========================================"
    echo "OBJECT=${OBJ} OBSID=${OBSID} FILTER=${FILTER}"
    echo "========================================"

    LSCHECK_DIR_NAME="lscheck_${OBJ}_${OBSID}_px${FILTER}"
    LSCHECK_DIR_FULL="${CAL_ROOT_DIR}/${LSCHECK_DIR_NAME}"

    if [[ "$SKIP_EXISTING" -eq 1 && -d "$LSCHECK_DIR_FULL" ]]; then
        echo "[INFO] ${LSCHECK_DIR_FULL} exists, skip"
        continue
    fi

    (
        cd "$EVENT_UF_DIR"

        interval_args=()
        if [[ -n "$INTERVAL_LIMIT" ]]; then
            interval_args=(--interval_limit "$INTERVAL_LIMIT")
        fi

        if [[ "$MODE" == "summary" ]]; then
            run_resolve_ftools_cluster_pileline.sh --summary-only "${interval_args[@]}" "$UF_EVT" "$LSCHECK_DIR_NAME" "$CAL_ROOT_DIR"
        else
            run_resolve_ftools_cluster_pileline.sh "${interval_args[@]}" "$UF_EVT" "$LSCHECK_DIR_NAME" "$CAL_ROOT_DIR"
        fi
    )

    echo
done < "$LIST_FILE"

echo "========================================"
echo "Finished."
echo "Valid targets   : $(wc -l < "$VALID_LIST")"
echo "Missing targets : $(wc -l < "$MISSING_LIST")"
echo "========================================"
