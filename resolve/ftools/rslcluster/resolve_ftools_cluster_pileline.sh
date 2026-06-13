#!/bin/bash

# ---------------------------------------------
# Script to run the full clustering pipeline
# Step 1: Add prev/next interval info
# Step 2: Run large/small cluster analysis
# Step 3: Perform cluster diagnostic check
# ---------------------------------------------

# --- Function to check if required file exists ---
check_file_exists() {
    if [ ! -f "$1" ]; then
        echo "Error: Required file '$1' not found." >&2
        exit 1
    fi
}

# --- Option parser ---
print_usage() {
    echo "Usage: $0 [--interval_limit INTERVAL_LIMIT] <input_file_uf.evt>" >&2
    echo "       (cl.evt is also needed in resolve_ftools_add_prevnext.sh)" >&2
}

interval_limit=""

while [ $# -gt 0 ]; do
    case "$1" in
        --interval_limit)
            if [ $# -lt 2 ]; then
                echo "Error: --interval_limit requires an integer value." >&2
                print_usage
                exit 1
            fi
            interval_limit="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Error: unknown option '$1'." >&2
            print_usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

# --- Check input argument ---
if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

if [ $# -gt 1 ]; then
    echo "Error: too many positional arguments: $*" >&2
    print_usage
    exit 1
fi

if [ -n "$interval_limit" ] && ! [[ "$interval_limit" =~ ^[0-9]+$ ]]; then
    echo "Error: --interval_limit must be a non-negative integer: '$interval_limit'" >&2
    exit 1
fi

interval_limit_args=()
if [ -n "$interval_limit" ]; then
    interval_limit_args=(--interval_limit "$interval_limit")
    echo ">>> Using --interval_limit $interval_limit for large/small cluster detection"
fi

# --- Input file name ---
input_file="$1"
base_top=$(basename "$input_file" _uf.evt)
cl_file="${base_top}_cl.evt"

check_file_exists "$input_file"
echo "[check] cl.evt file is needed to create cl.gti in resolve_ftools_add_prevnext.sh."
check_file_exists "$cl_file" 

# --- Step 1: Add prev/next interval columns ---
echo ">>> Running resolve_ftools_add_prevnext.sh on $input_file"
resolve_ftools_add_prevnext.sh "$input_file"

# --- Step 2: Run clustering for large and small pseudo events ---
base_name="${input_file%.evt}_noBL_prevnext_cutclgti.evt"
check_file_exists "$base_name"

echo ">>> Running large cluster detection on $base_name"
resolve_ftools_detect_pseudo_event_clusters.py "$base_name" \
    --mode large \
    --col_cluster ICLUSTERL \
    --col_member IMEMBERL \
    --outname large_ \
    "${interval_limit_args[@]}" \
    -d

echo ">>> Running small cluster detection on large_$base_name"
resolve_ftools_detect_pseudo_event_clusters.py "large_$base_name" \
    --mode small \
    --col_cluster ICLUSTERS \
    --col_member IMEMBERS \
    --outname small_ \
    "${interval_limit_args[@]}" \
    -d

# --- Step 3: Run QL diagnostic tool ---
final_file="small_large_${input_file%.evt}_noBL_prevnext_cutclgti.evt"
check_file_exists "$final_file"

echo ">>> Running resolve_ftools_qlcheck_cluster.py on $final_file"
resolve_ftools_qlcheck_cluster.py "$final_file"
