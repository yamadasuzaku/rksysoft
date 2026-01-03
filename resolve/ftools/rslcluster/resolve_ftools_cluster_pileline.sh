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

# --- Check input argument ---
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_file_uf.evt> (cl.evt is also needed in resolve_ftools_add_prevnext.sh) " >&2
    exit 1
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
    -d

echo ">>> Running small cluster detection on large_$base_name"
resolve_ftools_detect_pseudo_event_clusters.py "large_$base_name" \
    --mode small \
    --col_cluster ICLUSTERS \
    --col_member IMEMBERS \
    --outname small_ \
    -d

# --- Step 3: Run QL diagnostic tool ---
final_file="small_large_${input_file%.evt}_noBL_prevnext_cutclgti.evt"
check_file_exists "$final_file"

echo ">>> Running resolve_ftools_qlcheck_cluster.py on $final_file"
resolve_ftools_qlcheck_cluster.py "$final_file"
