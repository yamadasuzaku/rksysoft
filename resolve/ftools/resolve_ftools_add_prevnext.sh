#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e
set -o pipefail

# --- Function to check if required commands are available ---
check_command_exists() {
    for cmd in "$@"; do
        if ! command -v "$cmd" &> /dev/null; then
            echo "Error: Command '$cmd' not found in PATH." >&2
            exit 1
        fi
    done
}

# --- Function to check if required files exist ---
check_file_exists() {
    if [ ! -f "$1" ]; then
        echo "Error: Required file '$1' does not exist." >&2
        exit 1
    fi
}

# --- Input argument check ---
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_file_uf.evt>" >&2
    exit 1
fi

input_file="$1"
base_top=$(basename "$input_file" _uf.evt)
cl_file="${base_top}_cl.evt"

# --- Check availability of required commands ---
check_command_exists \
    resolve_util_ftselect.sh \
    resolve_util_ftmgtime.sh \
    resolve_tool_addcol_prev_next_interval.py

# --- Check if required input files exist ---
check_file_exists "$input_file"
check_file_exists "$cl_file"

# --- Step 1: Remove BL events (ITYPE >= 5) from the UF file ---
echo "Running resolve_util_ftselect.sh to remove BL..."
resolve_util_ftselect.sh "$input_file" "ITYPE<5" noBL
echo "Completed: BL removal"

# --- Step 2: Add previous/next interval columns to the cleaned event file ---
base_name="${base_top}_uf_noBL"
evt_prevnext="${base_name}_prevnext.evt"

echo "Running resolve_tool_addcol_prev_next_interval.py..."
resolve_tool_addcol_prev_next_interval.py "${base_name}.evt" -o "$evt_prevnext"
echo "Completed: prev/next interval columns added"

# --- Step 3: Generate GTI file from the CL file ---
echo "Running resolve_util_ftmgtime.sh to create GTI..."
resolve_util_ftmgtime.sh "$cl_file"
echo "Completed: GTI creation"

# --- Step 4: Apply GTI filtering to the event file ---
echo "Running resolve_util_ftselect.sh to apply GTI filtering..."
resolve_util_ftselect.sh "$evt_prevnext" "gtifilter(\"${base_top}_cl.gti\")" cutclgti
echo "Completed: GTI filtering"

echo "All tasks completed successfully."
