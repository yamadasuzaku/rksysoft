#!/bin/bash

# Check if a filename argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <filename.evt>"
    exit 1
fi

# Extract the filename without extension
filename="$1"
basename="${filename%.*}"

# Run create pixel rate using ftlist 
resolve_util_check_pixelrate.sh "$filename" > "${basename}_pixelrate.txt"

# Run plot rate
resolve_util_check_pixelrate_plot.py "${basename}_pixelrate.txt"

echo "Process completed for $filename"
