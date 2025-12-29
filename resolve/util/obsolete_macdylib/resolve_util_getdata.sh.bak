#!/bin/sh

csvfile=${1:-data.csv}  # 引数がなければデフォルト 'data.csv'

if [ ! -f "$csvfile" ]; then
    echo "Error: CSV file '$csvfile' not found."
    exit 1
fi

total_lines=$(wc -l < "$csvfile")
echo "Processing CSV file: $csvfile (Total lines: $total_lines)"

echo "=== Step 1: getdata ==="
date
resolve_util_getdata.py -f "$csvfile"
echo "getdata completed."

echo "=== Step 2: decrypt ==="
date
resolve_util_decrypt.py -f "$csvfile"
echo "decrypt completed."

echo "=== Step 3: unzip ==="
date
resolve_util_gunzip.py -f "$csvfile"
echo "unzip completed."

echo "All steps completed successfully."
