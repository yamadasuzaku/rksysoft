#!/bin/bash

# Default values
itype=0
boundary=10000
order=4

# Check for mandatory argument
if [ -z "$1" ]; then
    echo "Error: fname is a mandatory argument."
    echo "Usage: $0 <filename> [--itype <itype>] [--boundary <boundary>] [--order <order>]"
    exit 1
fi

# Set fname
fname="$1"
shift

# Parse optional arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --itype) itype="$2"; shift ;;
        --boundary) boundary="$2"; shift ;;
        --order) order="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Generate file tag
ftag="${fname%.evt}"

echo "Starting the process with the following settings:"
echo "Filename: $fname"
echo "itype: $itype"
echo "boundary: $boundary"
echo "order: $order"
echo "File tag: $ftag"

for pixel in $(seq 0 35); do
pixstr=$(printf "%02d" "$pixel")
echo "Processing pixel: $pixel (formatted as $pixstr)"
  
echo "Running resolve_ecal_pha_pi.py for pixel $pixstr..."
./resolve_ecal_pha_pi.py $fname TIME 1,1 PHA,PI 1,1 --filters PIXEL==$pixel,ITYPE==$itype -o $pixstr
echo "resolve_ecal_pha_pi.py completed for pixel $pixstr."
    
echo "Running resolve_ecal_fitpoly_csv.py for pixel $pixstr..."
./resolve_ecal_fitpoly_csv.py fplot_${ftag}_p${pixstr}.csv PHA PI ${boundary} $order $order fitpoly_p${pixstr}.png
echo "resolve_ecal_fitpoly_csv.py completed for pixel $pixstr."
done

echo "All processes completed."
