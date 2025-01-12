#!/bin/bash

if [ -z "$1" ]; then
  cat << EOF
usage : run_ecal_xspec_fitMnKa.sh (1: event file)
EOF
  exit 1
fi

# Function to check if a file exists
check_file_exists() {
  local file=$1
  if [ ! -f "$file" ]; then
    echo "Error: File '$file' not found."
    exit 1
  fi
}

# Input arguments
evtfile=$1

check_file_exists "$evtfile"

#for pixel_arg in 12
for pixel_arg in `seq 0 35`
#for pixel_arg in 0
do

# Determine PIXEL filter and output file tag
if [[ "$pixel_arg" =~ ^[0-9]+$ ]]; then
  pixel_filter="PIXEL=$pixel_arg:$pixel_arg"
  # 2桁の0詰め形式に変換
  pixel_tag=$(printf "PIXEL%02d" "$pixel_arg")
elif [[ "$pixel_arg" =~ ^[0-9,:]+$ ]]; then
  pixel_filter="PIXEL=$pixel_arg"
  pixel_tag="PIXEL$(echo "$pixel_arg" | tr -d ':,')"
else
  echo "Error: Invalid PIXEL argument '$pixel_arg'."
  exit 1
fi

echo "START fitting for " $pixel_arg
resolve_ecal_xspec_fitMnKa.sh $evtfile $pixel_arg

fitlog=55Fe_${pixel_tag}.fitlog
check_file_exists "$fitlog"

echo "format fitting result for " $fitlog
resolve_ecal_xspec_get_param.py $fitlog

done 

echo "create summary png file"

resolve_ecal_xspec_plot_fitMnKa.py