#!/bin/sh

# Check if the correct number of arguments is provided
if [ $# -eq 1 ]; then
  echo "===== start $0"
else
  echo "Error: Invalid number of arguments."
  # Display usage information if the number of arguments is incorrect
  cat << EOF
Usage: xtend_pileup_check_quick.sh cl.evt

Description:
  This script is used to check pileup of Xtend using cl.evt

Arguments:
  cl           Cleaned Event File

Example:
- quick check
  xtend_pileup_check_quick.sh xa300041010xtd_p0300000a0_cl.evt 
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

# Function to check if a command exists in PATH
check_command_exists() {
  local cmd=$1
  if ! command -v $cmd >/dev/null 2>&1; then
    echo "Error: Command '$cmd' not found in PATH."
    exit 1
  fi
}

# Check if the required commands are available in PATH
check_command_exists xtend_create_img.sh
check_command_exists xtend_pileup_gen_plfraction.py
check_command_exists xselect

# Quick analysis from cl.evt
cl=$1
check_file_exists $cl
clbase=$(basename "$cl" | sed -e 's/\.evt\.gz$//' -e 's/\.evt$//')
climg=${clbase}.img

# Display the cleaned event file and the image file names
echo "clevt : $cl  climg : $climg"

# Step 1: Create an image from the cleaned event file
echo "(1) create image from cl event"
xtend_create_img.sh $cl

# Step 2: Check pileup
echo "\n(2) check pileup"
check_file_exists $climg
xtend_pileup_gen_plfraction.py $climg
