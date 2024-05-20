#!/bin/sh

if [ $# -eq 2 ]; then
  echo "===== start $0"
else
  echo "Error: Invalid number of arguments."
  cat << EOF
Usage: resolve_pileup_check_detail_v0.sh cl.evt uf.evt

Description:
  This script is used to check pileup of Xtend. 

Arguments:
  cl           Cleaned Event File
  uf           Unscreend Event File 

Example:
- detailed check 
  xtend_pileup_run.sh xa300041010xtd_p0300000a0_cl.evt xa300041010xtd_p0300000a0_uf.evt
EOF
exit
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

# quick analysis from cl.evt 
cl=$1
check_file_exists $cl
clbase=$(basename "$cl" | sed -e 's/\.evt\.gz$//' -e 's/\.evt$//')
climg=${clbase}.img
echo "clevt : $cl  climg : $climg"
echo "(1) create image from cl event"
xtend_create_img.sh $cl

echo "\n(2) check pileup"
check_file_exists $climg
xtend_pileup_gen_plfraction.py $climg

# detailed analysis using uf.evt
uf=$2
check_file_exists $uf
ufbase=$(basename "$uf" | sed -e 's/\.evt\.gz$//' -e 's/\.evt$//')

echo "(3) extract events from the input uf event with several filters"
xtend_pileup_genevt_v0.py $uf $cl

echo "(4) generate images from the events"
for ev in `ls *clgti.evt`
do
echo $ev
xtend_create_img_nocut.sh $ev
done
 
echo "(5) check pileup from several uf events"
for ev in `ls *clgti.img`
do
echo $ev
xtend_pileup_gen_plfraction.py $ev
done
