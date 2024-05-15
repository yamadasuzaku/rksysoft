#!/bin/sh

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Error: Invalid number of arguments."
   cat << EOF
Usage: resolve_pr_plot_v1_loop.sh prfile (plot_flag)

Description:
  This script is used to plot pulse records 

Arguments:
  infile      prfitsfile
  plot_flag   (Optional) 

Example:
  resolve_pr_plot_v1_loop.sh xa300065010rsl_a0pxpr_uf_clip1gtipx1000.fits

EOF
    exit 1
fi

infile="$1"
flag=""

if [ $# -eq 2 ]; then
    flag="--plot_flag"
fi

shfile=./resolve_pr_plot_v1.py

for pixel in `seq 0 35`
do
for itype in `seq 0 5`
do
    echo $pixel $itype
$shfile $infile --target_pixel=$pixel --target_itype=$itype $flag
done
done

