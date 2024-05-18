#!/bin/sh

if [ _$1 = _ ];
then
echo "usage : resolve_util_plot_pixelrate.sh"
exit
fi

obs=$1
outfile="${obs%.evt}_pixelrate.txt"
rm -f $outfile

# calc rates 
resolve_util_check_pixelrate.sh $obs | tee $outfile

# plot rates
resolve_util_check_pixelrate_plot.py $outfile
