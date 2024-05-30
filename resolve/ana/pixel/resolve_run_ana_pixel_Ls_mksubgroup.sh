#!/bin/sh

for i in `seq 0 35`
do
echo ".............. start"  
echo PIXEL = $i 
resolve_ana_pixel_Ls_mksubgroup.py xa000114000rsl_p0px1000_uf_prevnext_cutclgti.fits TIME PREV_INTERVAL -f "PIXEL==${i}"
echo ".............. finish"  
done
