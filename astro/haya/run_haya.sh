#!/bin/bash

#cut_rate=10
cut_rate=50

echo $cut_rate

# create lightcurve
resolve_ana_haya_mklc.py xa201064010rsl_p0px1000_cl.evt 

# create gtifile  RATE > 50
resolve_ana_haya_maketime.py xa201064010rsl_p0px1000_cl.lc --cut_rate $cut_rate

# create event and pi file 
resolve_ana_haya_mkpi_gticut.py xa201064010rsl_p0px1000_cl.evt xa201064010rsl_p0px1000_cl_cut${cut_rate}.gti
