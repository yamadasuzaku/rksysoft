#!/bin/bash

# create lightcurve
resolve_ana_haya_mklc.py xa201064010rsl_p0px1000_cl.evt

# create gtifile  RATE > 50
resolve_ana_haya_maketime.py xa201064010rsl_p0px1000_cl.lc


# create event and pi file 
resolve_ana_haya_mkpi_gticut.py xa201064010rsl_p0px1000_cl.evt xa201064010rsl_p0px1000_cl_cut50.gti
