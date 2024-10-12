plot device xs
plot type model
egrid lin 500 : 10000 9500 eV
# set plot style
comp pow
comp slab
comp relation 1 2
par 1 1 norm value 100000
par 1 2 v value 1.0
par 1 2 rms value 1.0
par 1 2 dv value 1.0
par 1 2 fe25 value 23
calculate
par show
par show
plot rx 0.5 10
plot ry 0.005 10000.0
plot x log
plot y log
plot fill disp false
plot
plot device cps spexfig_fe25_nh23.ps
plot
plot close 2
plot close 2
plot x lin
plot y lin
# dump qdpfile
plot adum spexqdp_fe25_nh23 overwrite
plot adum spexqdp_fe25_nh23 overwrite
# dump absorption line properties as .asc file
ascdump file spextral_fe25_nh23 1 2 tral
ascdump file spextral_fe25_nh23 1 2 tral
