#!/bin/bash

if [ _$1 = _ ];
then
cat << EOF
usage : xtend_create_detimg.sh (1:cleaned event file)
EOF
exit
fi

evt_file=$1
new_filename="${evt_file%.evt}_detimg.fits"

tmp=tmp.log
rm -f $tmp
rm -f $new_filename

cat << EOF > exclude_calsource.reg
physical
-circle(920.0,1530.0,92.0)
-circle(919.0,271.0,91.0)
EOF

xselect << EOF 2>&1 | tee $tmp

no

read event $evt_file ./
yes


set image DET
filter region exclude_calsources.reg
filter pha_cutoff 83 1667

extract image

save image $new_filename
exit
no

EOF


