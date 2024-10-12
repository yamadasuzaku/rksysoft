#!/bin/bash

# Define elements and their respective ranges
elements=("fe")
ranges=("25 26")
#elements=("fe" "ca" "ni")
#ranges=("1 27" "1 21" "1 29")

# Get the length of the elements array
length=${#elements[@]}
nh_m2=23
# Create directories for output if they do not exist
mkdir -p output_spexqdp
mkdir -p output_spexcom
mkdir -p output_spexfig
mkdir -p output_spextral

# Log the start time
start_time=$(date)
echo "Script started at: $start_time"

# Loop through each element and its range
for (( j=0; j<$length; j++ )); do
  element=${elements[$j]}
  start=$(echo ${ranges[$j]} | awk '{print $1}')
  end=$(echo ${ranges[$j]} | awk '{print $2}')

  # Loop through the specified range
  for i in $(seq -f "%02g" $start $end); do
    echo "Processing ${element}${i}..."

    # Run SPEX with the current element and range
    spex << EOF
log save spexcom_${element}${i} overwrite
plot device xs
plot type model
egrid lin 500:10000 9500 eV
# set plot style
com pow
com slab
com r 1 2
par 1 1 norm value 100000
par 1 2 v    value 1.0
par 1 2 rms  value 1.0
par 1 2 dv   value 1.0
par 1 2 ${element}${i} value ${nh_m2}
calc 
par show 

plot rx 0.5 10
plot ry 0.005 5000.0
plot x log 
plot y log 
plot fill disp false 
plot
plot device cps spexfig_${element}${i}.ps 
plot 
plot close 2

plot x lin
plot y lin
# dump qdpfile
plot adum spexqdp_${element}${i} overwrite

# dump absorption line properties as .asc file
ascdump file spextral_${element}${i} 1 2 tral

quit
EOF
done
done

# Move output files to their respective directories
echo "Moving output files to directories..."
mv spexqdp_*.qdp output_spexqdp
mv spextral_*.asc output_spextral
mv spexfig*.ps output_spexfig
mv spexcom_*.com output_spexcom

# Log the end time
end_time=$(date)
echo "Script ended at: $end_time"
