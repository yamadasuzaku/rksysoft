#!/bin/bash

# Default values for elements, ranges, and nh_m2
elements=("fe")
ranges=("25 26")
nh_m2=23

# Check for command-line arguments to update the values
while getopts ":e:r:n:" opt; do
  case $opt in
    e) IFS=',' read -r -a elements <<< "$OPTARG" ;;  # Elements passed as comma-separated string
    r) IFS=',' read -r -a ranges <<< "$OPTARG" ;;    # Ranges passed as comma-separated string
    n) nh_m2=$OPTARG ;;                              # Update nh_m2
    \?) echo "Invalid option -$OPTARG" >&2 ;;
  esac
done

# Get the length of the elements array
length=${#elements[@]}

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
    echo "Processing ${element}${i} with nh_m2=${nh_m2}..."

    # Run SPEX with the current element, range, and nh_m2
    spex << EOF
log save spexcom_${element}${i}_nh${nh_m2} overwrite
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
plot ry 0.005 10000.0
plot x log 
plot y log 
plot fill disp false 
plot
plot device cps spexfig_${element}${i}_nh${nh_m2}.ps 
plot 
plot close 2

plot x lin
plot y lin
# dump qdpfile
plot adum spexqdp_${element}${i}_nh${nh_m2} overwrite

# dump absorption line properties as .asc file
ascdump file spextral_${element}${i}_nh${nh_m2} 1 2 tral

quit
EOF
done
done

# Move output files to their respective directories
echo "Moving output files to directories..."
mv spexqdp_*_nh${nh_m2}.qdp output_spexqdp
mv spextral_*_nh${nh_m2}.asc output_spextral
mv spexfig*_nh${nh_m2}.ps output_spexfig
mv spexcom_*_nh${nh_m2}.com output_spexcom

# Log the end time
end_time=$(date)
echo "Script ended at: $end_time"
