#!/bin/bash

# Define elements and their respective ranges
#elements=("fe")
#ranges=("25 26")
elements=("fe" "ca" "ni")
ranges=("1 27" "1 21" "1 29")

# Get the length of the elements array
length=${#elements[@]}

# Create directories for output if they do not exist
mkdir -p output_qdp
mkdir -p output_asc
mkdir -p output_png
mkdir -p output_com

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
log save ${element}${i} overwrite
plot device xs
plot type model
plot fill disp false 
egrid lin 500:10000 9500 eV
com pow
com slab 
com r 1 2 
par 1 1 norm v 100000
par 1 2 ${element}${i} v 23
calc 
plot adum ${element}${i} overwrite
ascdump file ${element}${i} 1 2 tral
quit
EOF
    # Run the resolve_spex_plotmodel_fromqdp_with_tral.py script
    resolve_spex_plotmodel_fromqdp_with_tral.py ${element}${i}.qdp -s ${element}${i}.asc --output ${element}${i}_spec.png --emin 0.5 --y1 0.5 --y2 5.0
    echo "Generated ${element}${i}_spec.png"

  done
done

# Move output files to their respective directories
echo "Moving output files to directories..."
mv *.qdp output_qdp
mv *.asc output_asc
mv *.png output_png
mv *.com output_com

# Log the end time
end_time=$(date)
echo "Script ended at: $end_time"
