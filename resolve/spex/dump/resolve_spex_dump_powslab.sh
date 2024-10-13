#!/bin/bash

# Default values for elements, ranges, and nh_m2
elements=("fe")
ranges=("24 26")
nh_m2=23

# usage 関数を定義して、使用方法を表示
usage() {
  echo "Usage: $0 [-e elements] [-r ranges] [-n nh_m2]"
  echo "  -e elements : Elements to process (comma-separated, e.g., 'fe,ca,ni')"
  echo "  -r ranges   : Corresponding ranges (comma-separated, e.g., '1 27,1 21,1 29')"
  echo "  -n nh_m2    : Value of nh_m2 (e.g., 25)"
  echo "Example: $0 -e 'fe,ca,ni' -r '1 27,1 21,1 29' -n 22"
  exit 1
}

# getopts を使ってコマンドラインオプションを解析
while getopts ":e:r:n:h" opt; do
  case $opt in
    e) IFS=',' read -r -a elements <<< "$OPTARG" ;;  # -e の引数を配列に
    r) IFS=',' read -r -a ranges <<< "$OPTARG" ;;    # -r の引数を配列に
    n) nh_m2=$OPTARG ;;                              # -n の引数を変数に
    h) usage ;;                                      # -h が指定されたら usage を表示
    \?) echo "Invalid option -$OPTARG" >&2; usage ;; # 無効なオプションが指定されたら usage を表示
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
var calc new
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
