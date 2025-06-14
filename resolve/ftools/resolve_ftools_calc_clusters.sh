#!/bin/bash

# Set default values
pixel=19
evtfile=""
allpixel=false

# Display usage instructions
usage() {
    echo "Usage: $0 -f <evtfile> [-p <pixel_number>] [-a]"
    echo "  -f <evtfile>       Input event file (required)"
    echo "  -p <pixel_number>  Pixel number (default: 19, range: 0–35)"
    echo "  -a                 Process all pixels (overrides pixel selection)"
    exit 1
}

# Parse command-line arguments
while getopts ":f:p:a" opt; do
    case $opt in
        f)
            evtfile="$OPTARG"
            ;;
        a)
            allpixel=true
            ;;
        p)
            pixel="$OPTARG"
            if ! [[ $pixel =~ ^[0-9]+$ ]] || [ "$pixel" -lt 0 ] || [ "$pixel" -gt 35 ]; then
                echo "Error: Pixel number must be an integer between 0 and 35."
                exit 1
            fi
            ;;
        *)
            usage
            ;;
    esac
done

# Check if the required input file is specified
if [ -z "$evtfile" ]; then
    echo "Error: evtfile is required."
    usage
fi

# Check if the specified file exists
if [ ! -f "$evtfile" ]; then
    echo "Error: File '$evtfile' not found."
    exit 1
fi

# Branch depending on whether all pixels are processed or just one
if $allpixel; then
    echo "Processing all pixels for evtfile: $evtfile"
    resolve_ana_pixel_Ls_define_cluster.py "$evtfile"
    
    # Output filenames for all-pixel case
    cluster_outfile="addcluster_${evtfile%.evt}.evt"
    cluster_outfile_imposi="addcluster_${evtfile%.evt}_imposi.evt"
    cluster_outfile_im1="addcluster_${evtfile%.evt}_im1.evt"
else
    # Format the pixel number as a two-digit string
    pixel_padded=$(printf "%02d" "$pixel")
    # Output filenames for individual pixel
    pixelcut_outfile="${evtfile%.evt}_p${pixel_padded}.evt"
    cluster_outfile="addcluster_${evtfile%.evt}_p${pixel_padded}.evt"
    cluster_outfile_imposi="addcluster_${evtfile%.evt}_p${pixel_padded}_imposi.evt"
    cluster_outfile_im1="addcluster_${evtfile%.evt}_p${pixel_padded}_im1.evt"

    echo "Processing pixel $pixel for evtfile: $evtfile (outfile: $pixelcut_outfile)"
    ftselect infile="$evtfile" outfile="$pixelcut_outfile" expr="PIXEL==$pixel" chatter=5 clobber=yes

    resolve_ana_pixel_Ls_define_cluster.py "$pixelcut_outfile"
fi

# Common post-processing: apply selection based on IMEMBER column
ftselect infile="$cluster_outfile" \
         outfile="$cluster_outfile_imposi" \
         expr="IMEMBER>0" chatter=5 clobber=yes

ftselect infile="$cluster_outfile" \
         outfile="$cluster_outfile_im1" \
         expr="IMEMBER==1" chatter=5 clobber=yes

# Display output file summary
echo "Processing completed. Output files:"
echo "  $cluster_outfile"
echo "  $cluster_outfile_imposi"
echo "  $cluster_outfile_im1"

if $allpixel; then
    echo "  finish"
else
    echo "  $pixelcut_outfile"
    echo "  finish"
fi
