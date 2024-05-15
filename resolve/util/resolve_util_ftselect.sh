#!/bin/sh

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Error: Invalid number of arguments."
   cat << EOF
Usage: resolve_util_ftselect.sh eventfile expr

Description:
  This script is used to perform selection on an event file using the ftselect command.

Arguments:
  eventfile    Path to the event file.
  expr         Selection expression.
  output_suffix   (Optional) Suffix for the output file name. If provided, the output file name will be eventfile_\${output_suffix}.fits. If not provided, the default suffix is '_cut'.

Example:
  resolve_util_ftselect.sh test_pxpr.evt "gtifilter(\"xa300065010rsl_px5000_exp.gti\")"
  resolve_util_ftselect.sh xa300065010rsl_a0pxpr_uf.evt "FLAG_CLIPPED==b0&&gtifilter(\"xa300065010rsl_px1000_exp.gti\")" clip0gtipx1000
EOF

    exit 1
fi

obs="$1"
expr="$2"
outfile_suffix="_cut"

if [ $# -eq 3 ]; then
    outfile_suffix="_$3"
fi

outfile="${obs%.evt}${outfile_suffix}.fits"

echo "Selected event file: $eventfile"
echo "Selection expression: $expr"
echo "Output file: $outfile"

ftselect \
    "infile=$obs" \
    "outfile=$outfile" \
    "expr=$expr" \
    "clobber=yes" \
    "chatter=5"
echo "Output written to $outfile"
