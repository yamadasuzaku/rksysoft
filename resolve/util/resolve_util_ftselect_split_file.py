#!/usr/bin/env python

import argparse
import subprocess
import os

def generate_unique_filename(base_name, itypenames, plotpixels, slope_differ, quick_double, extension=".fits"):
    """Generate a descriptive file name based on the input parameters."""
    if itypenames == '0,1,2,3,4':
        itypes = 'all'
    else:
        itypes = '_'.join(itypenames.split(','))

    if plotpixels == ','.join(map(str, range(36))):
        pixels = 'all'
    else:
        pixels = '_'.join([f"{int(p):02d}" for p in plotpixels.split(',')])

    return f"{base_name}_itype_{itypes}_slope_{slope_differ}_quick_{quick_double}_pixel_{pixels}{extension}"

def main():
    parser = argparse.ArgumentParser(description='')
    parser = argparse.ArgumentParser(
        description='Run ftselect to split event files',
        epilog='Examples of usage:\n'
               '(      Hp, pixel2, qd=b01, sd=b01) resolve_util_ftselect_split_file.py xa000114000rsl_a0pxpr_uf_fillprenext_cutclgti.fits -y 0 -p 2 \n'
               '(all type, pixel2, qd=b00, sd=b00) resolve_util_ftselect_split_file.py xa000114000rsl_a0pxpr_uf_fillprenext_cutclgti.fits -qd b00 -sd b00 -p2 -p 0  \n'
               '(all type & pixel, qd=b00, sd=b00) resolve_util_ftselect_split_file.py xa000114000rsl_a0pxpr_uf_fillprenext_cutclgti.fits\n'
               , 
        formatter_class=argparse.RawTextHelpFormatter
    )    
    parser.add_argument("infile", type=str, help='Path to the input event file.')
    parser.add_argument('--outfile', '-o', type=str, default="auto", help='Path to the output event file (default: auto).')
    parser.add_argument('--itypenames', '-y', type=str, help='Comma-separated list of ITYPE values (default: "0,1,2,3,4")', default='0,1,2,3,4')
    parser.add_argument('--plotpixels', '-p', type=str, help='Comma-separated list of pixels to plot (default: "0,1,2,...,35")', default=','.join(map(str, range(36))))
    parser.add_argument('--chatter', '-c', type=int, help='Level of verbosity for the command output (default: 5)', default=5)
    parser.add_argument('--clobber', '-cl', type=str, choices=['yes', 'no'], help='Whether to overwrite the output file if it exists (default: yes)', default='yes')
    parser.add_argument('--quick_double', '-qd', type=str, choices=['b00', 'b01'], help='Value for QUICK_DOUBLE (default: b01)', default='b01')
    parser.add_argument('--slope_differ', '-sd', type=str, choices=['b00', 'b01'], help='Value for SLOPE_DIFFER (default: b01)', default='b01')

    args = parser.parse_args()

    # Handle automatic output file name generation
    if args.outfile == "auto":
        base_name = os.path.splitext(os.path.basename(args.infile))[0]
        args.outfile = generate_unique_filename(base_name, args.itypenames, args.plotpixels, args.slope_differ, args.quick_double)

    # Convert input arguments into expression for ftselect if not default
    expr_parts = []

    if args.plotpixels != ','.join(map(str, range(36))):
        pixel_expr = '||'.join([f"(PIXEL=={p})" for p in args.plotpixels.split(',')])
        expr_parts.append(f"({pixel_expr})")

    if args.itypenames != '0,1,2,3,4':
        itype_expr = '||'.join([f"(ITYPE=={i})" for i in args.itypenames.split(',')])
        expr_parts.append(f"({itype_expr})")

    expr_parts.append(f"(QUICK_DOUBLE=={args.quick_double})")
    expr_parts.append(f"(SLOPE_DIFFER=={args.slope_differ})")

    expr = '&&'.join(expr_parts)

    # Call the bash script
    command = [
        'bash', 'resolve_util_ftselect_with_oname.sh',
        args.infile,
        args.outfile,
        expr,
        str(args.chatter),
        args.clobber
    ]

    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)

if __name__ == '__main__':
    main()
