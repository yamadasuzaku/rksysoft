#!/usr/bin/env python

import os
import subprocess
import argparse
from astropy.io import fits
import sys
import matplotlib.pyplot as plt
from pdf2image import convert_from_path

n_max_phafile = 10

def generate_xspec_data(flist):
    """
    Generate XSPEC data commands for a list of PHA files.
    
    Args:
        flist (list of str): List of PHA file paths.

    Returns:
        str: XSPEC data commands as a single string.
    """
    return '\n'.join([f"data {i+1}:{i+1} {pha}" for i, pha in enumerate(flist)])

def generate_xspec_data_we(flist, emin, emax):
    """
    Generate XSPEC data and ignore commands with energy ranges.

    Args:
        flist (list of str): List of PHA file paths.
        emin (float): Minimum energy for the ignore range.
        emax (float): Maximum energy for the ignore range.

    Returns:
        tuple: (data commands, ignore commands) as strings.
    """
    data_lines = generate_xspec_data(flist)
    ignore_lines = '\n'.join([f"ignore {i+1}:**-{emin} {emax}-**" for i, pha in enumerate(flist)])
    return data_lines, ignore_lines

def generate_xspec_qdp_label(flist, laboffset=5):
    """
    Generate QDP label commands for XSPEC plots.

    Args:
        flist (list of str): List of PHA file paths.
        laboffset (int): Offset for label indices.

    Returns:
        tuple: (label position commands, label color commands) as strings.
    """
    # Create simplified labels by removing a prefix from the file names
    flabel_list = [pha.replace("rsl_source_", "") for pha in flist]
    # Generate label position commands
    label_lines = '\n'.join([f"LABEL {i+1 + laboffset} VPosition 0.2 {0.98 - 0.03*i} \"{pha}\""
                             for i, pha in enumerate(flabel_list)])
    # Generate label color commands
    label_cols = '\n'.join([f"LABEL {i+1 + laboffset} color {i+1}" for i in range(len(flist))])

    return label_lines, label_cols

def generate_newlines(n):
    """Generate a string with n newline characters."""
    return "\n" * max(0, n)

def check_xspec_data(flist):
    headerlist = []
    for i, pha in enumerate(flist):
        header1 = fits.open(pha)[1].header
        rmf = header1["RESPFILE"]
        check_file_exists(rmf)
        arf = header1["ANCRFILE"]
        check_file_exists(arf)
        obsid = header1["OBS_ID"]
        dateobs = header1["DATE-OBS"]
        objname = header1["OBJECT"]
        exposure = header1["EXPOSURE"]
        headerlist.append(header1)
    return headerlist

def color_text(text, color):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    return f"{colors.get(color, colors['reset'])}{text}{colors['reset']}"

def check_file_exists(filepath):
    if not os.path.exists(filepath):
        print(color_text(f"Error: {filepath} does not exist.", "red"))
        sys.exit(1)

def check_command_exists(command):
    if subprocess.call(f"type {command}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        print(color_text(f"Error: {command} is not available in the PATH.", "red"))
        sys.exit(1)

def run_shell_script(script_content, script_name):
    with open(script_name, "w") as script_file:
        script_file.write(script_content)
    os.chmod(script_name, 0o755)
    subprocess.run(f"./{script_name}", shell=True)
#    os.remove(script_name)

def convert_ps_to_pdf(ps_file, output_pdf=None):
    """
    Convert a PostScript (.ps) file to PDF using the `ps2pdf` command.
    
    Args:
        ps_file (str): Path to the input .ps file.
        output_pdf (str, optional): Path to the output PDF file. 
                                    Defaults to the same name as the .ps file with a .pdf extension.
    """
    # Check if the `ps2pdf` command exists
    if shutil.which("ps2pdf") is None:
        print("Error: `ps2pdf` command is not available.")
        return

    if not os.path.exists(ps_file):
        print(f"Error: {ps_file} not found.")
        return

    # Default output PDF name based on the input file name
    if not output_pdf:
        output_pdf = os.path.splitext(ps_file)[0] + ".pdf"

    try:
        # Execute the `ps2pdf` command
        subprocess.run(["ps2pdf", ps_file, output_pdf], check=True, text=True, capture_output=True)
        print(f"Conversion successful: {output_pdf}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e.stderr}")

def convert_pdf_to_png(pdf_file, output_png=None):
    """
    Convert a PDF file to a PNG image. Only the first page is converted.

    Args:
        pdf_file (str): Path to the input PDF file.
        output_png (str, optional): Path to the output PNG file. 
                                    Defaults to the same name as the PDF file with a .png extension.
    """
    if not os.path.exists(pdf_file):
        print(f"Error: {pdf_file} not found.")
        return

    # Default output PNG name based on the input file name
    if not output_png:
        output_png = os.path.splitext(pdf_file)[0] + ".png"

    try:
        # Convert the first page of the PDF to PNG
        images = convert_from_path(pdf_file, dpi=300)
        images[0].save(output_png, 'PNG')  # Save the first page as a PNG image
        print(f"PNG conversion successful: {output_png}")
    except Exception as e:
        print(f"PNG conversion failed: {str(e)}")

def read_xspec_log(filename):
    """
    Read lines from an XSPEC log file.

    Args:
        filename (str): Path to the XSPEC log file.

    Returns:
        list of str: Lines from the log file.
    """
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return []

    with open(filename, 'r') as f:
        return f.readlines()

def text_to_image(text_lines, output_file, width=800, height=2000, font_size=7):
    """
    Render a list of text lines onto an image and save it as a PNG file.

    Args:
        text_lines (list of str): Lines of text to be rendered on the image.
        output_file (str): Path to the output PNG file.
        width (int): Width of the image in pixels. Defaults to 800.
        height (int): Height of the image in pixels. Defaults to 2000.
        font_size (int): Font size for the text. Defaults to 7.
    """
    # Create a figure with the specified size (adjusted for 100 DPI)
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.axis('off')  # Hide the axes for a clean image

    # Calculate line height and maximum number of lines that fit
    line_height = font_size + 4  # Adjust line spacing
    max_lines = height // line_height

    # Limit the displayed lines to fit within the image
    display_lines = text_lines[:max_lines]

    # Draw each line on the image, starting from the top
    for i, line in enumerate(display_lines):
        ax.text(
            0, 
            1 - (i + 1) * (line_height / height),  # Adjust vertical position
            line.strip(), 
            fontsize=font_size, 
            ha='left', 
            va='top', 
            family='monospace'  # Use monospace font for uniform text alignment
        )

    # Save the figure as a PNG file
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Close the figure to free resources
    print(f"Saved to {output_file}")

def parse_args():
    """
    Parse command-line arguments for the QL spectral fit script.

    Returns:
        argparse.Namespace: Parsed arguments as a Namespace object.
    """
    # Initialize the argument parser with a description and example usage
    parser = argparse.ArgumentParser(
        description='Resolve QL spectral fit',
        epilog='''
        Example:
        (1) Standard usage: resolve_ana_qlfit.py phafilelist
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required argument: list of PHA files
    parser.add_argument('phalist', help='List of PHA files including RMF and ARF file paths.')

    # Optional arguments
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode.')
    parser.add_argument('--show', '-s', action='store_true', help='Show plots using plt.show(). Default is to not display plots.')
    parser.add_argument('--emin', '-l', type=float, default=2.0, help='Minimum energy in keV (default: 2.0).')
    parser.add_argument('--emax', '-x', type=float, default=10.0, help='Maximum energy in keV (default: 10.0).')
    parser.add_argument('--rmf', '-rmf', default=None, help='Response file (RMF).')
    parser.add_argument('--arf', '-arf', default=None, help='Auxiliary response file (ARF).')
    parser.add_argument('--progflags', '-pg', type=str, default="1,1,1",
                        help='Comma-separated flags for different processing steps (e.g., "0,1,0").')
    parser.add_argument('--xscale', '-xs', choices=['off', 'on'], default="on",
                        help='X-axis scale: "off" for linear, "on" for log scale (default: on).')
    parser.add_argument('--fname', '-f', type=str, default='mkspec',
                        help='Output filename tag (default: mkspec).')

    # Parse the arguments
    args = parser.parse_args()

    # Print parsed arguments for verification
    print("----- Configuration -----")
    args_dict = vars(args)  # Convert Namespace to a dictionary for better readability
    for arg, value in args_dict.items():
        print(f"{arg}: {value}")
    print("-------------------------")

    return args

def main():
    ################### Argument Setting ###############################
    # Parse command-line arguments
    args = parse_args()
    emin = args.emin
    emax = args.emax
    rmf = args.rmf
    arf = args.arf
    xscale = args.xscale
    fname = args.fname

    # Read PHA file list from the input file
    with open(args.phalist, 'r') as file:
        phalist = [line.strip() for line in file]

    # Verify the PHA file headers
    headerlist = check_xspec_data(phalist)

    # Generate energy range tag (e.g., "emin2000_emax10000" for emin=2.0 keV, emax=10.0 keV)
    enetag = f"emin{int(1e3 * emin)}_emax{int(1e3 * emax)}"

    ################### Process Flags ###############################
    # Parse program flags (comma-separated values) and convert them to integers
    progflags = args.progflags or ""  # Handle cases where no flags are provided
    flag_values = [int(x) for x in progflags.split(',')]

    # Pad the flag list with zeros if it has fewer than `nprog` entries
    nprog = 3  # Number of supported programs: pleff, plld, fitpl (as of 2024.11.11)
    if len(flag_values) < nprog:
        flag_values += [0] * (nprog - len(flag_values))

    # Convert flag values to a dictionary with True/False for each process
    procdic = {
        "pleff": bool(flag_values[0]),
        "plld": bool(flag_values[1]),
        "fitpl": bool(flag_values[2])
    }
    print(f"Process flags: {procdic}")

    # Additional processing can go here based on the `procdic` dictionary

    header1 = headerlist[0]
    obsid = header1["OBS_ID"]
    dateobs = header1["DATE-OBS"]
    objname = header1["OBJECT"]
    exposure = header1["EXPOSURE"]

    title=f"{obsid} {objname} {dateobs} {exposure:0.2e}s" 

    if procdic["pleff"]:

        datadef = generate_xspec_data(phalist)
        label_lines, label_cols = generate_xspec_qdp_label(phalist)

        # xspec commands
        print(color_text(f"Stage 0: Running xspec to check effective area of {phalist}", "blue"))
        xspec_script = f"""#!/bin/sh  
rm -rf {fname}_eff.ps {fname}_eff.pdf
xspec << EOF

{datadef}

plot efficien

iplot
{label_lines}
{label_cols}
r x 0.5 20.0
r y 0.1 500
LOG Y ON 
font roman
label rotate 
label titile {title}
LOG X {xscale} 1
plot
hard {fname}_eff.ps/cps
q

exit

EOF
        """
        run_shell_script(xspec_script, f"run_xspec_{fname}_eff.sh")   
        convert_ps_to_pdf(f"{fname}_eff.ps")
        convert_pdf_to_png(f"{fname}_eff.pdf")

    if procdic["plld"]:
        # xspec commands
        datadef, ignoredef = generate_xspec_data_we(phalist, emin, emax)
        label_lines, label_cols = generate_xspec_qdp_label(phalist)

        print(color_text(f"Stage 1: Running xspec to plot raw spectrum of {phalist}", "blue"))
        xspec_script = f"""#!/bin/sh  
rm -rf {fname}.ps {fname}.pdf {fname}.pco {fname}.qdp {fname}.fi {fname}.xcm
xspec << EOF

{datadef}
{ignoredef}

show rate 
plot ldata
setplot energy

save file {fname}.fi
save all {fname}.xcm

iplot
{label_lines}
{label_cols}
font roman
label rotate 
label titile {title}
LOG X {xscale} 1
plot
hard {fname}.ps/cps
we {fname}
q

exit

EOF
        """
        run_shell_script(xspec_script, f"run_xspec_{fname}.sh")   
        convert_ps_to_pdf(f"{fname}.ps")
        convert_pdf_to_png(f"{fname}.pdf")


    if procdic["fitpl"]:
        # xspec commands
        datadef, ignoredef = generate_xspec_data_we(phalist, emin, emax)        
        label_lines, label_cols = generate_xspec_qdp_label(phalist)

        print(color_text(f"Stage 2: Running xspec to plot a powerlaw fit of {phalist}", "blue"))
        xspec_script = f"""#!/bin/sh  
rm -rf {fname}_fitpl.ps {fname}_fitpl.pdf {fname}_fitpl.pco {fname}_fitpl.qdp {fname}_fitpl.fi {fname}_fitpl.xcm {fname}_fitpl_xspecfitlog.txt
xspec << EOF

{datadef}
{ignoredef}
show rate 
plot ldata
setplot energy

model tbabs * power 

{generate_newlines(3*n_max_phafile)}

renorm 

fit 100

plot ldata delchi eeuf ratio 

save file {fname}_fitpl.fi 

save all {fname}_fitpl.xcm

log {fname}_fitpl_xspecfitlog.txt

show file 
show rate 
flux 2.0 10.0 
show rate
show model 
show fit 
show pa

log none 

iplot
{label_lines}
{label_cols}

font roman
label rotate 
label titile {title}

LOG X {xscale} 1 2 3 4

plot 

hard {fname}_fitpl.ps/cps
we {fname}_fitpl
q

exit

EOF
        """
        run_shell_script(xspec_script, f"run_xspec_{fname}_fitpl.sh")   
        convert_ps_to_pdf(f"{fname}_fitpl.ps")
        convert_pdf_to_png(f"{fname}_fitpl.pdf")

        text_lines = read_xspec_log(f"{fname}_fitpl_xspecfitlog.txt")
        output_file = f"{fname}_fitpl_xspecfitlog.png"
        text_to_image(text_lines, output_file, width=800, height=2000, font_size=8)

if __name__ == "__main__":
    main()
