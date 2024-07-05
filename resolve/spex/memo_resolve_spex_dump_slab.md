# SPEX : a test scirpt to dump slab's data

This shell script processes spectral data for different elements and their respective ranges, generating output files and organizing them into directories. Below is a detailed explanation of each part of the script.

# Usage 

When `SPEX` is ready, run the script. 

``` bash:
resolve_spex_dump_slab.sh
```

## Detail 

- The script defines two arrays: `elements` and `ranges`. Each element in the `elements` array corresponds to a range in the `ranges` array. The ranges specify the start and end indices for processing each element.

- The length of the `elements` array is stored in the `length` variable.

- The script creates four directories (`output_qdp`, `output_asc`, `output_png`, and `output_com`) to store different types of output files. The `-p` option ensures that the directories are created only if they do not already exist.

- The script logs the start time using the `date` command and echoes it to the console.

- The script loops through each element and its corresponding range. The `awk` command is used to extract the start and end values from the `ranges` array.

- Within each element, the script loops through the specified range using the `seq` command, which generates a sequence of numbers formatted with two digits.

- The script runs the SPEX software for the current element and range, saving logs and generating plots. The SPEX commands are provided within a here-document (`<< EOF ... EOF`).

- The script runs a Python script (`resolve_spex_plotmodel_fromqdp_with_tral.py`) to process the generated QDP files and produce PNG images. It echoes a message when each PNG file is generated.

- After processing all elements and ranges, the script moves the generated files to their respective directories.

- Finally, the script logs the end time using the `date` command and echoes it to the console.
