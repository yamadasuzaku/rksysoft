# Xtend Pileup Check Quick Script

This script is used to check pileup of Xtend using a cleaned event file (`cl.evt`).

## Usage

Run [xtend_pileup_check_quick.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/xtend/xtend_pileup_check_quick_v0.sh) with a cleaned event file. 

```sh
xtend_pileup_check_quick.sh cl.evt
```

## Detailed Steps

### Check Required Commands

The script verifies that the following commands are available in the system's PATH:
- [xtend_create_img.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/xtend/xtend_create_img.sh)
- [xtend_pileup_gen_plfraction.py](https://github.com/yamadasuzaku/rksysoft/blob/main/xtend/xtend_pileup_gen_plfraction.py)
- xselect

### Check File Existence

The script checks if the provided cleaned event file (cl.evt) exists.

### Create Image from Cleaned Event File

The script calls [xtend_create_img.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/xtend/xtend_create_img.sh) to create an image from the cleaned event file.

### Check Pileup

Finally, the script calls [xtend_pileup_gen_plfraction.py](https://github.com/yamadasuzaku/rksysoft/blob/main/xtend/xtend_pileup_gen_plfraction.py) to check the pileup of the generated image file.
