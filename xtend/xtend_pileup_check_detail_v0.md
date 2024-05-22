# Xtend Pileup Check Detailed Script

This script is used to check pileup of Xtend using an unscreened event file, using GTI of el.evt. 

## Usage

Run [xtend_pileup_check_detail_v0.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/xtend/xtend_pileup_check_detail_v0.sh) with a cleaned event file and an unscreened event. 

```sh
xtend_pileup_check_detail_v0.sh cl.evt uf.evt
```

## Detailed Steps

### Check Required Commands

The script verifies that the following commands are available in the system's PATH:
- [xtend_create_img.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/xtend/xtend_create_img.sh)
- [xtend_pileup_gen_plfraction.py](https://github.com/yamadasuzaku/rksysoft/blob/main/xtend/xtend_pileup_gen_plfraction.py)
- [xtend_pileup_genimg_uf.py](https://github.com/yamadasuzaku/rksysoft/blob/main/xtend/xtend_pileup_genimg_uf.py)
- [xtend_util_ftmgtime.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/xtend/xtend_util_ftmgtime.sh)
- xselect

### Flow

- (1) create image from cl event 

``` bash:
xtend_create_img.sh $cl
```

- (2) check pileup 

``` bash:
xtend_pileup_gen_plfraction.py $climg
```

- (3) create GTI from the cl.evt

``` bash:
xtend_util_ftmgtime.sh $cl
clgti=${clbase}.gti
check_file_exists $clgti
```

- (4) generate images from the events

``` bash:
xtend_pileup_genimg_uf.py $uf $cl $clgti 
```
 
- (5) check pileup from several uf events

``` bash:
for ev in `ls *clgti.img`
do
echo $ev
xtend_pileup_gen_plfraction.py $ev
done
``` 