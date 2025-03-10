#!/bin/bash

wget --content-disposition -N https://paperpile.com/eb/tozsOWANLH

wget --content-disposition -N https://paperpile.com/eb/kDJhOrGtHV

wget --content-disposition -N https://paperpile.com/eb/SWkixDABKH

wget --content-disposition -N https://paperpile.com/eb/uZktvxulCw

wget --content-disposition -N https://paperpile.com/eb/IwPKEiYPmA


for file in ASTRO-H.bib Cyg-X-1.bib Plasma-Physics.bib Calorimeter.bib XRISM.bib
do
rk_paper_util_bibtexparser.py $file    
done
