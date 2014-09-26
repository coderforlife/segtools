

These scripts were pulled from rocce and were located
in /data/scratch/aperez/ZT16.  They are scripts Alex uses
to extract png files from mrc stack and to histogram equalize
those png files

mrcstack2png.sh
===============

Converts .mrc file to png files


generate_reference.sh  
=====================
 
Calls generate_reference.m matlab script to create a text file containing a histogram
of pixel intensities.  The histogram has 256 bins (it assumes 8 bit image)


process_td.sh
=============

Extracts training data from .mrc file


run_ehs.sh
==========

Calls run_ehs.m to histogram equalize the images using the histogram files from
generate_reference.sh.  Alex mentioned this needs matlab r2013a or better to work


compiled_versions
=================

This directory contains slightly altered versions of above scripts that have
been compiled via the matlab compiler.  See the README.md within that
directory for more information
