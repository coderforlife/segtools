These scripts were pulled from rocce and were located
in /data/scratch/aperez/ZT16

exact_histogram.m        generate_reference.m   mrcstack2png.sh  process_td.sh  run_ehs.m
find_nonborder_pixels.m  


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
generate_reference.sh



