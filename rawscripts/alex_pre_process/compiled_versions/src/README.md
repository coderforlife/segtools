
In this directory are slightly modified versions of Alex's matlab scripts.
The modifications mainly involve name changes to the scripts and adjustment of
input and output parameters

Mapping of Alex's scripts in rawscripts to these scripts
========================================================

Below is a list of the scripts that were modified or renamed

rawscripts/generate_reference.m => gen_histogram.m
==================================================

In addition to the name change the input parameters were changed to two
parameters (image file, output histogram text file).  This eliminates the
assumptions in the old script with regard to image file names. 

NEW script merge_histograms.m
=============================

In the rawscripts directory, run_ehs.m would take a directory of histogram txt files
and generate a single histogram before processing the image.  This would be
done for every image.  This matlab script merges the histogram txt files into 
one file.

rawscripts/run_ehs.m => run_ehs.m
=================================

The run_ehs.m in this directory now takes (imput image, a combined histogram
text file and an output image path).  run_ehs.m no longer examines thousands
of histogram txt files to generate a combined histogram, the code now just
loads the single histogram.
