segtools
===================
Contains tools to assist in segmentation of EM data


pysegtools
----------
Python segmentation tools. The main program is "imstack"


imstack
-------
This program works with stacks of images, converting, filtering, and manipulating them. The
interface is very flexible but a bit confusing. The command line is built with a series of commands
that each consume and/or produce image stacks. For example, the -L or --load command produces a
single image stack from a file and consumes nothing. More than one image stack can be in memory
at once and some commands even consume or produce multiple image stacks at once. The most
recently produced image stack is consumed first. The order of image stacks can be changed as well.
A single image stack does not need to be "homogeneous" (the same dimensions / pixel type) in every
slice, however some filters and file formats do require homogeneous stacks.

Image stack file formats supported:
 * PIL-supported stacks including TIFF, GIF, IM, and SPIDER [all readonly] 
   see http://pillow.readthedocs.org/en/latest/handbook/image-file-formats.html
 * MRC [homogeneous only]
 * MAT
 * (collections of 2D images)

2D image file formats supported:
 * PIL-supported formats including TIFF, GIF, PNG, and JPEG 
   see http://pillow.readthedocs.org/en/latest/handbook/image-file-formats.html
 * MAT
 * MHA/MHD

Filters supported:
 * Stacks: split, combine, select slices
 * Channels: extract, combine
 * Blur: Gaussian, median, mean
 * Flip: x/y/z
 * Rotate: cw/ccw/full [2D-only at the moment]
 * Inversion (black to white, white to black)
 * Thresholding: auto (Otsu's method), hysteresis, *multi
 * Edge Detection: Prewitt, Scharr, Sobel, Canny [2D-only]
 * Masks: calculate from solid regions near edges [2D-only]
 * Filling: solid, mean, mirror, reflect, nearest, wrap [2D-only]
 * Background Padding: cropping, adding (masks can be 'resolved' into rectangular regions in various ways)
 * Histogram: saving, equalization (standard and exact) (supports masks) [exact histogram equalization is 2D-only at the moment]
 * Labelling: labelling, re-labelling, re-numbering
 * Resize: binning (mean, median), bicubic interpolation
 * Pixel Type Conversions: scaling conversion, shrinking integers, byte order, raw
 * Complex: real, imaginary, complexify, FFT, IFFT [2D-only at the moment]

Features in the works:
 * Image stacks file formats: MHA/MHD, multi-TIFF, bioformats supported formats
 * 2D image file formats: bioformats supported formats
 * Filters: ansiotropic diffusion, CLAHE, grayscale, color space conversions, paletting,
   morphological filters (opening, closing, dilation, erosion, propogation, fill holes, hit-or-miss, tophat)
 * 3D Filters: rotation, exact histogram equalization, FFTs


rawscripts
----------
This directory contains scripts pulled from various directories. These are primarily for reference.
At this point all of these scripts are now built into the imstack program.
