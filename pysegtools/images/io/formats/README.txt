This folder contains the various file formats known to the image IO library. These can either be
stack-based/3D formats (in which case they need to extend the ImageStack class) or 2D image formats
in which case they need to call iminfo/imread/imsave.register(ext, function).

Some special formats are not included in this folder. For 3D the collection of 2D images stack is
not included and for 2D images everything supported by PIL is not included here.
