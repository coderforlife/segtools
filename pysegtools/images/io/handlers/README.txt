This folder contains the various file handler known to the image IO library. These can either be
stack-based/3D handlers in which case they need to extend the FileImageStack class or 2D image
handlers in which case they need to call to extend the FileImageSource class.

A single handler may handle multiple file formats (such as the PIL handler dealing with all
PIL-supported formats). Additionally multiple handlers may be able to deal with a single format in
which case the user can select which handler to use with the "handler" option.

The special image stack handler using a collection of 2D image sources is not included here.
