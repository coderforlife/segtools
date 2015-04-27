This folder contains the various filters known to the image library. These are dynamically loaded
by the image library. To implement a new filter you need to create a new Python file and implement
the Command or CommandEasy class to expose it on the command line. You may also like to make use
of the FilteredImageStack, FilteredImageSlice, UnchangingFilteredImageStack, and
UnchangingFilteredImageSlice from _stack in this folder, however this not a requirement.