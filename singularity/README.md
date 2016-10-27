[singularity]: http://singularity.lbl.gov/
[sudo]: https://www.sudo.ws/
# chm_singularity

Generates [Singularity][singularity] image for Segtools.


# Run requirements

* [Singularity 2.0,2.1][singularity]
* Linux

# Build requirements 

* [Singularity 2.0,2.1][singularity]
* Make
* Linux
* Bash
* [sudo][sudo] superuser access (required by [Singularity][singularity] to build images)

# To build

Run the following command to create the singularity image which will
be put under the **build/** directory

```Bash
make singularity
```

# To test

The image is built with a self test mode. To run the self test issue the following command after building:

```Bash
cd build/
./segtools-0.1.img --check
```

### Expected output from above command

```Bash
Module       Required  Installed
Python       v2.7      v2.7.5     ✓
numpy        v1.7      v1.7.1     ✓
scipy        v0.12     v0.12.1    ✓

Optional:
cython       v0.19     v0.25      ✓ (for some optimized libraries)
pillow       v2.0      v3.4.2     ✓ (for loading common image formats)
h5py         v2.0      v2.6.0     ✓ (for loading MATLAB v7.3 files)
psutil       v2.0      v4.4.2     ✓ (for tasks)
subprocess32 v3.2.6    v3.2.7     ✓ (for tasks on POSIX systems)
```

# To use
 
Simply run with same arguments as imstack in Segtools

