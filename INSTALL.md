```
cd ~               # or wherever
mkdir imstack-env
cd imstack-env
virtualenv .
source bin/activate
git clone git@github.com:slash-segmentation/segtools.git
pip install -e segtools[OPT,MATLAB,PIL]
imstack --check
```

Where OPT requires fftw to be installed (for optimum performance), PIL requries pillow to be
installed, and MATLAB requires h5py to be installed.