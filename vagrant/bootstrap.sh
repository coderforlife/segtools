#!/usr/bin/env bash

echo "Installing base packages"
yum install -y cmake git epel-release wget tcsh xauth xclock mlocate time tree
yum install -y numpy numpy-f2py scipy
yum install -y gcc gcc-gfortran gcc-c++ python python-devel python-virtualenv atlas atlas-devel lapack lapack-devel lapack64 lapack64-develzlib zlib-devel libtiff libtiff-devel libjpeg-turbo libjpeg-turbo-devel hdf5 hdf5-devel fftw fftw-devel libpng libpng-devel libpng-static
yum install -y xorg-x11-fonts-*
yum install -y mesa-*
yum install -y python-pip python-wheel
yum install -y python-pillow python-pillow-devel
yum install -y R readline readline-devel python-singledispatch
updatedb
pip install wheel
pip install numpy
pip install cython
pip install scipy --upgrade
pip install pillow --upgrade
pip install psutil
pip install subprocess32
pip install h5py
pip install pyfftw
pip install rpy2

