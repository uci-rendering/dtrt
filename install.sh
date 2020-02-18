#!/bin/bash

sudo apt-get update && sudo apt-get -y upgrade

sudo apt-get install -y alien dpkg-dev debhelper build-essential libtbb-dev libglfw3-dev libpython-dev libzip-dev python3-pip libopenexr-dev pkg-config libeigen3-dev libssl-dev

pip3 install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip3 install OpenEXR scikit-image pytest

mkdir -p dependencies
cd dependencies

# Install Cmake (3.16.3)
wget https://github.com/Kitware/CMake/releases/download/v3.16.3/cmake-3.16.3.tar.gz
tar -zxvf cmake-3.16.3.tar.gz
cd cmake-3.16.3
./bootstrap
sudo make install -j
cd ..

git clone https://github.com/embree/embree.git
git clone https://github.com/pybind/pybind11.git

# Compile Embree
cd embree
mkdir -p build
cd build
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DEMBREE_ISPC_SUPPORT=OFF ..
sudo make install -j8

# Compile Pybind11
cd ../../pybind11
mkdir -p build
cd build
cmake ..
make check -j8
sudo make install -j8

echo '''
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
''' >> ~/.bashrc
