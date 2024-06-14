#!/bin/bash

n=$(nproc)

sudo apt install -y git cmake make libcamera-dev libopencv-dev gcc g++

name="LCCV"
url="https://github.com/kbarni/LCCV"
echo "Cloning $url into $name"
git clone $url $name
cd $name
echo "Compiling $name"
mkdir build
cd build
cmake ..
make -j$n
echo "Installing $name"
sudo make install
cd ../..
rm -rf $name
echo "Done"
