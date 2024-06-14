#!/bin/bash

n=$(nproc)

sudo apt install -y git cmake make build-essential gcc g++ libprotobuf-dev protobuf-compiler libomp-dev libvulkan-dev

name="ncnn"
url="https://github.com/Tencent/ncnn"
echo "Cloning $url into $name"
git clone $url $name
cd $name
git submodule update --init
echo "Compiling $name"
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=OFF ..
make -j$n
echo "Installing $name"
make install
sudo cp -rf ./install/* /usr/local/
cd ../..
rm -rf $name
echo "Done"

