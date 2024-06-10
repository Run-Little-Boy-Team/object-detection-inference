#!/bin/bash

version="1.18.0"

n=$(nproc)

arch=$(uname -m)
if [ "$arch" == "x86_64" ]; then
    arch="x64"
elif [ "$arch" == "aarch64" ]; then
    arch="aarch64"
else
    echo "Unsupported architecture: $arch"
    exit 1
fi
echo "Architecture: $arch"

if [ "$1" == "gpu" ] && [ "$arch" == "x64" ]; then
    gpu=-gpu
    echo "GPU support enabled"
    sudo apt install -y nvidia-cuda-toolkit nvidia-cudnn
else
    gpu=""
fi

sudo apt install -y cmake make wget libopencv-dev build-essential gcc g++ libprotobuf-dev protobuf-compiler libomp-dev libvulkan-dev

name="onnxruntime-linux-$arch$gpu-$version"
file="$name.tgz"
url="https://github.com/microsoft/onnxruntime/releases/download/v$version/$file"
echo "Downloading $url"
wget -q $url
echo "Extracting $file"
tar -xf $file
rm $file
echo "Installing $name"
sudo cp -rf $name/lib/* /usr/local/lib/
sudo cp -rf $name/include/* /usr/local/include/
rm -rf $name
echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib >> /home/$USER/.bashrc
source /home/$USER/.bashrc

name="ncnn"
url="https://github.com/Tencent/ncnn"
echo "Cloning $url into $name"
git clone $url $name
cd $name
git submodule update --init
echo "Compiling $name"
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_REQUANT=ON -DNCNN_BUILD_EXAMPLES=OFF ..
make -j$n
echo "Installing $name"
make install
sudo cp -rf ./install/* /usr/local/
cd ../..
rm -rf $name

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
