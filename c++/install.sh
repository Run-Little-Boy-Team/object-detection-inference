#!/bin/bash

version="1.18.0"

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

sudo apt install -y wget libopencv-dev

name="onnxruntime-linux-$arch$gpu-$version"
file="$name.tgz"
url="https://github.com/microsoft/onnxruntime/releases/download/v$version/$file"
echo "Downloading $url"
wget -q $url
echo "Extracting $file"
tar -xf $file
rm $file
echo "Installing $name"
cp -rf $name/lib/* /usr/local/lib/
cp -rf $name/include/* /usr/local/include/
rm -rf $name
echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib >> /home/$USER/.bashrc
source /home/$USER/.bashrc
name="LCCV"
url="https://github.com/kbarni/LCCV"
echo "Cloning $url into $name"
git clone $url $name
cd $name
echo "Compiling $name"
mkdir build
cd build
cmake ..
make -j4
echo "Installing $name"
sudo make install
cd ../..
rm -rf $name
echo "Done"