#!/bin/bash

version="1.18.0"

n=$(nproc)

sudo apt install -y wget

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
echo "Done"
