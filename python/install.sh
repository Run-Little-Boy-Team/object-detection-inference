#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

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

sudo apt install -y libcap-dev

conda create -y -n rlb_inference
conda activate rlb_inference
conda install -y python=3.10
pip install pyqt5 opengl
pip install numpy opencv-python pyyaml picamera2

if [ "$1" == "gpu" ] && [ "$arch" == "x64" ]; then
    echo "GPU support enabled"
    conda install -y cudatoolkit=11.8
    conda install -y cudnn=8.9
    pip install onnxruntime-gpu==$version
else
    pip install onnxruntime==$version
fi

echo "Done"