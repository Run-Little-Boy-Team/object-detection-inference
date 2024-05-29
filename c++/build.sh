#!/bin/bash
origin_dir=$(pwd)
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $script_dir
cmake -B build -S . && make -C ./build && cd ../..
cd $origin_dir