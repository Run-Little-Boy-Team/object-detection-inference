#!/bin/bash

cd ./run/c++ && cmake -B build -S . && make -C ./build && cd ../..