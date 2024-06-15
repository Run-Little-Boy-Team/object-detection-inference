#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

$SCRIPT_DIR/install_lccv.sh
$SCRIPT_DIR/install_ncnn.sh
$SCRIPT_DIR/install_onnx.sh
