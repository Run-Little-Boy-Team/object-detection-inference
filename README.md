# Run RLB object detection
Running [RLB object detection](https://github.com/Run-Little-Boy-Team/object-detection) model in Python or C++.

## Installation

### Python
Run [./python/install.sh](./python/install.sh).

Note: the Python version is missing some functionalities that the C++ version have.

### C++
Run [./c++/install.sh](./c++/install.sh).

Note: the green LED detection algorithm is based on [this](https://github.com/NareshBisht/OpenCV-Color-Detection).

### GPU support
Run the install script with the `gpu` argument.

## Usage

### Python
#### Run
```bash
python3 ./python/main.py <args>
```

### C++
#### Build
```bash
./c++/build.sh
```
#### Run
```bash
./c++/main <args>
```

### CLI arguments
Run the desired version of the program without any argument or with `--help` to print help.
#### Example
```bash
--source 0 --configuration ./config.yaml --model ./models/model.onnx
```
