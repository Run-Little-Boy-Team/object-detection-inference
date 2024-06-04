# Run RLB object detection
Running RLB object detection model in Python or C++.

Note: the Python version is missing some functionalities that the C++ version have.

## Installation

### Python
Run [./python/install.sh](./python/install.sh).

### C++
Run [./c++/install.sh](./c++/install.sh).

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
