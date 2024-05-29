# Run RLB object detection
Running RLB object detection model in Python or C++.

## Installation

### Python
Run [./run/python](./run/python/install.sh).

### C++
Run [./run/c++/install.sh](./run/c++/install.sh).

### GPU support
Run the install script with the `gpu` argument.


## Usage

### Python
#### Run
```bash
python3 ./run/python/main.py <args>
```

### C++
#### Build
```bash
./run/c++/build.sh
```
#### Run
```bash
./run/c++/main <args>
```

### CLI arguments
Run the desired version of the program without any argument or with `--help` to print help.
#### Example
```bash
--source 0 --configuration ./config.yaml --model ./models/model.onnx
```
