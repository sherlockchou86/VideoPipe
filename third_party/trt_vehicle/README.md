
## What `trt_vehicle` can do:
1. Vehicle detector
2. Vehicle type classifier
3. Vehicle color classifier
4. Vehicle plate detector and recognizer
5. Vehicle wheel detector
6. Vehicle feature encoder
7. Vehicle logo detector


## How to install tensorrt and cuda ?
Refer to NVIDIA official web

`CUDA 11.1 + TensorRT 7.2 for this repository (tested)`

## How to generate trt model from onnx ?
```shell
trtexec --onnx=./vehicle.onnx --saveEngine=vehicle.trt --buildOnly=true
```

## How to build trt_vehicle ?

1. `cd ./build`, and run `sh ./build.sh`.
2. It will generate a `.so` library named `libtrt_vehicle.so`.
3. Or include source code directly, no `.so` library needed.


## How to debug for trt_vehicle ?

Make sure you have build `trt_vehicle` correctly
1. Change model paths to your specific values in `./main/*.cpp`.
2. Select one of `./main/*.cpp` files and click `Run` button in vscode, choose one launch item at the top of window(`trt_vehicle`).
