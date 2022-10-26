
## trt_vehicle can do
1. vehicle detector
2. vehicle type classifier
3. vehicle color classifier
4. vehicle plate detector and recognizer
5. vehicle wheel detector
6. vehicle feature encoder
7. vehicle logo detector


## How to install tensorrt and cuda ?
refer to NVIDIA official web

`cuda 11.1  tensorrt 7.2 for this repository (tested)`

## How to generate trt model from onnx ?
```shell
trtexec --onnx=./vehicle.onnx --saveEngine=vehicle.trt --buildOnly=true
```

## How to build trt_vehicle ?

1. `cd ./build`, and run `sh ./build.sh`.
2. it will generate a `.so` library named `libtrt_vehicle.so`.
3. or include source code directly, no `.so` library needed.


## How to debug for trt_vehicle ?

make sure you have build trt_vehicle correctly
1. change model paths to your specific values in `./main/*.cpp`.
2. select one of `./main/*.cpp` files and click `Run` button in vscode, choose one launch item at the top of window(`trt_vehicle`).
