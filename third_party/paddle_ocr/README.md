
This code is derived from(modified some codes) `paddle_ocr` project: https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/deploy/cpp_infer 

## how to install paddle_inference sdk (by downloading tar package from [official web](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#linux))?

1. Choose the right tar file according to your version of CUDA and GPUs
2. Put the tar file at anywhere and unzip it
3. Run `build/install_paddle_inference.sh`, it will help you to install it automatically, **please modify the root path of paddle_inference at the first line in script**.

> CUDA 11.1  CUDNN 8.0.5 tensorrt7.2.1 for this repo.

## how to build paddle_ocr?

1. `cd ./build`, and run `sh ./build.sh`.
2. It will generate a `*.so` library named `libpaddle_ocr.so`.

## How to debug `paddle_ocr`?

Make sure you have build paddle_ocr correctly

1. Change model paths to your specific values in `./main/main.cpp`.
2. Select ./main/main.cpp file and click `Run` button in vscode, choose one launch item at the top of window(`paddle_ocr`).
