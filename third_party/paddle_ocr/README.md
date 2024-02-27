
This code is derived from(modified some codes) `paddle_ocr` project: https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/deploy/cpp_infer 

## how to install paddle_inference sdk (by downloading tar package from [official web](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#linux))?

1. Choose the right tar file according to your version of CUDA and GPUs
2. Put the tar file at anywhere and unzip it
3. Run `tools/install_paddle_inference.sh`, it will help you to install it automatically, **please modify the root path of paddle_inference at the first line in script**.

> CUDA 11.1 TensorRT8.5 for this repo (tested).

## how to build paddle_ocr?

we can build paddle_ocr separately.

0. set the right library path and include path for PaddlePaddle in `CMakeLists.txt`
1. `mkdir build && cd build`
2. `cmake ..`
3. `make -j8`

all lib files saved to `build/libs`, all samples saved to `build/samples`. please refer to videopipe about how to run samples for paddle_ocr.

## Sample screenshot ##
### text recognize
![](../../doc/3rdparty/8.png)