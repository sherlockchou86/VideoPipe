base64 library from github: https://github.com/ReneNyffenegger/cpp-base64, merged .cpp and .h files together.

**NOTE**

used for encoding image to base64 string when interact with mLLM in VidepPipe.

test file `base64_test.cpp`, which convert cv::Mat to base64 string and convert back to cv::Mat
```
mkdir build
cd build
cmake ..
make -j8
./base64_test
```