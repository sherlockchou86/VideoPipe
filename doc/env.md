

## personal environment ##
---------
VS Code + Ubuntu 18.04 C++17  gcc 7.5

---------
apt-get install ffmpeg/gstreamer/other dependency.

install opencv 4 from GitHub, with ffmpeg/gstreamer ON (cuda optional).

opencv 4.6.0 cmake command:

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_CUDA=ON \
-D BUILD_opencv_cudacodec=OFF \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN=6.1 \
-D WITH_V4L=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=/windows2/zhzhi/opencv_contrib-4.6.0/modules \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D BUILD_EXAMPLES=OFF ..

---------
VcXsrv for screen display from WSL1 to local pc (or from remote machine to local desktop)
install: https://sourceforge.net/p/vcxsrv/wiki/Home/

export DISPLAY=local_ip:0.0 (or add to ~/.bashrc)

---------
maybe you need install a nginx as rtmp server for debug purpose. 

Also, maybe you need rtsp server to receive its stream for debug purpose.

## tips ##
use shared_ptr/make_shared in whole project, do not use new/delete.

the pipe is driven by stream data, if your app is not responding, maybe no stream input.