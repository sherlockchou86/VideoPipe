#bin bash

#安装依赖
sudo apt-get install nasm -y
sudo apt-get install yasm -y
sudo apt install libx264-dev libx265-dev  \
         libfdk-aac-dev libmp3lame-dev libvorbis-dev

# 安装x264
git clone https://code.videolan.org/videolan/x264.git
cd x264 \
 && git checkout stable \
 && ./configure --enable-shared \
 && make -j10 \
 && make install \
 &&  cd ../


# 安装ffmpeg支持nvidia硬件加速
# 查看对应的nvidia驱动版本和显卡型号，然后去nvidia官网下载对应的nv-codec-headers，有的计算卡卡不带编解码器就不支持硬件加速
# 1、安装nv-codec-headers
# 2、安装ffmpeg
##  此处可能需要手动修改cuda动态库应用的路径

git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd ./nv-codec-headers \
   && make -j10 \
   && make install \
   && cd ../

git clone https://github.com/FFmpeg/FFmpeg.git
cd FFmpeg
git checkout n5.1.2
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
 && ./configure --enable-gpl --enable-libx264 \
                 --enable-nonfree --enable-shared \
                 --disable-static \
                 --enable-cuda --enable-cuvid --enable-nvenc \
                 --enable-libnpp --extra-cflags=-I/usr/local/cuda/include \
                 --extra-ldflags=-L/usr/local/cuda/lib64 \
 && make -j10 \
 && make install \
 && cd ../

# 安装依赖