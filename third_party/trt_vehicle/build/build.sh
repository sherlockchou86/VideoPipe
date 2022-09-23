
# install tensorrt and cuda correctly first

# compile
g++ -c -g -fPIC ../models/*.cpp ../util/*.cpp

# link to shared lib
g++ -shared ./*.o \
-lopencv_core \
-lopencv_imgcodecs \
-lopencv_imgproc \
-lnvinfer \
-lnvinfer_plugin \
-lcudart \
-lcuda \
-lcublas \
-L/usr/local/cuda/lib64 \
-L/usr/local/TensorRT/lib \
-o libtrt_vehicle.so

# copy to system path
cp ./libtrt_vehicle.so /usr/local/lib/libtrt_vehicle.so
rm -f ./*.o
