
# compile
g++ -c -g -fPIC ../src/*.cpp

# create shared lib
g++ -shared ./*.o \
-lopencv_core \
-lopencv_imgproc \
-lopencv_imgcodecs \
-lopencv_highgui \
-lopencv_videoio \
-lopencv_freetype \
-lpaddle_inference \
-lgflags \
-lglog \
-lpaddle2onnx \
-lonnxruntime \
-ldnnl \
-liomp5 \
-lpthread \
-std=c++17 \
-o ./libpaddle_ocr.so

# copy to system path
cp ./libpaddle_ocr.so /usr/local/lib/libpaddle_ocr.so
rm -f ./*.o