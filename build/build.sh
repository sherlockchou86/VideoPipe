# For shell & vscode
# Build VideoPipe (via shell)

NODES=../nodes/*.cpp
INFERS_NODES=../nodes/infers/*.cpp
OSD_NODES=../nodes/osd/*.cpp
RECORD_NODES=../nodes/record/*.cpp
OBJECTS=../objects/*.cpp
ELEMENTS_OBJECTS=../objects/elements/*.cpp
SHAPES_OBJECTS=../objects/shapes/*.cpp
BA=../ba/*.cpp
UTILS_ANALYSIS_BOARD=../utils/analysis_board/*.cpp
LOGGER=../utils/logger/*cpp

# compile
g++ -c -g -fPIC \
$NODES $INFERS_NODES $OSD_NODES $OBJECTS $ELEMENTS_OBJECTS $SHAPES_OBJECTS $BA $UTILS_ANALYSIS_BOARD $LOGGER $RECORD_NODES


# create shared library
# build paddle ocr first
g++ -shared ./*.o \
-fdiagnostics-color=always \
-g \
-lopencv_core \
-lopencv_videoio \
-lopencv_highgui \
-lopencv_imgproc \
-lopencv_imgcodecs \
-lopencv_dnn \
-lopencv_video \
-lopencv_freetype \
-lpaddle_ocr \
-ltrt_vehicle \
-lpthread \
-std=c++17 \
-lstdc++fs \
-o ./libvp.so

# copy to system path
cp ./libvp.so /usr/local/lib/libvp.so
rm -f ./*.o