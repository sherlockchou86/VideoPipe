# For shell & vscode
# Build VideoPipe (via shell)

NODES=../nodes/*.cpp
INFERS_NODES=../nodes/infers/*.cpp
OSD_NODES=../nodes/osd/*.cpp
RECORD_NODES=../nodes/record/*.cpp
TRACK_NODES=../nodes/track/*.cpp
TRACK_SORT=../nodes/track/sort/*.cpp
TRACK_DSORT=../nodes/track/deep_sort/*.cpp
BROKER_NODES=../nodes/broker/*.cpp
BA_NODES=../nodes/ba/*.cpp
OBJECTS=../objects/*.cpp
SHAPES_OBJECTS=../objects/shapes/*.cpp
BA_OBJECTS=../objects/ba/*.cpp
UTILS_ANALYSIS_BOARD=../utils/analysis_board/*.cpp
LOGGER=../utils/logger/*cpp

# compile
g++ -c -g -fPIC \
-std=c++17 \
`pkg-config --cflags --libs glib-2.0` \
$NODES \
$INFERS_NODES \
$OSD_NODES $OBJECTS \
$BA_OBJECTS \
$SHAPES_OBJECTS \
$BA_NODES \
$UTILS_ANALYSIS_BOARD \
$LOGGER \
$RECORD_NODES \
$TRACK_NODES \
$TRACK_SORT \
$BROKER_NODES


# create shared library
# build paddle_ocr & trt_vehilce first
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