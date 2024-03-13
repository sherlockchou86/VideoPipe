#[[
    prepare in advance：
    1. OpenCV >= 4.6
    2. Gstreamer >= 1.14
    3. CUDA, TensorRT, PADDLE for optional

    build command:
    1. mkdir build && cd build
    2. cmake -DVP_WITH_CUDA=ON -DVP_WITH_TRT=ON -DVP_WITH_PADDLE=ON -DVP_BUILD_COMPLEX_SAMPLES=ON ..
    3. make -j8
    [note:]
    1. -D***=ON means *** enabled
    2. -D***=OFF means *** disabled(default)

    results:
    1. all libs saved to build/libs
    2. all samples saved to build/bin
]]

cmake_minimum_required(VERSION 3.10)
project(video_pipe VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -w -fdiagnostics-color=always -pthread")
# save all libs(including third_party's) to 'libs'
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/libs)

# optional for build, modify values when configure the project using 'cmake -DVP_WITH_CUDA=OFF ..'
option(VP_WITH_CUDA "prepared CUDA or not?" OFF)
option(VP_WITH_TRT "prepared TensorRT or not?" OFF)
option(VP_WITH_PADDLE "prepared PaddlePaddle or not?" OFF)
option(VP_BUILD_COMPLEX_SAMPLES "build complex samples or not? (maybe source code not provided)" OFF)

# OpenCV required
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

# Gstreamer required
include(FindPkgConfig)
pkg_check_modules(GST   REQUIRED gstreamer-1.0)
pkg_check_modules(GSTAPP REQUIRED gstreamer-app-1.0)
pkg_check_modules(GST_RTSP REQUIRED gstreamer-rtsp-server-1.0)
message(STATUS "GStreamer library status:")
message(STATUS "    version: ${GST_VERSION}")
message(STATUS "    libraries: ${GST_LIBRARIES} ${GSTAPP_LIBRARIES} ${GST_RTSP_LIBRARIES}")
message(STATUS "    include path: ${GST_INCLUDE_DIRS}")
include_directories(${GST_INCLUDE_DIRS})
set (GST_DEPEND_LIBS ${GST_LIBRARIES} ${GSTAPP_LIBRARIES} ${GST_RTSP_LIBRARIES})

# collect dependent libs for videopipe
list(APPEND VP_DEPEND_LIBS ${OpenCV_LIBS} ${GST_DEPEND_LIBS} stdc++fs)

# optional for CUDA
if(VP_WITH_CUDA) # CUDA enabled
    add_definitions(-DVP_WITH_CUDA)
endif()

# optional for TensorRT
if(VP_WITH_TRT)  # TensorRT enabled
    add_definitions(-DVP_WITH_TRT)
    set(VP_BUILD_FROM ON)  # set flag for sub project
    set(VP_TRT_LIB_PATH "")
    set(VP_TRT_INC_PATH "")
    message("-------------start build trt_vehicle--------------")
    add_subdirectory(third_party/trt_vehicle) # build trt_vehicle
    list(APPEND VP_DEPEND_LIBS trt_vehicle)
    link_directories(${VP_TRT_LIB_PATH})
    message(STATUS "TensorRT library status:")
    message(STATUS "    include path: ${VP_TRT_INC_PATH}")
    message(STATUS "    library path: ${VP_TRT_LIB_PATH}")
    message("--------------end build trt_vehicle---------------")
endif()

# optional for PaddlePaddle
if(VP_WITH_PADDLE)  # PaddlePaddle enabled
    add_definitions(-DVP_WITH_PADDLE)
    set(VP_BUILD_FROM ON)  # set flag for sub project
    set(VP_PADDLE_LIB_PATH "")
    set(VP_PADDLE_INC_PATH "")
    message("-------------start build paddle_ocr--------------")
    add_subdirectory(third_party/paddle_ocr) # build paddle_ocr
    list(APPEND VP_DEPEND_LIBS paddle_ocr)
    link_directories(${VP_PADDLE_LIB_PATH})
    message(STATUS "PaddlePaddle library status:")
    message(STATUS "    include path: ${VP_PADDLE_INC_PATH}")
    message(STATUS "    library path: ${VP_PADDLE_LIB_PATH}")
    message("--------------end build paddle_ocr---------------")
endif()

# tinyexpr
message("-------------start build tinyexpr--------------")
add_subdirectory(third_party/tinyexpr)
list(APPEND VP_DEPEND_LIBS tinyexpr)
message("--------------end build tinyexpr---------------")

# collect source files for videopipe
file(GLOB_RECURSE NODES "nodes/*.cpp")
file(GLOB_RECURSE OBJECTS "objects/*.cpp")
file(GLOB_RECURSE UTILS "utils/*.cpp")
#...#
list(APPEND VP_CPPS_SOURCES ${NODES} ${OBJECTS} ${UTILS})

# build for videopipe
add_library(${PROJECT_NAME} SHARED ${VP_CPPS_SOURCES})
target_link_libraries(${PROJECT_NAME} ${VP_DEPEND_LIBS})

# build for samples
add_subdirectory(samples)