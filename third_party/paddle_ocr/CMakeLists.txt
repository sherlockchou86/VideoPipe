cmake_minimum_required(VERSION 3.10)
project(paddle_ocr VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -w -fdiagnostics-color=always -pthread")
# save all libs to the same directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/libs)

# PaddlePaddle required  (please refer to 'env/install_paddle_inference.sh')
set(PADDLE_LIB_PATH "/usr/local/lib")      # change this line if possible
set(PADDLE_INC_PATH "/usr/local/include")  # change this line if possible
link_directories(${PADDLE_LIB_PATH})     
include_directories(${PADDLE_INC_PATH})
if(DEFINED VP_BUILD_FROM) # tell videopipe where is paddle
    set(VP_PADDLE_LIB_PATH ${PADDLE_LIB_PATH} PARENT_SCOPE)     
    set(VP_PADDLE_INC_PATH ${PADDLE_INC_PATH} PARENT_SCOPE) 
endif()

# OpenCV required
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

# collect dependent libs for paddle_ocr
set(PADDLE_LIBS paddle_inference gflags glog paddle2onnx onnxruntime dnnl iomp5)
set(PADDLE_OCR_DEPEND_LIBS ${OpenCV_LIBS} ${PADDLE_LIBS} stdc++fs)

# collect source files for trt_vehicle
file(GLOB_RECURSE SRCS "src/*.cpp")
#...#
list(APPEND PADDLE_OCR_CPPS ${SRCS})

# build for paddle_ocr
add_library(${PROJECT_NAME} SHARED ${PADDLE_OCR_CPPS})
target_link_libraries(${PROJECT_NAME} ${PADDLE_OCR_DEPEND_LIBS})

# build samples for paddle_ocr
if(NOT DEFINED VP_BUILD_FROM)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/samples)  # save all exe to 'samples'
    add_executable(pd_ocr "samples/pd_ocr.cpp")
    target_link_libraries(pd_ocr ${PROJECT_NAME})
endif()