cmake_minimum_required(VERSION 2.6)

project(videopipe)

add_definitions(-DAPI_EXPORTS)
add_definitions(-O2 -pthread)

#cuda & opencv
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


# trt_vehicle
file(GLOB TRT_VEHICLE_SOURCES "third_party/trt_vehicle/**/*.cpp")
cuda_add_library(trt_vehicle SHARED ${TRT_VEHICLE_SOURCES})
target_link_libraries(trt_vehicle nvinfer cudart)

# paddle_ocr
file(GLOB PADDLE_OCR_SOURCES "third_party/paddle_ocr/**/*.cpp")
add_library(paddle_ocr SHARED ${PADDLE_OCR_SOURCES})
target_link_libraries(paddle_ocr paddle_inference paddle2onnx)


file(GLOB NODES "nodes/*.cpp")
file(GLOB INFERS_NODES "nodes/infers/*.cpp")
file(GLOB OSD_NODES "nodes/osd/*.cpp")
file(GLOB OBJECTS "objects/*.cpp")
file(GLOB ELEMENTS_OBJECTS "objects/elements/*.cpp")
file(GLOB SHAPES_OBJECTS "objects/shapes/*.cpp")
file(GLOB BA "ba/*.cpp")
file(GLOB UTILS_ANALYSIS_BOARD "utils/analysis_board/*.cpp")
file(GLOB LOGGER "utils/logger/*.cpp")


add_library(video_pipe SHARED ${NODES} ${INFERS_NODES} ${OSD_NODES} ${OBJECTS} ${ELEMENTS_OBJECTS} ${SHAPES_OBJECTS} ${BA} ${UTILS_ANALYSIS_BOARD} ${LOGGER})
target_link_libraries(video_pipe  ${OpenCV_LIBS} paddle_ocr trt_vehicle -lstdc++fs)


add_executable(1-1-1_sample sample/1-1-1_sample.cpp)
target_link_libraries(1-1-1_sample video_pipe  ${OpenCV_LIBS})

add_executable(1-1-N_sample sample/1-1-N_sample.cpp)
target_link_libraries(1-1-N_sample video_pipe  ${OpenCV_LIBS})

add_executable(1-N-N_sample sample/1-N-N_sample.cpp)
target_link_libraries(1-N-N_sample video_pipe  ${OpenCV_LIBS})

add_executable(N-1-N_sample sample/N-1-N_sample.cpp)
target_link_libraries(N-1-N_sample video_pipe  ${OpenCV_LIBS})

add_executable(N-N_sample sample/N-N_sample.cpp)
target_link_libraries(N-N_sample video_pipe  ${OpenCV_LIBS})

add_executable(paddle_infer_sample sample/paddle_infer_sample.cpp)
target_link_libraries(paddle_infer_sample video_pipe  ${OpenCV_LIBS})

add_executable(src_des_sample sample/src_des_sample.cpp)
target_link_libraries(src_des_sample video_pipe  ${OpenCV_LIBS})

add_executable(trt_infer_sample sample/trt_infer_sample.cpp)
target_link_libraries(trt_infer_sample video_pipe  ${OpenCV_LIBS})

add_executable(vp_logger_sample sample/vp_logger_sample.cpp)
target_link_libraries(vp_logger_sample video_pipe  ${OpenCV_LIBS})