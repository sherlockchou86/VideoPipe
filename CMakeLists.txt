#[[
    prepare in advance：
    1. OpenCV >= 4.6
    2. Gstreamer >= 1.4
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
    2. all samples saved to build/samples
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

# collect source files for videopipe
file(GLOB_RECURSE NODES "nodes/*.cpp")
file(GLOB_RECURSE OBJECTS "objects/*.cpp")
file(GLOB_RECURSE UTILS "utils/*.cpp")
#...#
list(APPEND VP_CPPS_SOURCES ${NODES} ${OBJECTS} ${UTILS})

# build for videopipe
add_library(${PROJECT_NAME} SHARED ${VP_CPPS_SOURCES})
target_link_libraries(${PROJECT_NAME} ${VP_DEPEND_LIBS})

# build for simple samples under 'sample/'
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/samples)  # save all exe to 'samples'
add_executable(1-1-1_sample "samples/1-1-1_sample.cpp")
target_link_libraries(1-1-1_sample ${PROJECT_NAME})

add_executable(1-1-N_sample "samples/1-1-N_sample.cpp")
target_link_libraries(1-1-N_sample ${PROJECT_NAME})

add_executable(1-N-N_sample "samples/1-N-N_sample.cpp")
target_link_libraries(1-N-N_sample ${PROJECT_NAME})

add_executable(ba_crossline_sample "samples/ba_crossline_sample.cpp")
target_link_libraries(ba_crossline_sample ${PROJECT_NAME})

add_executable(enet_seg_sample "samples/enet_seg_sample.cpp")
target_link_libraries(enet_seg_sample ${PROJECT_NAME})

add_executable(face_tracking_sample "samples/face_tracking_sample.cpp")
target_link_libraries(face_tracking_sample ${PROJECT_NAME})

add_executable(image_des_sample "samples/image_des_sample.cpp")
target_link_libraries(image_des_sample ${PROJECT_NAME})

add_executable(image_src_sample "samples/image_src_sample.cpp")
target_link_libraries(image_src_sample ${PROJECT_NAME})

add_executable(interaction_with_pipe_sample "samples/interaction_with_pipe_sample.cpp")
target_link_libraries(interaction_with_pipe_sample ${PROJECT_NAME})

add_executable(mask_rcnn_sample "samples/mask_rcnn_sample.cpp")
target_link_libraries(mask_rcnn_sample ${PROJECT_NAME})

add_executable(message_broker_sample "samples/message_broker_sample.cpp")
target_link_libraries(message_broker_sample ${PROJECT_NAME})

add_executable(multi_detectors_and_classifiers_sample "samples/multi_detectors_and_classifiers_sample.cpp")
target_link_libraries(multi_detectors_and_classifiers_sample ${PROJECT_NAME})

add_executable(N-1-N_sample "samples/N-1-N_sample.cpp")
target_link_libraries(N-1-N_sample ${PROJECT_NAME})

add_executable(N-N_sample "samples/N-N_sample.cpp")
target_link_libraries(N-N_sample ${PROJECT_NAME})

add_executable(openpose_sample "samples/openpose_sample.cpp")
target_link_libraries(openpose_sample ${PROJECT_NAME})

add_executable(record_sample "samples/record_sample.cpp")
target_link_libraries(record_sample ${PROJECT_NAME})

add_executable(rtsp_des_sample "samples/rtsp_des_sample.cpp")
target_link_libraries(rtsp_des_sample ${PROJECT_NAME})

add_executable(vp_logger_sample "samples/vp_logger_sample.cpp")
target_link_libraries(vp_logger_sample ${PROJECT_NAME})

# samples depend on PaddlePaddle
if(VP_WITH_PADDLE)
    add_executable(app_src_sample "samples/app_src_sample.cpp")
    target_link_libraries(app_src_sample ${PROJECT_NAME})
    
    add_executable(paddle_infer_sample "samples/paddle_infer_sample.cpp")
    target_link_libraries(paddle_infer_sample ${PROJECT_NAME})
endif()

# samples depend on TensorRT
if(VP_WITH_TRT)
    add_executable(ba_jam_sample "samples/ba_jam_sample.cpp")
    target_link_libraries(ba_jam_sample ${PROJECT_NAME})

    add_executable(ba_stop_sample "samples/ba_stop_sample.cpp")
    target_link_libraries(ba_stop_sample ${PROJECT_NAME})

    add_executable(body_scan_and_plate_detect_sample "samples/body_scan_and_plate_detect_sample.cpp")
    target_link_libraries(body_scan_and_plate_detect_sample ${PROJECT_NAME})

    add_executable(dynamic_pipeline_sample "samples/dynamic_pipeline_sample.cpp")
    target_link_libraries(dynamic_pipeline_sample ${PROJECT_NAME})

    add_executable(dynamic_pipeline_sample2 "samples/dynamic_pipeline_sample2.cpp")
    target_link_libraries(dynamic_pipeline_sample2 ${PROJECT_NAME})    

    add_executable(message_broker_sample2 "samples/message_broker_sample2.cpp")
    target_link_libraries(message_broker_sample2 ${PROJECT_NAME})

    add_executable(multi_trt_infer_nodes_sample "samples/multi_trt_infer_nodes_sample.cpp")
    target_link_libraries(multi_trt_infer_nodes_sample ${PROJECT_NAME})

    add_executable(plate_recognize_sample "samples/plate_recognize_sample.cpp")
    target_link_libraries(plate_recognize_sample ${PROJECT_NAME})

    add_executable(src_des_sample "samples/src_des_sample.cpp")
    target_link_libraries(src_des_sample ${PROJECT_NAME})

    add_executable(trt_infer_sample "samples/trt_infer_sample.cpp")
    target_link_libraries(trt_infer_sample ${PROJECT_NAME})

    add_executable(vehicle_body_scan_sample "samples/vehicle_body_scan_sample.cpp")
    target_link_libraries(vehicle_body_scan_sample ${PROJECT_NAME})

    add_executable(vehicle_cluster_based_on_classify_encoding_sample "samples/vehicle_cluster_based_on_classify_encoding_sample.cpp")
    target_link_libraries(vehicle_cluster_based_on_classify_encoding_sample ${PROJECT_NAME})

    add_executable(vehicle_tracking_sample "samples/vehicle_tracking_sample.cpp")
    target_link_libraries(vehicle_tracking_sample ${PROJECT_NAME})
endif()

# build for complex samples under 'sample/sub_dir'
if(VP_BUILD_COMPLEX_SAMPLES)
    # save exe to 'samples/face_recognize'
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/samples/face_recognize)
    add_executable(face_recognize_pipeline "samples/face_recognize/face_recognize_pipeline.cpp")
    target_link_libraries(face_recognize_pipeline ${PROJECT_NAME})

    if(VP_WITH_TRT) # samples depend on TensorRT   
        # save exe to 'samples/similiarity_search'
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/samples/similiarity_search)
        add_executable(face_encoding_pipeline "samples/similiarity_search/face_encoding_pipeline.cpp")
        target_link_libraries(face_encoding_pipeline ${PROJECT_NAME})
        add_executable(vehicle_encoding_pipeline "samples/similiarity_search/vehicle_encoding_pipeline.cpp")
        target_link_libraries(vehicle_encoding_pipeline ${PROJECT_NAME})

        # save exe to 'samples/vehicle_behaviour_analysis'
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/samples/vehicle_behaviour_analysis)
        add_executable(vehicle_ba_pipeline "samples/vehicle_behaviour_analysis/vehicle_ba_pipeline.cpp")
        target_link_libraries(vehicle_ba_pipeline ${PROJECT_NAME})

        # save exe to 'samples/vehicle_property_and_similiarity_search'
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/samples/vehicle_property_and_similiarity_search)
        add_executable(vehicle_encoding_classify_pipeline "samples/vehicle_property_and_similiarity_search/vehicle_encoding_classify_pipeline.cpp")
        target_link_libraries(vehicle_encoding_classify_pipeline ${PROJECT_NAME})
    endif()
endif()