cmake_minimum_required(VERSION 3.10)
project(trt_vehicle VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -w -fdiagnostics-color=always -pthread")
# save all libs to the same directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/libs)

# TensorRT required
set(TRT_LIB_PATH "/usr/local/tensorRT/lib")      # change this line if possible
set(TRT_INC_PATH "/usr/local/tensorRT/include")  # change this line if possible
link_directories(${TRT_LIB_PATH})     
include_directories(${TRT_INC_PATH})
if(DEFINED VP_BUILD_FROM) # tell videopipe where is trt
    set(VP_TRT_LIB_PATH ${TRT_LIB_PATH} PARENT_SCOPE)     
    set(VP_TRT_INC_PATH ${TRT_INC_PATH} PARENT_SCOPE) 
endif()

# CUDA required
find_package(CUDA REQUIRED) 
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

# OpenCV required
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

# collect dependent libs for trt_vehicle
set(TRT_VEHICLE_DEPEND_LIBS ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvinfer stdc++fs)

# collect source files for trt_vehicle
file(GLOB_RECURSE MODELS "models/*.cpp")
file(GLOB_RECURSE UTIL "util/*.cpp")
#...#
list(APPEND TRT_VEHICLE_CPPS ${MODELS} ${UTIL})

# build for trt_vehicle
add_library(${PROJECT_NAME} SHARED ${TRT_VEHICLE_CPPS})
target_link_libraries(${PROJECT_NAME} ${TRT_VEHICLE_DEPEND_LIBS})

# build samples for trt_vehicle
if(NOT DEFINED VP_BUILD_FROM)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/samples)  # save all exe to 'samples'
    add_executable(vehicle_cluster "samples/vehicle_cluster.cpp")
    target_link_libraries(vehicle_cluster ${PROJECT_NAME})

    add_executable(vehicle_color_type "samples/vehicle_color_type.cpp")
    target_link_libraries(vehicle_color_type ${PROJECT_NAME})

    add_executable(vehicle_compare "samples/vehicle_compare.cpp")
    target_link_libraries(vehicle_compare ${PROJECT_NAME})

    add_executable(vehicle_plate "samples/vehicle_plate.cpp")
    target_link_libraries(vehicle_plate ${PROJECT_NAME})

    add_executable(vehicle_scan "samples/vehicle_scan.cpp")
    target_link_libraries(vehicle_scan ${PROJECT_NAME})

    add_executable(vehicle_search "samples/vehicle_search.cpp")
    target_link_libraries(vehicle_search ${PROJECT_NAME})

    add_executable(vehicle "samples/vehicle.cpp")
    target_link_libraries(vehicle ${PROJECT_NAME})
endif()
