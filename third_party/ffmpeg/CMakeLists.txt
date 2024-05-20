cmake_minimum_required(VERSION 3.10)
project(ffmpeg VERSION 1.0)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE FFmpeg_LIBS
        /usr/local/lib/libav*.so
        /usr/local/lib/libsw*.so
        /usr/local/lib/libpostproc.so)

file(GLOB_RECURSE FFMPEG_CPPS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(${PROJECT_NAME} SHARED ${FFMPEG_CPPS})
target_link_libraries(${PROJECT_NAME} ${FFmpeg_LIBS} ${FFMPEG_SRC})