cmake_minimum_required(VERSION 3.10)
project(rtsp_server VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -w -fdiagnostics-color=always -pthread")

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
set(GST_DEPEND_LIBS ${GST_LIBRARIES} ${GSTAPP_LIBRARIES} ${GST_RTSP_LIBRARIES})


# build for rtsp_server
file(GLOB SRCS "run_rtsp_server.cpp")
add_executable(run_rtsp_server ${SRCS})
target_link_libraries(run_rtsp_server ${GST_DEPEND_LIBS})
