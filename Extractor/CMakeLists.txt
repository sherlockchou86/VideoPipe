cmake_minimum_required(VERSION 3.10)
project(VideoFrameExtractor)

find_package(OpenCV REQUIRED)

add_executable(video_extractor video_extractor.cpp)
target_link_libraries(video_extractor ${OpenCV_LIBS})
target_compile_features(video_extractor PRIVATE cxx_std_11)