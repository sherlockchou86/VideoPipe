cmake_minimum_required(VERSION 3.10)
project(doubao_analyzer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)
find_package(PkgConfig REQUIRED)

# 包含目录
include_directories(include)

# 源文件
set(SOURCES
    src/main.cpp
    src/DoubaoMediaAnalyzer.cpp
    src/utils.cpp
    src/config.cpp
)

# 创建可执行文件
add_executable(doubao_analyzer ${SOURCES})

# 链接库
target_link_libraries(doubao_analyzer 
    ${OpenCV_LIBS}
    CURL::libcurl
)

# 设置C++标准
target_compile_features(doubao_analyzer PRIVATE cxx_std_17)
