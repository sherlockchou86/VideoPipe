# third_party/cpp_analyzelib/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# 保护机制
if(DEFINED DOUBAO_ANALYZER_BUILT)
    message(STATUS "doubao_analyzer already built, skipping...")
    return()
endif()
set(DOUBAO_ANALYZER_BUILT TRUE CACHE INTERNAL "doubao_analyzer build flag")

project(doubao_analyzer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)

# 包含目录
include_directories(include)

# 库的源文件 - 确保所有必要的源文件都包含
set(LIB_SOURCES
    src/DoubaoMediaAnalyzer.cpp
    src/utils.cpp
    src/config.cpp
    # 如果有其他源文件，确保都添加在这里
)

# 检查源文件是否存在
foreach(src_file ${LIB_SOURCES})
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${src_file})
        message(WARNING "Source file ${src_file} does not exist")
    else()
        message(STATUS "Found source file: ${src_file}")
    endif()
endforeach()

# 可执行文件的源文件
set(APP_SOURCES
    src/main.cpp
)

# 创建共享库
add_library(doubao_analyzer_lib SHARED ${LIB_SOURCES})

# 设置库的属性
set_target_properties(doubao_analyzer_lib PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "doubao_analyzer"
    EXPORT_NAME doubao_analyzer
)

# 设置库的包含目录
target_include_directories(doubao_analyzer_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenSSL_INCLUDE_DIR}
)

# 链接库的依赖
target_link_libraries(doubao_analyzer_lib 
    ${OpenCV_LIBS}
    CURL::libcurl
    OpenSSL::SSL
    OpenSSL::Crypto
)

# 添加导出宏定义
target_compile_definitions(doubao_analyzer_lib PRIVATE DOUBAO_ANALYZER_EXPORTS)

# 创建可执行文件（可选）
add_executable(doubao_analyzer_app ${APP_SOURCES})
target_link_libraries(doubao_analyzer_app doubao_analyzer_lib)

# 设置C++标准
target_compile_features(doubao_analyzer_lib PRIVATE cxx_std_17)
target_compile_features(doubao_analyzer_app PRIVATE cxx_std_17)

# 创建别名以便其他项目使用
add_library(doubao::doubao_analyzer ALIAS doubao_analyzer_lib)

message(STATUS "doubao_analyzer_lib target created successfully")