cmake_minimum_required(VERSION 3.10)
project(llmlib_test VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -w -fdiagnostics-color=always -pthread")

# OpenCV required
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenSSL REQUIRED)
message(STATUS "OpenSSL library status:")
message(STATUS "    version: ${OPENSSL_VERSION}")
message(STATUS "    libraries: ${OPENSSL_LIBRARIES}")
message(STATUS "    include path: ${OPENSSL_INCLUDE_DIR}")
include_directories(${OPENSSL_INCLUDE_DIR})

add_executable(llmlib_ollama_test "llmlib_ollama_test.cpp")
target_link_libraries(llmlib_ollama_test ${OpenCV_LIBS} ${OPENSSL_LIBRARIES})

add_executable(llmlib_openai_test "llmlib_openai_test.cpp")
target_link_libraries(llmlib_openai_test ${OpenCV_LIBS} ${OPENSSL_LIBRARIES})

add_executable(llmlib_multi_chat_test "llmlib_multi_chat_test.cpp")
target_link_libraries(llmlib_multi_chat_test ${OpenCV_LIBS} ${OPENSSL_LIBRARIES})