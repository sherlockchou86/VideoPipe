编译说明
1. 安装OpenCV
首先需要安装OpenCV库：

Ubuntu/Debian:

bash
sudo apt-get update
sudo apt-get install libopencv-dev
Windows (使用vcpkg):

bash
vcpkg install opencv
macOS (使用Homebrew):

bash
brew install opencv
2. 编译命令
使用g++编译:

bash
g++ -std=c++11 video_extractor.cpp -o video_extractor `pkg-config --cflags --libs opencv4`
使用CMake (推荐):

创建 CMakeLists.txt 文件：

cmake
cmake_minimum_required(VERSION 3.10)
project(VideoFrameExtractor)

find_package(OpenCV REQUIRED)

add_executable(video_extractor video_extractor.cpp)
target_link_libraries(video_extractor ${OpenCV_LIBS})
target_compile_features(video_extractor PRIVATE cxx_std_11)
编译：

bash
mkdir build
cd build
cmake ..
make
使用方法
1. 提取所有帧
bash
./video_extractor input_video.mp4 ./output_frames
2. 按帧间隔提取
bash
# 每10帧提取一帧
./video_extractor input_video.mp4 ./output_frames -interval 10
3. 按时间间隔提取
bash
# 每2.5秒提取一帧
./video_extractor input_video.mp4 ./output_frames -time 2.5
代码特点
命名规则: 严格按照 0, 1, 2, 3... 的序列命名JPEG文件

格式支持: 输出标准的JPEG格式图片

灵活提取: 支持全帧提取、按帧间隔提取、按时间间隔提取

错误处理: 完善的错误检测和处理机制

信息显示: 显示视频信息和提取进度

输出示例
运行程序后，会在指定目录生成类似以下的文件：

text
output_frames/
├── 0.jpeg
├── 1.jpeg
├── 2.jpeg
├── 3.jpeg
└── ...
这个代码提供了完整的视频帧提取功能，可以根据您的具体需求进行调整和扩展。

whj00@navyai:~/VideoPipe/ExtractorVideo$ g++ -std=c++17 video_extractor.cpp -o video_extractor `pkg-config --cflags --libs opencv4`
video_extractor.cpp: In member function ‘bool VideoFrameExtractor::initialize(const string&, const string&)’:
video_extractor.cpp:33:30: error: ‘create_directories’ is not a member of ‘std::filesystem’
   33 |             std::filesystem::create_directories(outputDir);
中文分析处理
这个错误是因为编译器没有找到 std::filesystem 库。我来为您提供修复后的代码，包含多种解决方案。