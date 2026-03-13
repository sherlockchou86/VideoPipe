#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>  // 用于mkdir
#include <errno.h>     // 用于错误处理

class VideoFrameExtractor {
private:
    cv::VideoCapture videoCapture;
    std::string outputDir;
    int frameCount;
    
public:
    VideoFrameExtractor() : frameCount(0) {}
    
    // 创建目录的辅助函数
    bool createDirectory(const std::string& path) {
        // 方法1: 使用系统调用
        int result = system(("mkdir -p \"" + path + "\"").c_str());
        if (result == 0) {
            std::cout << "创建目录: " << path << std::endl;
            return true;
        }
        
        // 方法2: 使用mkdir (更安全的方式)
        std::string command = "mkdir -p \"" + path + "\"";
        if (system(command.c_str()) == 0) {
            return true;
        }
        
        std::cerr << "警告: 无法创建目录: " << path << std::endl;
        std::cerr << "请手动创建目录或检查权限" << std::endl;
        return false;
    }
    
    // 检查目录是否存在
    bool directoryExists(const std::string& path) {
        struct stat info;
        if (stat(path.c_str(), &info) != 0) {
            return false;  // 无法访问
        }
        return (info.st_mode & S_IFDIR) != 0;  // 是目录
    }
    
    // 初始化视频文件
    bool initialize(const std::string& videoPath, const std::string& outputDirectory) {
        // 打开视频文件
        videoCapture.open(videoPath);
        if (!videoCapture.isOpened()) {
            std::cerr << "错误: 无法打开视频文件: " << videoPath << std::endl;
            return false;
        }
        
        // 设置输出目录
        outputDir = outputDirectory;
        if (outputDir.back() != '/') {
            outputDir += '/';
        }
        
        // 创建输出目录
        if (!directoryExists(outputDir)) {
            std::cout << "目录不存在，尝试创建: " << outputDir << std::endl;
            if (!createDirectory(outputDir)) {
                std::cerr << "错误: 无法创建输出目录" << std::endl;
                return false;
            }
        } else {
            std::cout << "使用现有目录: " << outputDir << std::endl;
        }
        
        // 显示视频信息
        double fps = videoCapture.get(cv::CAP_PROP_FPS);
        double totalFrames = videoCapture.get(cv::CAP_PROP_FRAME_COUNT);
        double width = videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
        double height = videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
        double duration = totalFrames / fps;
        
        std::cout << "=== 视频信息 ===" << std::endl;
        std::cout << " - 文件路径: " << videoPath << std::endl;
        std::cout << " - 帧率: " << fps << " fps" << std::endl;
        std::cout << " - 总帧数: " << totalFrames << " 帧" << std::endl;
        std::cout << " - 分辨率: " << width << " x " << height << std::endl;
        std::cout << " - 时长: " << std::fixed << std::setprecision(2) << duration << " 秒" << std::endl;
        std::cout << " - 输出目录: " << outputDir << std::endl;
        std::cout << "=================" << std::endl;
        
        return true;
    }
    
    // 生成文件名（0,1,2,3命名规则）
    std::string generateFilename(int frameNumber) {
        std::stringstream filename;
        filename << outputDir << frameNumber << ".jpg";
        return filename.str();
    }
    
    // 提取所有帧
    bool extractAllFrames() {
        cv::Mat frame;
        frameCount = 0;
        
        std::cout << "开始提取所有帧..." << std::endl;
        
        // 获取总帧数用于进度显示
        double totalFrames = videoCapture.get(cv::CAP_PROP_FRAME_COUNT);
        
        while (true) {
            // 读取下一帧
            videoCapture >> frame;
            
            // 如果帧为空，说明视频结束
            if (frame.empty()) {
                break;
            }
            
            // 生成文件名并保存
            std::string filename = generateFilename(frameCount);
            std::vector<int> compression_params;
            compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
            compression_params.push_back(95); // JPEG质量参数 (0-100)
            
            bool success = cv::imwrite(filename, frame, compression_params);
            
            if (success) {
                // 显示进度
                double progress = (frameCount + 1) / totalFrames * 100;
                std::cout << "\r进度: " << std::fixed << std::setprecision(1) << progress 
                         << "% - 已提取 " << frameCount + 1 << " 帧" << std::flush;
            } else {
                std::cerr << "\n错误: 无法保存帧: " << filename << std::endl;
                return false;
            }
            
            frameCount++;
        }
        
        std::cout << "\n提取完成! 共提取 " << frameCount << " 帧" << std::endl;
        return true;
    }
    
    // 按间隔提取帧
    bool extractFramesByInterval(int interval) {
        cv::Mat frame;
        frameCount = 0;
        int currentFrame = 0;
        
        if (interval <= 0) {
            std::cerr << "错误: 间隔必须大于0" << std::endl;
            return false;
        }
        
        std::cout << "按间隔 " << interval << " 帧提取..." << std::endl;
        
        double totalFrames = videoCapture.get(cv::CAP_PROP_FRAME_COUNT);
        
        while (true) {
            // 读取下一帧
            videoCapture >> frame;
            
            if (frame.empty()) {
                break;
            }
            
            // 如果当前帧符合间隔要求
            if (currentFrame % interval == 0) {
                std::string filename = generateFilename(frameCount);
                std::vector<int> compression_params;
                compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
                compression_params.push_back(95);
                
                bool success = cv::imwrite(filename, frame, compression_params);
                
                if (success) {
                    double progress = (currentFrame + 1) / totalFrames * 100;
                    std::cout << "\r进度: " << std::fixed << std::setprecision(1) << progress 
                             << "% - 已提取 " << frameCount + 1 << " 帧" << std::flush;
                } else {
                    std::cerr << "\n错误: 无法保存帧: " << filename << std::endl;
                    return false;
                }
                
                frameCount++;
            }
            
            currentFrame++;
        }
        
        std::cout << "\n提取完成! 共提取 " << frameCount << " 帧" << std::endl;
        return true;
    }
    
    // 按时间间隔提取帧
    bool extractFramesByTimeInterval(double timeInterval) {
        double fps = videoCapture.get(cv::CAP_PROP_FPS);
        int frameInterval = static_cast<int>(fps * timeInterval);
        
        if (frameInterval < 1) frameInterval = 1;
        
        std::cout << "按时间间隔 " << timeInterval << " 秒提取 (约 " << frameInterval << " 帧)" << std::endl;
        
        return extractFramesByInterval(frameInterval);
    }
    
    // 获取提取的帧数
    int getExtractedFrameCount() const {
        return frameCount;
    }
    
    ~VideoFrameExtractor() {
        if (videoCapture.isOpened()) {
            videoCapture.release();
            std::cout << "视频资源已释放" << std::endl;
        }
    }
};

// 显示帮助信息
void showHelp(const std::string& programName) {
    std::cout << "视频帧提取工具" << std::endl;
    std::cout << "用法: " << programName << " <视频文件路径> <输出目录> [选项]" << std::endl;
    std::cout << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  -all          提取所有帧 (默认)" << std::endl;
    std::cout << "  -interval N   每N帧提取一帧" << std::endl;
    std::cout << "  -time T       每T秒提取一帧" << std::endl;
    std::cout << "  -help         显示此帮助信息" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  " << programName << " input.mp4 ./output" << std::endl;
    std::cout << "  " << programName << " input.mp4 ./output -interval 10" << std::endl;
    std::cout << "  " << programName << " input.mp4 ./output -time 2.5" << std::endl;
    std::cout << "  " << programName << " /path/to/video.mp4 /path/to/output -all" << std::endl;
}

// 使用示例
int main(int argc, char* argv[]) {
    // 检查参数
    if (argc < 3) {
        showHelp(argv[0]);
        return 1;
    }
    
    std::string videoPath = argv[1];
    std::string outputDir = argv[2];
    
    // 检查帮助参数
    if (std::string(argv[1]) == "-help" || std::string(argv[1]) == "--help") {
        showHelp(argv[0]);
        return 0;
    }
    
    VideoFrameExtractor extractor;
    
    // 初始化视频
    if (!extractor.initialize(videoPath, outputDir)) {
        return 1;
    }
    
    // 根据参数选择提取模式
    if (argc == 3) {
        // 默认提取所有帧
        return extractor.extractAllFrames() ? 0 : 1;
    } else if (argc >= 4) {
        std::string option = argv[3];
        
        if (option == "-interval" && argc == 5) {
            try {
                int interval = std::stoi(argv[4]);
                return extractor.extractFramesByInterval(interval) ? 0 : 1;
            } catch (const std::exception& e) {
                std::cerr << "错误: 无效的间隔参数" << std::endl;
                return 1;
            }
        } else if (option == "-time" && argc == 5) {
            try {
                double timeInterval = std::stod(argv[4]);
                return extractor.extractFramesByTimeInterval(timeInterval) ? 0 : 1;
            } catch (const std::exception& e) {
                std::cerr << "错误: 无效的时间参数" << std::endl;
                return 1;
            }
        } else if (option == "-all") {
            return extractor.extractAllFrames() ? 0 : 1;
        } else if (option == "-help") {
            showHelp(argv[0]);
            return 0;
        } else {
            std::cerr << "错误: 无效的参数" << std::endl;
            showHelp(argv[0]);
            return 1;
        }
    }
    
    return 0;
}