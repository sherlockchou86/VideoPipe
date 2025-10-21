#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <regex>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

namespace config {
    // API配置
    extern const std::string BASE_URL;
    extern const std::string MODEL_NAME;
    
    // 默认值
    extern const int DEFAULT_MAX_TOKENS;
    extern const int DEFAULT_VIDEO_FRAMES;
    extern const int DEFAULT_MAX_FILES;
    extern const double DEFAULT_TEMPERATURE;
    
    // 超时设置（秒）
    extern const int CONNECTION_TIMEOUT;
    extern const int IMAGE_ANALYSIS_TIMEOUT;
    extern const int VIDEO_ANALYSIS_TIMEOUT;
    
    // 文件扩展名
    extern const std::vector<std::string> IMAGE_EXTENSIONS;
    extern const std::vector<std::string> VIDEO_EXTENSIONS;
}

namespace utils {
    // 字符串工具
    std::string to_lower(const std::string& str);
    std::vector<std::string> split(const std::string& str, char delimiter);
    std::string trim(const std::string& str);
    std::string trim(const std::string& str, const std::string& chars_to_trim);
    bool starts_with(const std::string& str, const std::string& prefix);
    bool ends_with(const std::string& str, const std::string& suffix);
    
    // 文件工具
    bool file_exists(const std::string& path);
    std::string get_file_extension(const std::string& path);
    bool is_image_file(const std::string& path);
    bool is_video_file(const std::string& path);
    std::vector<std::string> find_media_files(const std::string& folder, 
                                             const std::string& file_type = "all",
                                             int max_files = 5);
    
    // Base64编码
    std::string base64_encode(const std::vector<unsigned char>& data);
    std::string base64_encode_file(const std::string& file_path);
    
    // 图像处理
    std::vector<unsigned char> encode_image_to_jpeg(const cv::Mat& image, int quality = 85);
    cv::Mat resize_image(const cv::Mat& image, int max_size = 800);
    
    // JSON工具
    nlohmann::json parse_json(const std::string& json_str);
    std::string json_to_string(const nlohmann::json& j);
    
    // 标签提取
    std::vector<std::string> extract_tags(const std::string& content);
    
    // 时间工具
    double get_current_time();
    void sleep_seconds(int seconds);
}
