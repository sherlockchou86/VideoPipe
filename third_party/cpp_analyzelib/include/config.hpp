#pragma once

#include <string>
#include <vector>

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
