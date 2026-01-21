#include "../include/config.hpp"

namespace config
{
    // API配置
    const std::string BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions";
    const std::string MODEL_NAME = "doubao-1.5-vision-lite-250315";

    // 默认值
    const int DEFAULT_MAX_TOKENS = 1500;
    const int DEFAULT_VIDEO_FRAMES = 5;
    const int DEFAULT_MAX_FILES = 5;
    const double DEFAULT_TEMPERATURE = 0.1;

    // 超时设置（秒）
    const int CONNECTION_TIMEOUT = 10;
    const int IMAGE_ANALYSIS_TIMEOUT = 60;
    const int VIDEO_ANALYSIS_TIMEOUT = 120;

    // 文件扩展名
    const std::vector<std::string> IMAGE_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp",
        ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF", ".WEBP"};

    const std::vector<std::string> VIDEO_EXTENSIONS = {
        ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv",
        ".MP4", ".AVI", ".MOV", ".MKV", ".FLV", ".WMV"};
}
