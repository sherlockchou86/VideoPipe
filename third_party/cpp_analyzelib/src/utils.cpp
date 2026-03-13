#include "../include/utils.hpp"
#include "../include/config.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <curl/curl.h>

namespace utils {

// 字符串工具
std::string to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    
    return tokens;
}

std::string trim(const std::string& str) {
    return trim(str, " \t\n\r");
}

std::string trim(const std::string& str, const std::string& chars_to_trim) {
    size_t start = str.find_first_not_of(chars_to_trim);
    if (start == std::string::npos) return "";
    
    size_t end = str.find_last_not_of(chars_to_trim);
    return str.substr(start, end - start + 1);
}

bool starts_with(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() && 
           str.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && 
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// 文件工具
bool file_exists(const std::string& path) {
    return std::filesystem::exists(path);
}

std::string get_file_extension(const std::string& path) {
    std::filesystem::path p(path);
    return p.extension().string();
}

bool is_image_file(const std::string& path) {
    std::string ext = to_lower(get_file_extension(path));
    return std::find(config::IMAGE_EXTENSIONS.begin(), 
                    config::IMAGE_EXTENSIONS.end(), ext) != config::IMAGE_EXTENSIONS.end();
}

bool is_video_file(const std::string& path) {
    std::string ext = to_lower(get_file_extension(path));
    return std::find(config::VIDEO_EXTENSIONS.begin(), 
                    config::VIDEO_EXTENSIONS.end(), ext) != config::VIDEO_EXTENSIONS.end();
}

std::vector<std::string> find_media_files(const std::string& folder, 
                                         const std::string& file_type,
                                         int max_files) {
    std::vector<std::string> files;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(folder)) {
            if (files.size() >= max_files) break;
            
            if (entry.is_regular_file()) {
                std::string path = entry.path().string();
                
                if (file_type == "all") {
                    if (is_image_file(path) || is_video_file(path)) {
                        files.push_back(path);
                    }
                } else if (file_type == "image" && is_image_file(path)) {
                    files.push_back(path);
                } else if (file_type == "video" && is_video_file(path)) {
                    files.push_back(path);
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error accessing folder: " << e.what() << std::endl;
    }
    
    return files;
}

// Base64编码
std::string base64_encode(const std::vector<unsigned char>& data) {
    static const std::string base64_chars = 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";
    
    std::string encoded;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];
    
    for (const auto& byte : data) {
        char_array_3[i++] = byte;
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            
            for(i = 0; i < 4; i++) {
                encoded += base64_chars[char_array_4[i]];
            }
            i = 0;
        }
    }
    
    if (i > 0) {
        for(j = i; j < 3; j++) {
            char_array_3[j] = '\0';
        }
        
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;
        
        for (j = 0; j < i + 1; j++) {
            encoded += base64_chars[char_array_4[j]];
        }
        
        while(i++ < 3) {
            encoded += '=';
        }
    }
    
    return encoded;
}

std::string base64_encode_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }
    
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file), {});
    return base64_encode(buffer);
}

// 图像处理
std::vector<unsigned char> encode_image_to_jpeg(const cv::Mat& image, int quality) {
    std::vector<unsigned char> buffer;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
    cv::imencode(".jpg", image, buffer, params);
    return buffer;
}

cv::Mat resize_image(const cv::Mat& image, int max_size) {
    int height = image.rows;
    int width = image.cols;
    
    if (std::max(height, width) <= max_size) {
        return image.clone();
    }
    
    double scale = static_cast<double>(max_size) / std::max(height, width);
    int new_width = static_cast<int>(width * scale);
    int new_height = static_cast<int>(height * scale);
    
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height));
    return resized;
}

// JSON工具
nlohmann::json parse_json(const std::string& json_str) {
    return nlohmann::json::parse(json_str);
}

std::string json_to_string(const nlohmann::json& j) {
    return j.dump();
}

// 标签提取
std::vector<std::string> extract_tags(const std::string& content) {
    std::vector<std::string> tags;
    
    try {
        // 查找数组格式 ['tag1', 'tag2']
        size_t start = content.find("['");
        size_t end = content.find("']");
        
        if (start != std::string::npos && end != std::string::npos && start < end) {
            std::string tags_str = content.substr(start + 2, end - start - 2);
            auto temp_tags = split(tags_str, ',');
            
            for (const auto& tag : temp_tags) {
                std::string clean_tag = trim(tag);
                clean_tag = trim(clean_tag, "'\"");
                if (!clean_tag.empty()) {
                    tags.push_back(clean_tag);
                }
            }
            
            if (!tags.empty()) return tags;
        }
        
        // 正则表达式匹配其他格式
        std::regex pattern1(R"(标签[：:]\s*([^。，！？!?]+))");
        std::regex pattern2(R"(['"]([^'"]+)['"])");
        std::regex pattern3(R"(([^,，、]+?)(?=,|，|、|$))");
        
        std::smatch matches;
        
        if (std::regex_search(content, matches, pattern1) && matches.size() > 1) {
            auto temp_tags = split(matches[1].str(), ',');
            for (const auto& tag : temp_tags) {
                std::string clean_tag = trim(tag);
                if (!clean_tag.empty()) {
                    tags.push_back(clean_tag);
                }
            }
        }
        
        // 去重并限制数量
        std::sort(tags.begin(), tags.end());
        tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
        
        if (tags.size() > 5) {
            tags.resize(5);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error extracting tags: " << e.what() << std::endl;
    }
    
    return tags;
}

// 时间工具
double get_current_time() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

void sleep_seconds(int seconds) {
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

// 或者更兼容的版本
std::string get_filename(const std::string& filepath) {
    size_t last_slash = filepath.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        return filepath.substr(last_slash + 1);
    }
    return filepath;
}

} // namespace utils
