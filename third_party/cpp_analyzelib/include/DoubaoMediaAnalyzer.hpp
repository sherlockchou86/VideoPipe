#pragma once

#include <string>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>

#ifdef _WIN32
    #ifdef DOUBAO_ANALYZER_EXPORTS
        #define DOUBAO_API __declspec(dllexport)
    #else
        #define DOUBAO_API __declspec(dllimport)
    #endif
#else
    #define DOUBAO_API __attribute__((visibility("default")))
#endif


struct AnalysisResult {
    bool success;
    std::string content;
    double response_time;
    nlohmann::json usage;
    nlohmann::json raw_response;
    std::string error;
    
    AnalysisResult() : success(false), response_time(0.0) {}
};

class DoubaoMediaAnalyzer {
private:
    std::string api_key_;
    std::string base_url_;
    
public:
    explicit DoubaoMediaAnalyzer(const std::string& api_key);
    
    // 连接测试
    bool test_connection();
    
    // 单张图片分析
    AnalysisResult analyze_single_image(const std::string& image_path, 
                                       const std::string& prompt,
                                       int max_tokens = 1500);
    
    // 单个视频分析
    AnalysisResult analyze_single_video(const std::string& video_path,
                                       const std::string& prompt,
                                       int max_tokens = 2000,
                                       int num_frames = 5);
    
    // 批量分析
    std::vector<AnalysisResult> batch_analyze(const std::string& media_folder,
                                             const std::string& prompt,
                                             int max_files = 5,
                                             const std::string& file_type = "all");
    
    // 标签提取
    std::vector<std::string> extract_tags(const std::string& content);

    
    ~DoubaoMediaAnalyzer();  // 添加这行    
    
private:
    // 内部方法
    std::vector<std::string> extract_video_frames(const std::string& video_path, int num_frames);
    AnalysisResult send_analysis_request(const nlohmann::json& payload, int timeout);
    AnalysisResult process_response(const std::string& response_text, double response_time);
    
    // HTTP请求
    std::string make_http_request(const std::string& url, 
                                 const std::string& method,
                                 const std::string& data,
                                 const std::vector<std::string>& headers,
                                 int timeout);
};
