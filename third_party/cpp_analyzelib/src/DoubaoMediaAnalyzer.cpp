#include "../include/DoubaoMediaAnalyzer.hpp"
#include "../include/utils.hpp"
#include "../include/config.hpp"
#include <curl/curl.h>
#include <sstream>
#include <iostream>

    
// HTTP回调函数
static size_t write_callback(void* contents, size_t size, size_t nmemb, std::string* response) {
    size_t total_size = size * nmemb;
    response->append(static_cast<char*>(contents), total_size);
    return total_size;
}

DoubaoMediaAnalyzer::DoubaoMediaAnalyzer(const std::string& api_key) 
    : api_key_(api_key), base_url_(config::BASE_URL) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

DoubaoMediaAnalyzer::~DoubaoMediaAnalyzer() {
    curl_global_cleanup();
}

bool DoubaoMediaAnalyzer::test_connection() {
    try {
        nlohmann::json payload = {
            {"model", config::MODEL_NAME},
            {"messages", {
                {
                    {"role", "user"},
                    {"content", "请回复'连接测试成功'"}
                }
            }},
            {"max_tokens", 50}
        };
        
        auto result = send_analysis_request(payload, config::CONNECTION_TIMEOUT);
        
        if (result.success) {
            std::cout << "✅ 豆包API连接正常" << std::endl;
            return true;
        } else {
            std::cout << "❌ API连接失败: " << result.error << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "❌ 连接测试异常: " << e.what() << std::endl;
        return false;
    }
}

AnalysisResult DoubaoMediaAnalyzer::analyze_single_image(const std::string& image_path, 
                                                        const std::string& prompt,
                                                        int max_tokens) {
    AnalysisResult result;
    
    try {
        if (!utils::file_exists(image_path)) {
            result.success = false;
            result.error = "图片文件不存在: " + image_path;
            return result;
        }
        
        std::string image_data = utils::base64_encode_file(image_path);
        
        nlohmann::json payload = {
            {"model", config::MODEL_NAME},
            {"messages", {
                {
                    {"role", "user"},
                    {"content", {
                        {
                            {"type", "image_url"},
                            {"image_url", {
                                {"url", "data:image/jpeg;base64," + image_data}
                            }}
                        },
                        {
                            {"type", "text"},
                            {"text", prompt}
                        }
                    }}
                }
            }},
            {"max_tokens", max_tokens},
            {"temperature", config::DEFAULT_TEMPERATURE},
            {"stream", false}
        };
        
        double start_time = utils::get_current_time();
        result = send_analysis_request(payload, config::IMAGE_ANALYSIS_TIMEOUT);
        result.response_time = utils::get_current_time() - start_time;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error = "分析异常: " + std::string(e.what());
    }
    
    return result;
}

AnalysisResult DoubaoMediaAnalyzer::analyze_single_video(const std::string& video_path,
                                                        const std::string& prompt,
                                                        int max_tokens,
                                                        int num_frames) {
    AnalysisResult result;
    
    try {
        if (!utils::file_exists(video_path)) {
            result.success = false;
            result.error = "视频文件不存在: " + video_path;
            return result;
        }
        
        std::cout << "🎬 正在提取视频关键帧..." << std::endl;
        auto frames_base64 = extract_video_frames(video_path, num_frames);
        
        if (frames_base64.empty()) {
            result.success = false;
            result.error = "无法从视频中提取有效帧";
            return result;
        }
        
        std::cout << "✅ 成功提取 " << frames_base64.size() << " 个关键帧" << std::endl;
        
        // 构建多图消息
        nlohmann::json content = nlohmann::json::array();
        content.push_back({{"type", "text"}, {"text", prompt}});
        
        for (size_t i = 0; i < frames_base64.size(); ++i) {
            content.push_back({
                {"type", "image_url"},
                {"image_url", {
                    {"url", "data:image/jpeg;base64," + frames_base64[i]},
                    {"detail", "low"}
                }}
            });
            
            content.push_back({
                {"type", "text"},
                {"text", "这是视频的第" + std::to_string(i+1) + "个关键帧"}
            });
        }
        
        nlohmann::json payload = {
            {"model", config::MODEL_NAME},
            {"messages", {
                {
                    {"role", "user"},
                    {"content", content}
                }
            }},
            {"max_tokens", max_tokens},
            {"temperature", config::DEFAULT_TEMPERATURE},
            {"stream", false}
        };
        
        double start_time = utils::get_current_time();
        result = send_analysis_request(payload, config::VIDEO_ANALYSIS_TIMEOUT);
        result.response_time = utils::get_current_time() - start_time;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error = "视频分析异常: " + std::string(e.what());
    }
    
    return result;
}

std::vector<AnalysisResult> DoubaoMediaAnalyzer::batch_analyze(const std::string& media_folder,
                                                              const std::string& prompt,
                                                              int max_files,
                                                              const std::string& file_type) {
    std::vector<AnalysisResult> results;
    
    auto media_files = utils::find_media_files(media_folder, file_type, max_files);
    
    if (media_files.empty()) {
        std::cout << "❌ 在 " << media_folder << " 中未找到媒体文件" << std::endl;
        return results;
    }
    
    std::cout << "📁 找到 " << media_files.size() << " 个媒体文件进行批量分析" << std::endl;
    
    for (size_t i = 0; i < media_files.size(); ++i) {
        const auto& media_path = media_files[i];
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "📊 分析第 " << i+1 << "/" << media_files.size() 
                  << " 个文件: " << std::filesystem::path(media_path).filename().string() << std::endl;
        
        try {
            auto file_size = std::filesystem::file_size(media_path);
            std::cout << "📏 文件大小: " << file_size << " 字节" << std::endl;
        } catch (...) {
            std::cout << "⚠️  无法读取文件大小信息" << std::endl;
        }
        
        AnalysisResult result;
        bool is_video = utils::is_video_file(media_path);
        
        if (is_video) {
            std::cout << "🎬 检测到视频文件" << std::endl;
            result = analyze_single_video(media_path, prompt);
        } else {
            std::cout << "🖼️  检测到图片文件" << std::endl;
            
            // 显示图片信息
            try {
                cv::Mat img = cv::imread(media_path);
                if (!img.empty()) {
                    std::cout << "🖼️  图片尺寸: " << img.cols << "x" << img.rows << std::endl;
                } else {
                    std::cout << "⚠️  无法读取图片尺寸信息" << std::endl;
                }
            } catch (...) {
                std::cout << "⚠️  无法读取图片尺寸信息" << std::endl;
            }
            
            result = analyze_single_image(media_path, prompt);
        }
        
        if (result.success) {
            std::cout << "✅ 分析成功!" << std::endl;
            std::cout << "⏱️  响应时间: " << result.response_time << "秒" << std::endl;
            std::cout << "📝 分析结果: " << result.content << std::endl;
            
            auto tags = extract_tags(result.content);
            if (!tags.empty()) {
                std::cout << "🏷️  提取标签: ";
                for (size_t j = 0; j < tags.size(); ++j) {
                    if (j > 0) std::cout << ", ";
                    std::cout << tags[j];
                }
                std::cout << std::endl;
            }
        } else {
            std::cout << "❌ 分析失败: " << result.error << std::endl;
        }
        
        // 添加文件信息
        result.raw_response["file"] = std::filesystem::path(media_path).filename().string();
        result.raw_response["path"] = media_path;
        result.raw_response["type"] = is_video ? "video" : "image";
        
        results.push_back(result);
        
        // 添加延迟避免频繁调用
        if (i < media_files.size() - 1) {
            std::cout << "⏳ 等待3秒后继续..." << std::endl;
            utils::sleep_seconds(3);
        }
    }
    
    return results;
}

std::vector<std::string> DoubaoMediaAnalyzer::extract_tags(const std::string& content) {
    return utils::extract_tags(content);
}

// 私有方法实现
std::vector<std::string> DoubaoMediaAnalyzer::extract_video_frames(const std::string& video_path, int num_frames) {
    std::vector<std::string> frames_base64;
    
    try {
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("无法打开视频文件");
        }
        
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        double duration = (fps > 0) ? total_frames / fps : 0;
        
        std::cout << "📹 视频信息: " << total_frames << "帧, " 
                  << fps << "FPS, " << duration << "秒" << std::endl;
        
        // 计算提取帧的位置
        std::vector<int> frame_positions;
        if (total_frames <= num_frames) {
            for (int i = 0; i < total_frames; ++i) {
                frame_positions.push_back(i);
            }
        } else {
            int step = total_frames / num_frames;
            for (int i = 0; i < num_frames; ++i) {
                frame_positions.push_back(i * step);
            }
            frame_positions.push_back(total_frames - 1);  // 确保包含最后一帧
        }
        
        for (size_t i = 0; i < frame_positions.size(); ++i) {
            cap.set(cv::CAP_PROP_POS_FRAMES, frame_positions[i]);
            cv::Mat frame;
            bool ret = cap.read(frame);
            
            if (ret && !frame.empty()) {
                // 调整帧大小以控制文件大小
                cv::Mat resized_frame = utils::resize_image(frame, 800);
                
                // 编码为base64
                auto jpeg_data = utils::encode_image_to_jpeg(resized_frame, 85);
                std::string frame_base64 = utils::base64_encode(jpeg_data);
                frames_base64.push_back(frame_base64);
                
                std::cout << "  提取第" << i+1 << "/" << frame_positions.size() 
                          << "帧 (位置: " << frame_positions[i] << "/" << total_frames << ")" << std::endl;
            }
        }
        
        cap.release();
        
    } catch (const std::exception& e) {
        throw std::runtime_error("视频帧提取失败: " + std::string(e.what()));
    }
    
    return frames_base64;
}

AnalysisResult DoubaoMediaAnalyzer::send_analysis_request(const nlohmann::json& payload, int timeout) {
    AnalysisResult result;
    
    try {
        std::vector<std::string> headers = {
            "Authorization: Bearer " + api_key_,
            "Content-Type: application/json"
        };
        
        std::string payload_str = payload.dump();
        std::string response = make_http_request(base_url_, "POST", payload_str, headers, timeout);
        
        return process_response(response, 0); // response_time will be set by caller
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error = "HTTP请求异常: " + std::string(e.what());
        return result;
    }
}

AnalysisResult DoubaoMediaAnalyzer::process_response(const std::string& response_text, double response_time) {
    AnalysisResult result;
    result.response_time = response_time;
    
    try {
        auto json_response = nlohmann::json::parse(response_text);
        
        if (json_response.contains("choices") && json_response["choices"].is_array() && 
            !json_response["choices"].empty()) {
            
            auto choice = json_response["choices"][0];
            if (choice.contains("message") && choice["message"].contains("content")) {
                result.success = true;
                result.content = choice["message"]["content"].get<std::string>();
                
                if (json_response.contains("usage")) {
                    result.usage = json_response["usage"];
                }
                
                result.raw_response = json_response;
            } else {
                result.success = false;
                result.error = "响应格式异常: 缺少content字段";
            }
        } else {
            result.success = false;
            result.error = "响应格式异常: " + response_text;
        }
        
    } catch (const nlohmann::json::parse_error& e) {
        result.success = false;
        result.error = "JSON解析失败: " + std::string(e.what()) + " - Response: " + response_text;
    } catch (const std::exception& e) {
        result.success = false;
        result.error = "处理响应异常: " + std::string(e.what());
    }
    
    return result;
}

std::string DoubaoMediaAnalyzer::make_http_request(const std::string& url, 
                                                  const std::string& method,
                                                  const std::string& data,
                                                  const std::vector<std::string>& headers,
                                                  int timeout) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize CURL");
    }
    
    std::string response;
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, method.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, data.length());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout);
    
    // 设置headers
    struct curl_slist* header_list = nullptr;
    for (const auto& header : headers) {
        header_list = curl_slist_append(header_list, header.c_str());
    }
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
    
    CURLcode res = curl_easy_perform(curl);
    
    curl_slist_free_all(header_list);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        throw std::runtime_error("HTTP请求失败: " + std::string(curl_easy_strerror(res)));
    }
    
    return response;
}
