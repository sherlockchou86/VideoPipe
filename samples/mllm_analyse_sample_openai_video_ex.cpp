#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_mllm_analyser_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"
#include "../utils/config_reader.h"

#include "../third_party/cpp_analyzelib/include/DoubaoMediaAnalyzer.hpp"
#include "../third_party/cpp_analyzelib/include/utils.hpp"
#include "../third_party/cpp_analyzelib/include/config.hpp"


#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <cstring> 

// 提示词函数
std::string get_image_prompt()
{
    return R"(请仔细观察图片内容，为图片生成合适的标签。要求：
1. 仔细观察图片的各个细节
2. 生成的标签要准确反映图片内容
3. 标签数量不超过5个
4. 输出格式：通过分析图片，生成的标签为：['标签1', '标签2', '标签3'])";
}

std::string get_video_prompt()
{
    return R"(请仔细观察视频的关键帧内容，为视频生成合适的标签。要求：
1. 综合分析视频的整体内容和关键帧
2. 生成的标签要准确反映视频的主题、场景、动作等
3. 标签数量不超过8个
4. 输出格式：通过分析视频，生成的标签为：['标签1', '标签2', '标签3'])";
}

void print_usage()
{
    std::cout << "用法: doubao_analyzer [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  --api-key KEY        豆包API密钥 (必需)" << std::endl;
    std::cout << "  --image PATH         单张图片路径" << std::endl;
    std::cout << "  --video PATH         单个视频路径" << std::endl;
    std::cout << "  --folder PATH        媒体文件夹路径" << std::endl;
    std::cout << "  --file-type TYPE     分析的文件类型 [all|image|video] (默认: all)" << std::endl;
    std::cout << "  --prompt TEXT        自定义提示词" << std::endl;
    std::cout << "  --max-files NUM      最大分析文件数量 (默认: 5)" << std::endl;
    std::cout << "  --video-frames NUM   视频提取帧数 (默认: 5)" << std::endl;
    std::cout << "  --output PATH        结果保存路径" << std::endl;
    std::cout << "  --help               显示此帮助信息" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  doubao_analyzer --api-key YOUR_KEY --image test.jpg" << std::endl;
    std::cout << "  doubao_analyzer --api-key YOUR_KEY --video test.mp4 --video-frames 8" << std::endl;
    std::cout << "  doubao_analyzer --api-key YOUR_KEY --folder ./media --file-type all" << std::endl;
}

void print_result(const AnalysisResult &result, const std::string &media_type)
{
    if (result.success)
    {
        std::cout << "✅ " << media_type << "分析成功!" << std::endl;
        std::cout << "⏱️  响应时间: " << result.response_time << "秒" << std::endl;
        std::cout << "📝 分析结果:" << std::endl
                  << result.content << std::endl;

        auto tags = utils::extract_tags(result.content);
        if (!tags.empty())
        {
            std::cout << "🏷️  提取标签: ";
            for (size_t i = 0; i < tags.size(); ++i)
            {
                if (i > 0)
                    std::cout << ", ";
                std::cout << tags[i];
            }
            std::cout << std::endl;
        }
    }
    else
    {
        std::cout << "❌ " << media_type << "分析失败: " << result.error << std::endl;
    }
}

void print_statistics(const std::vector<AnalysisResult> &results)
{
    int success_count = 0;
    int total_count = results.size();
    int video_count = 0;
    int image_count = 0;

    double total_time = 0;
    double video_total_time = 0;
    double image_total_time = 0;
    int video_success_count = 0;
    int image_success_count = 0;

    for (const auto &result : results)
    {
        if (result.success)
        {
            success_count++;
            total_time += result.response_time;
        }

        if (result.raw_response.contains("type"))
        {
            std::string type = result.raw_response["type"];
            if (type == "video")
            {
                video_count++;
                if (result.success)
                {
                    video_total_time += result.response_time;
                    video_success_count++;
                }
            }
            else if (type == "image")
            {
                image_count++;
                if (result.success)
                {
                    image_total_time += result.response_time;
                    image_success_count++;
                }
            }
        }
    }

    std::cout << "\n📊 分析统计:" << std::endl;
    std::cout << "   总文件数: " << total_count << std::endl;
    std::cout << "   成功分析: " << success_count << "/" << total_count << std::endl;
    std::cout << "   图片文件: " << image_count << std::endl;
    std::cout << "   视频文件: " << video_count << std::endl;

    if (success_count > 0)
    {
        double avg_time = total_time / success_count;
        std::cout << "⏱️  平均响应时间: " << avg_time << "秒" << std::endl;

        if (image_success_count > 0)
        {
            double avg_image_time = image_total_time / image_success_count;
            std::cout << "   图片平均时间: " << avg_image_time << "秒" << std::endl;
        }

        if (video_success_count > 0)
        {
            double avg_video_time = video_total_time / video_success_count;
            std::cout << "   视频平均时间: " << avg_video_time << "秒" << std::endl;
        }
    }
}


/*
* ## video_classification_direct ##
* Direct video analysis using MLLM to get overall classification labels
*/
int main(int argc, char* argv[]) {
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::WARN);
    VP_LOGGER_INIT();

    // 检查命令行参数
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file_path>" << std::endl;
        std::cerr << "Example: " << argv[0] << " test_video.mp4" << std::endl;
        return -1;
    }

    std::string videoPath = argv[1];
    
    // 检查文件是否存在
    if (!std::filesystem::exists(videoPath)) {
        std::cerr << "Error: Video file does not exist: " << videoPath << std::endl;
        return -1;
    }

    // 从配置文件读取大模型配置
    auto& configReader = ConfigReader::getInstance();
    std::string configPath = "./key/config.ini";
    
    if (!configReader.loadConfig(configPath)) {
        std::cerr << "Error: Failed to load config file: " << configPath << std::endl;
        return -1;
    }

    // 读取配置参数
    std::string modelName = configReader.getValue("mllm_config", "model_name", "");
    std::string apiBase = configReader.getValue("mllm_config", "api_base", "");
    std::string apiKey = configReader.getValue("mllm_config", "api_key", "");

    // 验证配置参数
    if (modelName.empty() || apiBase.empty() || apiKey.empty()) {
        std::cerr << "Error: Invalid configuration parameters. Please check config.ini" << std::endl;
        return -1;
    }

    std::cout << "==========================================" << std::endl;
    std::cout << "Direct Video Classification Analysis" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Video: " << videoPath << std::endl;
    std::cout << "Model: " << modelName << std::endl;
    std::cout << "==========================================" << std::endl;


    // 解析命令行参数
    std::string api_key;
    std::string image_path;
    std::string video_path;
    std::string folder_path;
    std::string file_type = "all";
    std::string prompt;
    std::string output_path;
    int max_files = 5;
    int video_frames = 5; // 默认提取5帧


    // 创建分析器
    api_key = apiKey; // 从配置文件中读取
    video_path = videoPath; // 从命令行参数中读取
    
    DoubaoMediaAnalyzer analyzer(api_key);

    std::cout << "🚀 豆包大模型媒体分析调试工具（支持图片和视频）" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // 测试连接
    if (!analyzer.test_connection())
    {
        return 1;
    }

    std::vector<AnalysisResult> results;

    std::cout << "\n🎬 分析单个视频: " << video_path << std::endl;
    std::string analysis_prompt = prompt.empty() ? get_video_prompt() : prompt;
    auto result = analyzer.analyze_single_video(video_path, analysis_prompt, 2000, video_frames);
    print_result(result, "视频");

    result.raw_response["file"] = std::filesystem::path(video_path).filename().string();
    result.raw_response["path"] = video_path;
    result.raw_response["type"] = "video";
    results.push_back(result);
    
    // 统计信息
    if (!results.empty())
    {
        print_statistics(results);
    }

    return 1; 
}
