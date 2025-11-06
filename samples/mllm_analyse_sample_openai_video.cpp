#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_mllm_analyser_node.h"
#include "../nodes/osd/vp_mllm_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_file_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"
#include "../utils/config_reader.h"

#include <filesystem>
#include <iostream>

/*
* ## video_mllm_analyse_sample ##
* Video analyse based on Multimodal Large Language Model.
* Read MP4 video file and analyse key frames using MLLM to generate classification labels.
*/
int main(int argc, char* argv[]) {
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // 检查命令行参数
    if (argc < 2) {
        VP_ERROR("Usage: " + std::string(argv[0]) + " <video_file_path>");
        VP_ERROR("Please provide the path to MP4 video file.");
        return -1;
    }

    std::string videoPath = argv[1];
    
    // 检查文件是否存在
    if (!std::filesystem::exists(videoPath)) {
        VP_ERROR("Video file does not exist: " + videoPath);
        return -1;
    }

    // 检查文件扩展名
    if (videoPath.substr(videoPath.find_last_of(".") + 1) != "mp4") {
        VP_WARN("File extension is not .mp4, but will try to process anyway: " + videoPath);
    }

    // 从配置文件读取大模型配置
    auto& configReader = ConfigReader::getInstance();
    std::string configPath = "./key/config.ini";
    
    if (!configReader.loadConfig(configPath)) {
        VP_ERROR("Failed to load config file: " + configPath);
        return -1;
    }

    // 读取配置参数
    std::string modelName = configReader.getValue("mllm_config", "model_name", "");
    std::string apiBase = configReader.getValue("mllm_config", "api_base", "");
    std::string apiKey = configReader.getValue("mllm_config", "api_key", "");

    // 验证配置参数
    if (modelName.empty() || apiBase.empty() || apiKey.empty()) {
        VP_ERROR("Invalid configuration parameters. Please check config.ini");
        VP_ERROR("Model Name: " + modelName);
        VP_ERROR("API Base: " + apiBase);
        VP_ERROR("API Key: " + (apiKey.empty() ? "EMPTY" : "***" + apiKey.substr(apiKey.length() - 4)));
        return -1;
    }

    VP_INFO("Loaded MLLM configuration:");
    VP_INFO("  Model: " + modelName);
    VP_INFO("  API Base: " + apiBase);
    VP_INFO("  API Key: ***" + apiKey.substr(apiKey.length() - 4));
    VP_INFO("Processing video: " + videoPath);

    //预处理 查询视频文件帧和时长信息
   try {
        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            throw std::runtime_error("无法打开视频文件");
        }
        
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        double duration = (fps > 0) ? total_frames / fps : 0;
        
        std::cout << "📹 视频信息: " << total_frames << "帧, " 
                  << fps << "FPS, " << duration << "秒" << std::endl;
        
        // // 计算提取帧的位置
        // std::vector<int> frame_positions;
        // if (total_frames <= num_frames) {
        //     for (int i = 0; i < total_frames; ++i) {
        //         frame_positions.push_back(i);
        //     }
        // } else {
        //     int step = total_frames / num_frames;
        //     for (int i = 0; i < num_frames; ++i) {
        //         frame_positions.push_back(i * step);
        //     }
        //     frame_positions.push_back(total_frames - 1);  // 确保包含最后一帧
        // }

    } catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl;
        return -1;
    }   


    // 创建节点
    // 使用文件源节点读取MP4视频，设置帧率控制以避免处理过多帧
    auto video_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("video_file_src_0", 0, videoPath,0.5f,false,"avdec_h264",9);
    
    // 定义分析提示词
    auto video_analysis_prompt = "请仔细观察视频帧画面内容，为当前画面生成准确的分类标签。\n"
                                 "要求：\n"
                                 "1. 仔细分析画面中的主要对象、场景、活动、颜色、情绪等特征\n"
                                 "2. 生成的标签要具体且相关，最多不超过5个标签\n"
                                 "3. 考虑画面的整体主题和关键元素\n"
                                 "4. 输出格式严格按照：当前画面标签：['标签1', '标签2', '标签3']\n"
                                 "5. 如果画面模糊或无法识别，返回：['无法识别']";

    auto mllm_analyser_0 = std::make_shared<vp_nodes::vp_mllm_analyser_node>("mllm_analyser_0",           // 节点名称
                                                                             modelName,                   // MLLM模型名称
                                                                             video_analysis_prompt,       // 分析提示词
                                                                             apiBase,                     // API基础URL
                                                                             apiKey,                      // API密钥
                                                                             llmlib::LLMBackendType::OpenAI); // 后端类型

    auto mllm_osd_0 = std::make_shared<vp_nodes::vp_mllm_osd_node>("mllm_osd_0", "./vp_data/font/NotoSansCJKsc-Medium.otf");
    
    // 屏幕显示节点 - 实时显示分析结果
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    
    // 文件输出节点 - 可选，保存处理后的视频
    // auto file_des_0 = std::make_shared<vp_nodes::vp_file_des_node>("file_des_0", "output_video_with_labels.mp4");

    // 构建处理管道
    mllm_analyser_0->attach_to({video_src_0});
    mllm_osd_0->attach_to({mllm_analyser_0});
    screen_des_0->attach_to({mllm_osd_0});
    // file_des_0->attach_to({mllm_osd_0});  // 取消注释以保存输出视频

    VP_INFO("Starting video analysis pipeline...");
    video_src_0->start();

    // 调试面板
    vp_utils::vp_analysis_board board({video_src_0});
    board.display(1, false);

    // 等待处理完成或用户中断
    VP_INFO("Video analysis started. Press Enter to stop...");
    std::string wait;
    std::getline(std::cin, wait);
    
    VP_INFO("Stopping pipeline...");
    video_src_0->detach_recursively();
    VP_INFO("Video analysis completed.");

    return 0;
}
