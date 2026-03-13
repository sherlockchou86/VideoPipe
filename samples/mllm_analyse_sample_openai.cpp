#include "../nodes/vp_image_src_node.h"
#include "../nodes/infers/vp_mllm_analyser_node.h"
#include "../nodes/osd/vp_mllm_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_rtmp_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"
#include "../utils/config_reader.h"

#include <filesystem>

/*
* ## mllm_analyse_sample_openai ##
* image(frame) analyse based on Multimodal Large Language Model(from aliyun or other OpenAI-compatible api services).
* read images from disk and analyse the image using MLLM using the prepared prompt.
*/
int main() {
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();


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


    // create nodes
    auto image_src_0 = std::make_shared<vp_nodes::vp_image_src_node>("image_file_src_0", 0, "./vp_data/test_images/llm/understanding/%d.jpg", 2, 0.5);
    auto writing_prompt = "给图片打标签，要求包含：\n"
                          "1. 先仔细观察图片内容，为图片赋予适合的标签\n"
                          "2. 给出的标签最多不超过5个\n"
                          "3. 输出按以下格式：\n"
                          "通过仔细观察图片，可以为图片赋予这些标签：['标签1', '标签2', '标签3']。";

    auto mllm_analyser_0 = std::make_shared<vp_nodes::vp_mllm_analyser_node>("mllm_analyser_0",                                   // node name
                                                                             modelName,                                           // mllm model name (from aliyun, support image as input)
                                                                             writing_prompt,                                      // prompt
                                                                             apiBase,                                             // api base url
                                                                             apiKey,                                              // api key (from aliyun)
                                                                             llmlib::LLMBackendType::OpenAI);                     // backend type

    auto mllm_osd_0 = std::make_shared<vp_nodes::vp_mllm_osd_node>("osd_0", "./vp_data/font/NotoSansCJKsc-Medium.otf");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    // construct pipeline
    mllm_analyser_0->attach_to({image_src_0});
    mllm_osd_0->attach_to({mllm_analyser_0});
    screen_des_0->attach_to({mllm_osd_0});

    image_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({image_src_0});
    board.display(1, false);

    std::string wait;
    std::getline(std::cin, wait);
    image_src_0->detach_recursively();
}