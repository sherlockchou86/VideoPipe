#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_pegasus_analyser_node.h"
#include "../nodes/osd/vp_mllm_osd_node.h"
#include "../nodes/vp_screen_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## pegasus_analyse_sample ##
* whole-video semantic structuring based on TwelveLabs Pegasus video understanding model.
*
* unlike mllm_analyse_sample (which sends single frames to a multimodal LLM), Pegasus
* reasons over the ENTIRE video on the TwelveLabs cloud and returns a structured semantic
* description (summary / chapters / highlights / open-ended answers). the resulting text
* is written into frame_meta->description and rendered on screen by vp_mllm_osd_node.
*
* set TWELVELABS_API_KEY in your environment (grab a free key at https://twelvelabs.io),
* and point video_url at the same media the file source reads (or a public http url).
*/
int main() {
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    auto api_key = std::getenv("TWELVELABS_API_KEY") ? std::getenv("TWELVELABS_API_KEY") : "";
    auto video_url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4";

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/19.mp4", 0.6);
    auto pegasus_analyser_0 = std::make_shared<vp_nodes::vp_pegasus_analyser_node>("pegasus_analyser_0",     // node name
                                                                                   api_key,                  // TwelveLabs api key (x-api-key)
                                                                                   "Summarize this video in 3 short sentences.",  // prompt
                                                                                   video_url,                // direct http(s) video url
                                                                                   "",                       // video_id (alternative to url, pegasus1.2 only)
                                                                                   "pegasus1.5",             // model name
                                                                                   2048);                    // max output tokens
    auto pegasus_osd_0 = std::make_shared<vp_nodes::vp_mllm_osd_node>("osd_0", "./vp_data/font/NotoSansCJKsc-Medium.otf");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    // construct pipeline
    pegasus_analyser_0->attach_to({file_src_0});
    pegasus_osd_0->attach_to({pegasus_analyser_0});
    screen_des_0->attach_to({pegasus_osd_0});

    file_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0});
    board.display(1, false);

    std::string wait;
    std::getline(std::cin, wait);
    file_src_0->detach_recursively();
}
