
#include "VP.h"

#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_trt_vehicle_detector.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_split_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"
#include "../nodes/record/vp_record_node.h"
#include "../nodes/broker/vp_json_console_broker_node.h"
#include "../nodes/vp_rtsp_src_node.h"
#include "../nodes/vp_rtmp_des_node.h"
#include "../nodes/infers/vp_trt_vehicle_type_classifier.h"
/*
* ## dynamic_pipeline_sample2 ##
* insert/remove nodes to/from pipeline step by step, then process exits normally.
*/

#if dynamic_pipeline_sample2

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./test_video/vehicle_stop3.mp4", 0.4);

    auto trt_vehicle_detector = std::make_shared<vp_nodes::vp_trt_vehicle_detector>("trt_vehicle_detector", "./models/trt/vehicle/vehicle_v8.5.trt");
    auto osd = std::make_shared<vp_nodes::vp_osd_node>("osd", "../third_party/paddle_ocr/font/NotoSansCJKsc-Medium.otf");
    auto console_broker = std::make_shared<vp_nodes::vp_json_console_broker_node>("console_broker");
    auto split = std::make_shared<vp_nodes::vp_split_node>("split", true);
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    trt_vehicle_detector->attach_to({file_src_0});
    osd->attach_to({trt_vehicle_detector});
    console_broker->attach_to({osd});
    split->attach_to({console_broker});
    screen_des_0->attach_to({split});
    file_src_0->start();

    std::vector<std::shared_ptr<vp_nodes::vp_node>> src_nodes = {file_src_0};
    std::string wait;

    std::cin >> wait;         // display pipeline
    vp_utils::vp_analysis_board board(src_nodes);
    board.display(1, false);  // no block

    std::cin >> wait;         // add rtsp input
    auto rtsp_src_1 = std::make_shared<vp_nodes::vp_rtsp_src_node>("rtsp_src_1", 1, "rtsp://admin:admin12345@192.168.77.203", 0.4);
    trt_vehicle_detector->attach_to({rtsp_src_1});
    rtsp_src_1->start();
    src_nodes.push_back(rtsp_src_1);
    board.reload(src_nodes);

    std::cin >> wait;         // add rtmp output
    auto rtmp_des_1 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_1", 1, "rtmp://192.168.77.60/live/dynamic_pipeline_sample2");
    rtmp_des_1->attach_to({split});
    board.reload();

    std::cin >> wait;         // add classifier
    auto trt_vehicle_type_classifier = std::make_shared<vp_nodes::vp_trt_vehicle_type_classifier>("trt_type_cls", "./models/trt/vehicle/vehicle_type_v8.5.trt", std::vector<int>{0, 1, 2});
    osd->detach();
    osd->attach_to({trt_vehicle_type_classifier});
    trt_vehicle_type_classifier->attach_to({trt_vehicle_detector});
    board.reload();

    std::cin >> wait;         // remove broker
    console_broker->detach();
    split->detach();
    split->attach_to({osd});
    board.reload();

    std::cin >> wait;        // destroy pipeline and process exit
    for(auto& n: src_nodes) {
        n->detach_recursively();
    }
}

#endif