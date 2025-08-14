#include "../nodes/vp_rtmp_src_node.h"
#include "../nodes/infers/vp_trt_vehicle_detector.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/track/vp_sort_track_node.h"
#include "../nodes/vp_rtmp_des_node.h"
#include "../nodes/vp_screen_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## rtmp_src_sample ##
* 1 rtmp video input, 1 infer task, and 1 output.
*/

int main() {
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto rtmp_src_0 = std::make_shared<vp_nodes::vp_rtmp_src_node>("rtmp_src_0", 0, "rtmp://192.168.77.196/live/1000", 0.6);
    auto trt_vehicle_detector = std::make_shared<vp_nodes::vp_trt_vehicle_detector>("vehicle_detector", "./vp_data/models/trt/vehicle/vehicle_v8.5.trt");
    auto track_0 = std::make_shared<vp_nodes::vp_sort_track_node>("track_0");
    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node>("osd_0", "./vp_data/font/NotoSansCJKsc-Medium.otf");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    auto rtmp_des_0 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_0", 0, "rtmp://192.168.77.196/live/2000", vp_objects::vp_size{1280, 720}, 1024 * 2);

    // construct pipeline
    trt_vehicle_detector->attach_to({rtmp_src_0});
    track_0->attach_to({trt_vehicle_detector});
    osd_0->attach_to({track_0});
    rtmp_des_0->attach_to({osd_0});

    rtmp_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({rtmp_src_0});
    board.display(1, false);

    std::string wait;
    std::getline(std::cin, wait);
    rtmp_src_0->detach_recursively();
}