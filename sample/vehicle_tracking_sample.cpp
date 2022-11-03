

#include "VP.h"

#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_trt_vehicle_detector.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/track/vp_sort_track_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

#if vehicle_tracking_sample

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./test_video/8.mp4", 0.5);
    auto trt_vehicle_detector = std::make_shared<vp_nodes::vp_trt_vehicle_detector>("vehicle_detector", "./models/trt/vehicle/vehicle.trt");
    auto track_0 = std::make_shared<vp_nodes::vp_sort_track_node>("track_0");
    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node>("osd_0", "../third_party/paddle_ocr/font/NotoSansCJKsc-Medium.otf");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    // construct pipeline
    trt_vehicle_detector->attach_to({file_src_0});
    track_0->attach_to({trt_vehicle_detector});
    osd_0->attach_to({track_0});
    screen_des_0->attach_to({osd_0});

    // start pipeline
    file_src_0->start();

    // visualize pipeline for debug
    vp_utils::vp_analysis_board board({file_src_0});
    board.display();
}

#endif