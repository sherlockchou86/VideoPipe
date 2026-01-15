#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_yolo5_seg_node.h"
#include "../nodes/osd/vp_osd_node_v3.h"
#include "../nodes/vp_screen_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## yolov5-seg sample ##
* driving area segmentation(das) using yolov5-seg-v7.0.
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/vehicle_count.mp4", 0.6);
    // 2 classes using yolov5-seg
    auto das_detector = std::make_shared<vp_nodes::vp_yolo5_seg_node>("das_detector", "./vp_data/models/lane/das.onnx", "./vp_data/models/lane/das_2classes.txt");
    auto osd_v3_0 = std::make_shared<vp_nodes::vp_osd_node_v3>("osd_v3_0", "./vp_data/font/NotoSansCJKsc-Medium.otf", false);
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    // construct pipeline
    das_detector->attach_to({file_src_0});
    osd_v3_0->attach_to({das_detector});
    screen_des_0->attach_to({osd_v3_0});

    file_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0});
    board.display();
}