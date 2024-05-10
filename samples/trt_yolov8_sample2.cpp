#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_trt_yolov8_detector.h"
#include "../nodes/infers/vp_trt_yolov8_classifier.h"
#include "../nodes/osd/vp_osd_node_v3.h"
#include "../nodes/vp_screen_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## trt yolov8 sample2 ##
* detection/classification using yolov8 based on tensorrt
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/mask_rcnn.mp4");
    auto yolo8_detector = std::make_shared<vp_nodes::vp_trt_yolov8_detector>("yolo8_detector", "./vp_data/models/trt/others/yolov8s_v8.5.engine", "./vp_data/models/coco_80classes.txt");
    auto yolo8_classifier = std::make_shared<vp_nodes::vp_trt_yolov8_classifier>("yolo8_classifier", "./vp_data/models/trt/others/yolov8s-cls_v8.5.engine", "./vp_data/models/imagenet_1000labels1.txt");
    auto osd = std::make_shared<vp_nodes::vp_osd_node_v3>("osd", "./vp_data/font/NotoSansCJKsc-Medium.otf");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    // construct pipeline
    yolo8_detector->attach_to({file_src_0});
    yolo8_classifier->attach_to({yolo8_detector});
    osd->attach_to({yolo8_classifier});
    screen_des_0->attach_to({osd});

    // start pipeline
    file_src_0->start();

    // visualize pipeline for debug
    vp_utils::vp_analysis_board board({file_src_0});
    board.display();
}