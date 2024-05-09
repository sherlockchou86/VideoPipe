#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_trt_yolov8_detector.h"
#include "../nodes/infers/vp_trt_yolov8_seg_detector.h"
#include "../nodes/infers/vp_trt_yolov8_pose_detector.h"
#include "../nodes/osd/vp_osd_node_v3.h"
#include "../nodes/osd/vp_pose_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## trt yolov8 sample ##
* detection/segmentation/pose_estimation using yolov8 based on tensorrt
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes for 1st pipeline
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/face2.mp4");
    auto yolo8_detector = std::make_shared<vp_nodes::vp_trt_yolov8_detector>("yolo8_detector", "./vp_data/models/trt/others/yolov8s_v8.5.engine", "./vp_data/models/coco_80classes.txt");
    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node_v3>("osd_0", "./vp_data/font/NotoSansCJKsc-Medium.otf");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    // create nodes for 2nd pipeline
    auto file_src_1 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_1", 1, "./vp_data/test_video/face2.mp4");
    auto yolo8_seg_detector = std::make_shared<vp_nodes::vp_trt_yolov8_seg_detector>("yolo8_seg_detector", "./vp_data/models/trt/others/yolov8s-seg_v8.5.engine", "./vp_data/models/coco_80classes.txt");
    auto osd_1 = std::make_shared<vp_nodes::vp_osd_node_v3>("osd_1", "./vp_data/font/NotoSansCJKsc-Medium.otf");
    auto screen_des_1 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_1", 1);

    // create nodes for 3rd pipeline
    auto file_src_2 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_2", 2, "./vp_data/test_video/face2.mp4");
    auto yolo8_pose_detector = std::make_shared<vp_nodes::vp_trt_yolov8_pose_detector>("yolo8_pose_detector", "./vp_data/models/trt/others/yolov8s-pose_v8.5.engine");
    auto osd_2 = std::make_shared<vp_nodes::vp_pose_osd_node>("osd_2");
    auto screen_des_2 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_2", 2);

    // construct 1st pipeline
    yolo8_detector->attach_to({file_src_0});
    osd_0->attach_to({yolo8_detector});
    screen_des_0->attach_to({osd_0});

    // construct 2nd pipeline
    yolo8_seg_detector->attach_to({file_src_1});
    osd_1->attach_to({yolo8_seg_detector});
    screen_des_1->attach_to({osd_1});

    // construct 3rd pipeline
    yolo8_pose_detector->attach_to({file_src_2});
    osd_2->attach_to({yolo8_pose_detector});
    screen_des_2->attach_to({osd_2});

    // start pipelines
    file_src_0->start();
    file_src_1->start();
    file_src_2->start();

    // visualize pipeline for debug
    vp_utils::vp_analysis_board board({file_src_0, file_src_1, file_src_2});
    board.display();
}