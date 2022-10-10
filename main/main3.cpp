

#include "VP.h"

#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_yolo_detector_node.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_rtmp_des_node.h"
#include "../nodes/vp_file_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"

/*
* vp_yolo_detector_node sample
* load yolov5 model with onnx format
*/


#if MAIN3

int main() {
    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./test_video/13.mp4");
    auto yolo_vehicle_detector = std::make_shared<vp_nodes::vp_yolo_detector_node>("yolo_vehicle_detector", "./models/yolov5_vehicle.onnx", "", "./models/yolov5_5classes.txt");
    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node>("osd_0", "../third_party/paddle_ocr/font/NotoSansCJKsc-Medium.otf");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0, true, vp_objects::vp_size{640, 360});
    auto rtmp_des_0 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_0", 0, "rtmp://192.168.77.105/live/10000", vp_objects::vp_size{1280, 720});
    
    // construct pipeline
    yolo_vehicle_detector->attach_to({file_src_0});
    osd_0->attach_to({yolo_vehicle_detector});

    // split into 3 sub branches automatically
    screen_des_0->attach_to({osd_0});
    rtmp_des_0->attach_to({osd_0});

    // start pipeline
    file_src_0->start();

    // visualize pipeline for debug
    vp_utils::vp_analysis_board board({file_src_0});
    board.display(1, false);  // no block

    std::getchar();
}

#endif