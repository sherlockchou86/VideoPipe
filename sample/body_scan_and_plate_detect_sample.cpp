
#include "VP.h"

#include "../nodes/vp_image_src_node.h"
#include "../nodes/infers/vp_trt_vehicle_scanner.h"
#include "../nodes/infers/vp_trt_vehicle_plate_detector_v2.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/osd/vp_plate_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## body_scan_and_plate_detect_sample ##
* first channel detects wheels and vehicle type based on side view of vehicle
* second channel detects plate based on head view of vehicle
*/

#if body_scan_and_plate_detect_sample

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto image_src_0 = std::make_shared<vp_nodes::vp_image_src_node>("image_src_0", 0, "images/body2/%d.jpg");
    auto image_src_1 = std::make_shared<vp_nodes::vp_image_src_node>("image_src_1", 1, "images/plates2/%d.jpg");
    auto vehicle_scanner = std::make_shared<vp_nodes::vp_trt_vehicle_scanner>("vehicle_scanner", 
                                                                            "models/trt/vehicle/vehicle_scan_v8.5.trt");
    auto plate_detector = std::make_shared<vp_nodes::vp_trt_vehicle_plate_detector_v2>("plate_detector", 
                                                                                    "models/trt/plate/det_v8.5.trt", 
                                                                                    "models/trt/plate/rec_v8.5.trt");
    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node>("osd_0");
    auto osd_1 = std::make_shared<vp_nodes::vp_plate_osd_node>("osd_1", "../third_party/paddle_ocr/font/NotoSansCJKsc-Medium.otf");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    auto screen_des_1 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_1", 1);

    // construct first pipeline
    vehicle_scanner->attach_to({image_src_0});
    osd_0->attach_to({vehicle_scanner});
    screen_des_0->attach_to({osd_0});
    // construct second pipeline
    plate_detector->attach_to({image_src_1});
    osd_1->attach_to({plate_detector});
    screen_des_1->attach_to({osd_1});

    image_src_0->start();
    image_src_1->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({image_src_0, image_src_1});
    board.display();
}

#endif