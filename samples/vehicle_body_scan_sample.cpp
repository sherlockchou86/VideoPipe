#include "../nodes/vp_image_src_node.h"
#include "../nodes/infers/vp_trt_vehicle_scanner.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## vehicle_body_scan_sample ##
* detect wheels and vehicle type based on side view of vehicle
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto image_src_0 = std::make_shared<vp_nodes::vp_image_src_node>("image_src_0", 0, "./vp_data/test_images/body/%d.jpg");
    auto vehicle_scanner = std::make_shared<vp_nodes::vp_trt_vehicle_scanner>("vehicle_scanner", "./vp_data/models/trt/vehicle/vehicle_scan_v8.5.trt");
    auto osd = std::make_shared<vp_nodes::vp_osd_node>("osd");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    // construct pipeline
    vehicle_scanner->attach_to({image_src_0});
    osd->attach_to({vehicle_scanner});
    screen_des_0->attach_to({osd});

    image_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({image_src_0});
    board.display();
}