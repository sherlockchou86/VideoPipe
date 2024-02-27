#include "../nodes/vp_image_src_node.h"
#include "../nodes/infers/vp_trt_vehicle_detector.h"
#include "../nodes/infers/vp_trt_vehicle_color_classifier.h"
#include "../nodes/infers/vp_trt_vehicle_type_classifier.h"
#include "../nodes/infers/vp_trt_vehicle_feature_encoder.h"
#include "../nodes/track/vp_sort_track_node.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## multi_trt_infer_nodes_sample ##
* detect/classify/encoding on vehicle object using tensorrt
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto image_src_0 = std::make_shared<vp_nodes::vp_image_src_node>("image_src_0", 0, "./vp_data/test_images/vehicle/%d.jpg");
    auto trt_vehicle_detector = std::make_shared<vp_nodes::vp_trt_vehicle_detector>("trt_detector", "./vp_data/models/trt/vehicle/vehicle_v8.5.trt");
    auto trt_vehicle_color_classifier = std::make_shared<vp_nodes::vp_trt_vehicle_color_classifier>("trt_color_cls", "./vp_data/models/trt/vehicle/vehicle_color_v8.5.trt", std::vector<int>{0, 1, 2});
    auto trt_vehicle_type_classifier = std::make_shared<vp_nodes::vp_trt_vehicle_type_classifier>("trt_type_cls", "./vp_data/models/trt/vehicle/vehicle_type_v8.5.trt", std::vector<int>{0, 1, 2});
    auto trt_vehicle_feature_encoder = std::make_shared<vp_nodes::vp_trt_vehicle_feature_encoder>("trt_encoder", "./vp_data/models/trt/vehicle/vehicle_embedding_v8.5.trt", std::vector<int>{0, 1, 2});
    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node>("osd_0");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    
    // construct pipeline
    trt_vehicle_detector->attach_to({image_src_0});
    trt_vehicle_color_classifier->attach_to({trt_vehicle_detector});
    trt_vehicle_type_classifier->attach_to({trt_vehicle_color_classifier});
    trt_vehicle_feature_encoder->attach_to({trt_vehicle_type_classifier});
    osd_0->attach_to({trt_vehicle_feature_encoder});
    screen_des_0->attach_to({osd_0});

    // start pipeline
    image_src_0->start();

    // visualize pipeline for debug
    vp_utils::vp_analysis_board board({image_src_0});
    board.display();
}