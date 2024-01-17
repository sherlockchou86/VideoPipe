#include "../VP.h"

#include "../../nodes/vp_image_src_node.h"
#include "../../nodes/infers/vp_trt_vehicle_detector.h"
#include "../../nodes/infers/vp_trt_vehicle_feature_encoder.h"
#include "../../nodes/infers/vp_trt_vehicle_color_classifier.h"
#include "../../nodes/infers/vp_trt_vehicle_type_classifier.h"
#include "../../nodes/infers/vp_trt_vehicle_plate_detector.h"
#include "../../nodes/broker/vp_embeddings_properties_socket_broker_node.h"
#include "../../nodes/osd/vp_osd_node_v2.h"
#include "../../nodes/vp_screen_des_node.h"

#include "../../utils/analysis_board/vp_analysis_board.h"

/*
* ## vehicle_property_and_similiarity_search_sample ##
* generate embedding data AND classification labels for vehicles using VideoPipe, then send them via socket.
*/

#if vehicle_property_and_similiarity_search_sample

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto image_src_0 = std::make_shared<vp_nodes::vp_image_src_node>("image_src_0", 0, "../images/vehicle_property/%d.jpg");
    auto trt_vehicle_detector = std::make_shared<vp_nodes::vp_trt_vehicle_detector>("vehicle_detector", "../models/trt/vehicle/vehicle_v8.5.trt");
    auto trt_vehicle_plate_detector = std::make_shared<vp_nodes::vp_trt_vehicle_plate_detector>("vehicle_plate_detector", "../models/trt/plate/det_v8.5.trt", "../models/trt/plate/rec_v8.5.trt");
    auto trt_vehicle_color_classifier = std::make_shared<vp_nodes::vp_trt_vehicle_color_classifier>("color_cls", "../models/trt/vehicle/vehicle_color_v8.5.trt", std::vector<int>{0, 1, 2});
    auto trt_vehicle_type_classifier = std::make_shared<vp_nodes::vp_trt_vehicle_type_classifier>("type_cls", "../models/trt/vehicle/vehicle_type_v8.5.trt", std::vector<int>{0, 1, 2});
    auto trt_vehicle_feature_encoder = std::make_shared<vp_nodes::vp_trt_vehicle_feature_encoder>("encoder", "../models/trt/vehicle/vehicle_embedding_v8.5.trt", std::vector<int>{0, 1, 2});
    auto embedding_property_broker = std::make_shared<vp_nodes::vp_embeddings_properties_socket_broker_node>("embedding_property_broker", "192.168.77.68", 8989, "static/cropped_images");
    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node_v2>("osd_0", "../../third_party/paddle_ocr/font/NotoSansCJKsc-Medium.otf");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    
    // construct pipeline
    trt_vehicle_detector->attach_to({image_src_0});
    trt_vehicle_plate_detector->attach_to({trt_vehicle_detector});
    trt_vehicle_color_classifier->attach_to({trt_vehicle_plate_detector});
    trt_vehicle_type_classifier->attach_to({trt_vehicle_color_classifier});
    trt_vehicle_feature_encoder->attach_to({trt_vehicle_type_classifier});
    embedding_property_broker->attach_to({trt_vehicle_feature_encoder});
    osd_0->attach_to({embedding_property_broker});
    screen_des_0->attach_to({osd_0});

    // start pipeline
    image_src_0->start();

    // visualize pipeline for debug
    vp_utils::vp_analysis_board board({image_src_0});
    board.display();
}

#endif