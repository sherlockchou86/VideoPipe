#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_yolo_detector_node.h"
#include "../nodes/infers/vp_classifier_node.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## multi detectors and classifiers sample ##
* show multi infer nodes work together.
* 1 detector and 2 classifiers applied on primary class ids(1/2/3).
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/vehicle_stop.mp4", 0.6);
    /* primary detector */
    // labels for detector model
    // 0 - person
    // 1 - car
    // 2 - bus
    // 3 - truck
    // 4 - 2wheel
    auto primary_detector = std::make_shared<vp_nodes::vp_yolo_detector_node>("primary_detector", "./vp_data/models/det_cls/yolov3-tiny-2022-0721_best.weights", "./vp_data/models/det_cls/yolov3-tiny-2022-0721.cfg", "./vp_data/models/det_cls/yolov3_tiny_5classes.txt", 416, 416, 1);
    /* secondary classifier 1, applied to car(1)/bus(2)/truck(3) only */
    auto _1st_classifier = std::make_shared<vp_nodes::vp_classifier_node>("1st_classifier", "./vp_data/models/det_cls/vehicle/resnet18-batch=N-type_view_0322_nhwc.onnx", "", "./vp_data/models/det_cls/vehicle/vehicle_types.txt", 224, 224, 1, std::vector<int>{1, 2, 3}, 20, 20, 10, false, 1, cv::Scalar(), cv::Scalar(), true, true);
    /* secondary classifier 2, applied to car(1)/bus(2)/truck(3) only */
    auto _2nd_classifier = std::make_shared<vp_nodes::vp_classifier_node>("2nd_classifier", "./vp_data/models/det_cls/vehicle/resnet18-batch=N-color_view_0322_nhwc.onnx", "", "./vp_data/models/det_cls/vehicle/vehicle_colors.txt", 224, 224, 1, std::vector<int>{1, 2, 3}, 20, 20, 10, false, 1, cv::Scalar(), cv::Scalar(), true, true);
    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node>("osd_0");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_o", 0);

    // construct pipeline
    primary_detector->attach_to({file_src_0});
    _1st_classifier->attach_to({primary_detector});
    _2nd_classifier->attach_to({_1st_classifier});
    osd_0->attach_to({_2nd_classifier});
    screen_des_0->attach_to({osd_0});

    // start
    file_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0});
    board.display();
}