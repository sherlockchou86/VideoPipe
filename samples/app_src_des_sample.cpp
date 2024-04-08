#include "../nodes/vp_app_src_node.h"
#include "../nodes/infers/vp_yunet_face_detector_node.h"
#include "../nodes/osd/vp_face_osd_node.h"
#include "../nodes/vp_app_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## app_src_des_sample ##
* 1. receive images from host code， using vp_app_src_node
* 2. detect faces and draw results
* 3. display on screen in host code again using cv::imshow(...), using vp_app_des_node
* treat pipeline as a tool
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto app_src_0 = std::make_shared<vp_nodes::vp_app_src_node>("app_src_0", 0);
    auto yunet_face_detector_0 = std::make_shared<vp_nodes::vp_yunet_face_detector_node>("yunet_face_detector_0", "./vp_data/models/face/face_detection_yunet_2022mar.onnx");
    auto osd_0 = std::make_shared<vp_nodes::vp_face_osd_node>("osd_0");
    auto app_des_0 = std::make_shared<vp_nodes::vp_app_des_node>("app_des_0", 0);

    // register callback to display result
    std::string ori_win_title = "original frame using cv::imshow(...)";
    std::string osd_win_title = "osd frame using cv::imshow(...)";
    cv::namedWindow(ori_win_title,cv::WindowFlags::WINDOW_NORMAL);
    cv::namedWindow(osd_win_title,cv::WindowFlags::WINDOW_NORMAL);
    app_des_0->set_app_des_result_hooker([&](std::string node_name, std::shared_ptr<vp_objects::vp_meta> meta) {
        // only deal with frame meta
        if (meta->meta_type == vp_objects::vp_meta_type::FRAME) {
            auto frame_meta = std::dynamic_pointer_cast<vp_objects::vp_frame_meta>(meta);
            cv::imshow(ori_win_title, frame_meta->frame);

            // osd frame may be empty
            if (!frame_meta->osd_frame.empty()) {
                cv::imshow(osd_win_title, frame_meta->osd_frame);
            }

            cv::waitKey(0);
        }
    });

    // construct pipeline
    yunet_face_detector_0->attach_to({app_src_0});
    osd_0->attach_to({yunet_face_detector_0});
    app_des_0->attach_to({osd_0});

    // start pipeline
    app_src_0->start();
    
    // for debug purpose
    vp_utils::vp_analysis_board board({app_src_0});
    board.display(1, false);  // no block

    // read image in host code manually
    auto img = cv::imread("./vp_data/test_images/faces/0.jpg");
    // push image to pipeline in host code manually
    app_src_0->push_frames({img});

    // pipeline will process image asynchronously and display result here...
    std::string wait;
    std:getline(std::cin, wait);
}