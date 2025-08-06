#include "../nodes/vp_rtsp_src_node.h"
#include "../nodes/infers/vp_yunet_face_detector_node.h"
#include "../nodes/infers/vp_sface_feature_encoder_node.h"
#include "../nodes/osd/vp_face_osd_node_v2.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_rtmp_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## rtsp_src_sample ##
* 1 rtsp video input, 1 infer task, and 1 output.
* support switching/restarting input rtsp stream.
*/

int main() {
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto rtsp_src_0 = std::make_shared<vp_nodes::vp_rtsp_src_node>("rtsp_src_0", 0, "rtsp://192.168.77.213/live/mainstream", 0.6);
    auto yunet_face_detector_0 = std::make_shared<vp_nodes::vp_yunet_face_detector_node>("yunet_face_detector_0", "./vp_data/models/face/face_detection_yunet_2022mar.onnx");
    auto sface_face_encoder_0 = std::make_shared<vp_nodes::vp_sface_feature_encoder_node>("sface_face_encoder_0", "./vp_data/models/face/face_recognition_sface_2021dec.onnx");
    auto osd_0 = std::make_shared<vp_nodes::vp_face_osd_node_v2>("osd_0");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    // construct pipeline
    yunet_face_detector_0->attach_to({rtsp_src_0});
    sface_face_encoder_0->attach_to({yunet_face_detector_0});
    osd_0->attach_to({sface_face_encoder_0});
    screen_des_0->attach_to({osd_0});

    rtsp_src_0->start();

    /* manually switching/restarting RTSP video sources, where the video sources have different widths, heights, and frame rates.
       continuously print width, height, fps. */
    while (true) {
        auto width = rtsp_src_0->get_original_width();
        auto height = rtsp_src_0->get_original_height();
        auto fps = rtsp_src_0->get_original_fps();
        std::cout << "original_width: " << width << "original_height: " << height << "original_fps: " << fps << std::endl;
        this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    
    rtsp_src_0->detach_recursively();
}