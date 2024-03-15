#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_yunet_face_detector_node.h"
#include "../nodes/infers/vp_sface_feature_encoder_node.h"
#include "../nodes/osd/vp_face_osd_node_v2.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_rtmp_des_node.h"
#include "../nodes/vp_rtmp_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## nv_hard_codec_sample ##
* use hardware-based `nvv4l2decoder`/`nvv4l2h264enc` gstreamer plugins (come from DeepStream 4.0+) to decode/encode video stream, which would occupy NVIDIA GPUs.
* run `nvidia-smi -a` command to watch the Utilization of GPU/Decode/Encode .
*/

int main() {
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/face.mp4", 0.6, true, "nvv4l2decoder ! nvvideoconvert");
    auto yunet_face_detector_0 = std::make_shared<vp_nodes::vp_yunet_face_detector_node>("yunet_face_detector_0", "./vp_data/models/face/face_detection_yunet_2022mar.onnx");
    auto sface_face_encoder_0 = std::make_shared<vp_nodes::vp_sface_feature_encoder_node>("sface_face_encoder_0", "./vp_data/models/face/face_recognition_sface_2021dec.onnx");
    auto osd_0 = std::make_shared<vp_nodes::vp_face_osd_node_v2>("osd_0");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    auto rtmp_des_0 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_0", 0, "rtmp://192.168.77.60/live/10000", vp_objects::vp_size(), 1024 * 4000, true, "nvvideoconvert ! nvv4l2h264enc");

    // construct pipeline
    yunet_face_detector_0->attach_to({file_src_0});
    sface_face_encoder_0->attach_to({yunet_face_detector_0});
    osd_0->attach_to({sface_face_encoder_0});
    screen_des_0->attach_to({osd_0});
    rtmp_des_0->attach_to({osd_0});

    file_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0});
    board.display(1, false);

    std::string wait;
    std::getline(std::cin, wait);
    file_src_0->detach_recursively();

/*
    cv::VideoCapture cap("filesrc location=/windows2/zhzhi/vp_data/test_video/face.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! nvvideoconvert ! appsink");
    cv::Mat frame;
    cap.read(frame);
    return 0;
*/
}