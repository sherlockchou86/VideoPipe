
#include <vector>
#include <iostream>
#include <memory>

#include "../nodes/vp_file_src_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_split_node.h"
#include "../nodes/vp_rtmp_des_node.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_track_node.h"
#include "../nodes/vp_rtsp_src_node.h"
#include "../nodes/vp_udp_src_node.h"
#include "../nodes/vp_message_broker_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"
#include "../utils/statistics_board/vp_statistics_board.h"

#include "../nodes/infers/vp_yolo_detector_node.h"
#include "../nodes/infers/vp_openpose_detector_node.h"
#include "../nodes/infers/vp_yunet_face_detector_node.h"
#include "../nodes/infers/vp_sface_feature_encoder_node.h"
#include "../nodes/infers/vp_classifier_node.h"
#include "../nodes/osd/vp_pose_osd_node.h"
#include "../nodes/osd/vp_face_osd_node.h"
#include "../nodes/osd/vp_face_osd_node_v2.h"
#include "VP.h"

#if MAIN14

int main() {
    // create src nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./test_video/1.mp4", 0.5);
    auto file_src_1 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_1", 1, "./test_video/10.mp4", 0.4);
    auto rtsp_src_2 = std::make_shared<vp_nodes::vp_rtsp_src_node>("rtsp_src_2", 2, "rtsp://192.168.77.82/file6", 0.3);

    // primary infer node, face detector and vehicle detector
    auto openpose_detector_0 = std::make_shared<vp_nodes::vp_openpose_detector_node>("openpose_detector_0", "./models/openpose/pose/body_25_pose_iter_584000.caffemodel", "./models/openpose/pose/body_25_pose_deploy.prototxt","",368,368,1,0,0.1,vp_objects::vp_pose_type::body_25);
    //auto yunet_face_detector_0 = std::make_shared<vp_nodes::vp_yunet_face_detector_node>("yunet_face_detector_0", "./models/face/face_detection_yunet_2022mar.onnx");
    auto yunet_face_detector_1 = std::make_shared<vp_nodes::vp_yunet_face_detector_node>("yunet_face_detector_1", "./models/face/face_detection_yunet_2022mar.onnx");
    auto yolo_vehicle_detector_2 = std::make_shared<vp_nodes::vp_yolo_detector_node>("yolo_vehicle_detector_2", "./models/yolov3-tiny-2022-0721_best.weights", "./models/yolov3-tiny-2022-0721.cfg", "./models/yolov3_5classes.txt", 416, 416);

    // secondary infer node, face encoder and vehicle classifier
    auto resnet_classifier_0 = std::make_shared<vp_nodes::vp_classifier_node>("resnet_classifier_0", "./models/resnet18-v1-7.onnx", "", "./models/imagenet_1000labels1.txt", 128, 128, 1, std::vector<int>{2, 3, 4});
    //auto sface_face_encoder_0 = std::make_shared<vp_nodes::vp_sface_feature_encoder_node>("sface_face_encoder_0", "./models/face/face_recognition_sface_2021dec.onnx");
    auto sface_face_encoder_1 = std::make_shared<vp_nodes::vp_sface_feature_encoder_node>("sface_face_encoder_1", "./models/face/face_recognition_sface_2021dec.onnx");
    auto resnet_classifier_2 = std::make_shared<vp_nodes::vp_classifier_node>("resnet_classifier_2", "./models/resnet18-v1-7.onnx", "", "./models/imagenet_1000labels1.txt", 128, 128, 1, std::vector<int>{2, 3, 4});

    // draw something nodes
    auto osd_0 = std::make_shared<vp_nodes::vp_pose_osd_node>("osd_0");
    auto osd_1 = std::make_shared<vp_nodes::vp_face_osd_node_v2>("osd_1");
    auto osd_2 = std::make_shared<vp_nodes::vp_osd_node>("osd_2", vp_nodes::vp_osd_option());

    // display on screen for debug purpose
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    auto screen_des_1 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_1", 1);
    auto screen_des_2 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_2", 2);

    // push rtmp stream to cloud
    auto rtmp_des_0 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_0", 0, "rtmp://192.168.77.105/live/10000");
    auto rtmp_des_1 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_1", 1, "rtmp://192.168.77.105/live/10000");
    auto rtmp_des_2 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_2", 2, "rtmp://192.168.77.105/live/10000");

    // construct the pipe using previous nodes, just attach them one by one
    openpose_detector_0->attach_to({file_src_0});
    yunet_face_detector_1->attach_to({file_src_1});
    yolo_vehicle_detector_2->attach_to({rtsp_src_2});

    resnet_classifier_0->attach_to({openpose_detector_0});
    sface_face_encoder_1->attach_to({yunet_face_detector_1});
    resnet_classifier_2->attach_to({yolo_vehicle_detector_2});

    osd_0->attach_to({resnet_classifier_0});
    osd_1->attach_to({sface_face_encoder_1});
    osd_2->attach_to({resnet_classifier_2});

    //screen_des_0->attach_to({osd_0});
    rtmp_des_0->attach_to({osd_0});
    
    //screen_des_1->attach_to({osd_1});
    rtmp_des_1->attach_to({osd_1});

    //screen_des_2->attach_to({osd_2});
    rtmp_des_2->attach_to({osd_2});

    // start one/all channels
    file_src_0->start();
    file_src_1->start();
    rtsp_src_2->start();

    vp_utils::vp_analysis_board board({file_src_0, file_src_1, rtsp_src_2});
    board.display();

    std::getchar();
}

#endif
