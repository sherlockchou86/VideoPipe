
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
#include "../nodes/osd/vp_pose_osd_node.h"
#include "../nodes/osd/vp_face_osd_node.h"
#include "../nodes/osd/vp_face_osd_node_v2.h"
#include "VP.h"

#if MAIN5

int main() {
    // create src nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./test_video/8.mp4", 0.5);
    //auto rtsp_src_1 = std::make_shared<vp_nodes::vp_rtsp_src_node>("rtsp_src_1", 1, "rtsp://192.168.77.82/file6", 0.4);
    //auto udp_src_2 = std::make_shared<vp_nodes::vp_udp_src_node>("udp_src_2", 2, 6000, 0.5);
    //auto file_src_2 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_2", 2, "./test_video/1.mp4");

    // primary infer node, detector
    auto primary_infer1 = std::make_shared<vp_nodes::vp_yolo_detector_node>("yolo_vehicle_detector", "./models/yolov3-tiny-2022-0721_best.weights", "./models/yolov3-tiny-2022-0721.cfg", "./models/yolov3_5classes.txt", 416, 416);
    //auto primary_infer = std::make_shared<vp_nodes::vp_openpose_detector_node>("primary_infer", "./models/openpose/pose/mpi_15_pose_iter_160000.caffemodel", "./models/openpose/pose/mpi_15_pose_deploy.prototxt","",368,368,1,0,0.1,vp_objects::vp_pose_type::mpi_15);
    //auto primary_infer = std::make_shared<vp_nodes::vp_openpose_detector_node>("primary_infer", "./models/openpose/pose/coco_pose_iter_440000.caffemodel", "./models/openpose/pose/coco_pose_deploy.prototxt","",368,368,1,0,0.1,vp_objects::vp_pose_type::coco);
    
    auto primary_infer2 = std::make_shared<vp_nodes::vp_yunet_face_detector_node>("yunet_face_detector", "./models/face/face_detection_yunet_2022mar.onnx");
    //auto primary_infer3 = std::make_shared<vp_nodes::vp_openpose_detector_node>("openpose_detector", "./models/openpose/pose/body_25_pose_iter_584000.caffemodel", "./models/openpose/pose/body_25_pose_deploy.prototxt","",368,368,1,0,0.1,vp_objects::vp_pose_type::body_25);

    auto secondary_infer1 = std::make_shared<vp_nodes::vp_sface_feature_encoder_node>("sface_face_encoder", "./models/face/face_recognition_sface_2021dec.onnx");

    // simple iou tracker, deep copy or just transfer pointer
    // auto iou_tracker = std::make_shared<vp_nodes::vp_track_node>("iou_tracker");

    // split one branch into multi branches
    auto split = std::make_shared<vp_nodes::vp_split_node>("split", true);

    // draw something nodes
    /*
    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node>("osd_0", vp_nodes::vp_osd_option{100});
    auto osd_1 = std::make_shared<vp_nodes::vp_osd_node>("osd_1", vp_nodes::vp_osd_option{100});
    auto osd_2 = std::make_shared<vp_nodes::vp_osd_node>("osd_2", vp_nodes::vp_osd_option{100}); */

    /*
    auto osd_0 = std::make_shared<vp_nodes::vp_pose_osd_node>("osd_0");
    auto osd_1 = std::make_shared<vp_nodes::vp_pose_osd_node>("osd_1");
    auto osd_2 = std::make_shared<vp_nodes::vp_pose_osd_node>("osd_2"); */

    auto osd_0 = std::make_shared<vp_nodes::vp_face_osd_node>("osd_0");
    //auto osd_1 = std::make_shared<vp_nodes::vp_osd_node>("osd_1", vp_nodes::vp_osd_option{});
    //auto osd_2 = std::make_shared<vp_nodes::vp_pose_osd_node>("osd_2");

/*
    auto osd_00 = std::make_shared<vp_nodes::vp_face_osd_node_v2>("osd_00");
    auto osd_11 = std::make_shared<vp_nodes::vp_face_osd_node_v2>("osd_11");
    auto osd_22 = std::make_shared<vp_nodes::vp_face_osd_node_v2>("osd_22");
*/

    // display on screen for debug purpose
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    //auto screen_des_1 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_1", 1, true, vp_objects::vp_size{640, 360});
    //auto screen_des_2 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_2", 2, true, vp_objects::vp_size{640, 360});

    // push rtmp stream to cloud
    auto rtmp_des_0 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_0", 0, "rtmp://192.168.77.105/live/1000");
    //auto rtmp_des_1 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_1", 1, "rtmp://192.168.77.105/live/1001");
    //auto rtmp_des_2 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_2", 2, "rtmp://192.168.77.105/live/1002");

    // construct the pipe using previous nodes, just attach them one by one
    primary_infer1->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{file_src_0});

    primary_infer2->attach_to({primary_infer1});

    //primary_infer3->attach_to({primary_infer2});

    secondary_infer1->attach_to({primary_infer2});

    //iou_tracker->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{primary_infer2});

    split->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{secondary_infer1});
    
    osd_0->attach_to({split});
    //osd_1->attach_to({split});
    //osd_2->attach_to({split});
/*
    osd_00->attach_to({osd_0});
    osd_11->attach_to({osd_1});
    osd_22->attach_to({osd_2});
*/
    screen_des_0->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_0});
    rtmp_des_0->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_0});
    
    //screen_des_1->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_1});
    //rtmp_des_1->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_1});
    
    //screen_des_2->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_2});
    //rtmp_des_2->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_2});
    
    // start one/all channels
    file_src_0->start();
    //rtsp_src_1->start();
    //file_src_2->start();

    // statistics board for debug purpose
    // vp_utils::vp_statistics_board board(std::vector<std::shared_ptr<vp_nodes::vp_node>>{file_src_0, rtsp_src_1, udp_src_2});
    // board.display();

    vp_utils::vp_analysis_board board(std::vector<std::shared_ptr<vp_nodes::vp_node>>{file_src_0});
    board.display();

    std::getchar();
}

#endif
