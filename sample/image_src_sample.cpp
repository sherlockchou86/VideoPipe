
#include "VP.h"

#include "../nodes/vp_image_src_node.h"
#include "../nodes/infers/vp_yolo_detector_node.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_split_node.h"
#include "../nodes/vp_screen_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## image_des_sample ##
* show how vp_image_src_node works, read image from local file or receive image from remote via udp.
*/

#if image_src_sample

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto image_src_0 = std::make_shared<vp_nodes::vp_image_src_node>("image_file_src_0", 0, "./images/test_%d.jpg", 1, 0.4); // read 1 image EVERY 1 second from local files, such as test_0.jpg,test_1.jpg
    /* sending command for test: `gst-launch-1.0 filesrc location=16.mp4 ! qtdemux ! avdec_h264 ! videoconvert ! videoscale ! video/x-raw,width=416,height=416 ! videorate ! video/x-raw,framerate=1/1 ! jpegenc ! rtpjpegpay ! udpsink host=ip port=6000` */
    auto image_src_1 = std::make_shared<vp_nodes::vp_image_src_node>("image_udp_src_1", 1, "6000", 3);                       // receive 1 image EVERY 3 seconds from remote via udp , such as 127.0.0.1:6000
    auto yolo_detector = std::make_shared<vp_nodes::vp_yolo_detector_node>("yolo_detector", "models/det_cls/yolov3-tiny-2022-0721_best.weights", "models/det_cls/yolov3-tiny-2022-0721.cfg", "models/det_cls/yolov3_tiny_5classes.txt");
    auto osd = std::make_shared<vp_nodes::vp_osd_node>("osd");
    auto split = std::make_shared<vp_nodes::vp_split_node>("split_by_channel", true);    // split by channel index (important!)
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    auto screen_des_1 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_1", 1);
    
    // construct pipeline
    yolo_detector->attach_to({image_src_0, image_src_1});
    osd->attach_to({yolo_detector});
    split->attach_to({osd});
    screen_des_0->attach_to({split});
    screen_des_1->attach_to({split});

    image_src_0->start();  // start read from local file
    image_src_1->start();  // start receive from remote via udp

    // for debug purpose
    vp_utils::vp_analysis_board board({image_src_0, image_src_1});
    board.display();
}

#endif