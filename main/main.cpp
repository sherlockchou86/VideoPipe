

#include <vector>
#include <iostream>
#include <memory>

#include "../nodes/vp_file_src_node.h"
#include "../nodes/vp_file_des_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_primary_infer_node.h"
#include "../nodes/vp_split_node.h"
#include "../nodes/vp_rtsp_src_node.h"
#include "../nodes/vp_ba_node.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_rtmp_des_node.h"
#include "../nodes/vp_udp_src_node.h"
#include "../nodes/vp_fake_des_node.h"

#include "../objects/shapes/vp_point.h"
#include "../objects/shapes/vp_line.h"
#include "../objects/shapes/vp_rect.h"

#include "../objects/elements/vp_frame_element.h"

#include "VP.h"

#if MAIN
int main() {

    // 2 src nodes
    //auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "1.mp4", 0.5);
    //auto file_src_1 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_1", 1, "1.mp4", 0.5);

    //auto rtsp_src_0 = std::make_shared<vp_nodes::vp_rtsp_src_node>("rtsp_src_0",0,"rtsp://192.168.77.82/file1", 0.5);
    //auto rtsp_src_1 = std::make_shared<vp_nodes::vp_rtsp_src_node>("rtsp_src_1",1,"rtsp://192.168.77.82/file2", 0.5);
    auto rtsp_src_2 = std::make_shared<vp_nodes::vp_rtsp_src_node>("rtsp_src_2",2,"rtsp://192.168.77.82/file3", 0.5);
    auto rtsp_src_3 = std::make_shared<vp_nodes::vp_rtsp_src_node>("rtsp_src_3",3,"rtsp://192.168.77.82/file4", 0.5);
    //auto rtsp_src_4 = std::make_shared<vp_nodes::vp_rtsp_src_node>("rtsp_src_4",4,"rtsp://192.168.77.82/file5", 0.5);

    auto udp_src_0 = std::make_shared<vp_nodes::vp_udp_src_node>("udp_src_0", 0, 6000, 0.5);
    auto udp_src_1 = std::make_shared<vp_nodes::vp_udp_src_node>("udp_src_1", 1, 6001, 0.3);

    // 1 primary node
    auto primary_infer = std::make_shared<vp_nodes::vp_primary_infer_node>("primary_node", "");
    
    // 1 split node
    auto split = std::make_shared<vp_nodes::vp_split_node>("split", true, false);

    // 5 ba nodes
    std::vector<std::shared_ptr<vp_objects::vp_frame_element>> ba_0_elements = {};
    std::vector<std::shared_ptr<vp_objects::vp_frame_element>> ba_1_elements = {};
    std::vector<std::shared_ptr<vp_objects::vp_frame_element>> ba_2_elements = {};
    std::vector<std::shared_ptr<vp_objects::vp_frame_element>> ba_3_elements = {};
    //std::vector<std::shared_ptr<vp_objects::vp_frame_element>> ba_4_elements = {};
    auto ba_0 = std::make_shared<vp_nodes::vp_ba_node>("ba_0", ba_0_elements);
    auto ba_1 = std::make_shared<vp_nodes::vp_ba_node>("ba_1", ba_1_elements);
    auto ba_2 = std::make_shared<vp_nodes::vp_ba_node>("ba_2", ba_2_elements);
    auto ba_3 = std::make_shared<vp_nodes::vp_ba_node>("ba_3", ba_3_elements);
    //auto ba_4 = std::make_shared<vp_nodes::vp_ba_node>("ba_4", ba_4_elements);

    // 5 osd nodes
    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node>("osd_0", vp_nodes::vp_osd_option{10});
    auto osd_1 = std::make_shared<vp_nodes::vp_osd_node>("osd_1", vp_nodes::vp_osd_option{100});
    auto osd_2 = std::make_shared<vp_nodes::vp_osd_node>("osd_2", vp_nodes::vp_osd_option{100});
    auto osd_3 = std::make_shared<vp_nodes::vp_osd_node>("osd_3", vp_nodes::vp_osd_option{100});
    //auto osd_4 = std::make_shared<vp_nodes::vp_osd_node>("osd_4", vp_nodes::vp_osd_option{100});

    // 5 screen des nodes
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    auto screen_des_1 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_1", 1);
    //auto screen_des_2 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_2", 2);
    //auto screen_des_3 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_3", 3);
    //auto screen_des_4 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_4", 4);

    // 5 rtmp des nodes
    auto rtmp_des_0 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_0", 0, "rtmp://192.168.77.105/wtoe/10000");
    auto rtmp_des_1 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_1", 1, "rtmp://192.168.77.105/wtoe/10000");
    auto rtmp_des_2 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_2", 2, "rtmp://192.168.77.105/wtoe/10000");
    auto rtmp_des_3 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_3", 3, "rtmp://192.168.77.105/wtoe/10000");
    //auto rtmp_des_4 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_4", 4, "rtmp://192.168.77.105/wtoe/10000");

    auto fake_des_2 = std::make_shared<vp_nodes::vp_fake_des_node>("fake_des_2", 2);
    auto fake_des_3 = std::make_shared<vp_nodes::vp_fake_des_node>("fake_des_3", 3);

    // construct pipeline
    primary_infer->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{udp_src_0, udp_src_1, rtsp_src_2, rtsp_src_3});
    split->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{primary_infer});

    ba_0->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{split});
    ba_1->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{split});
    ba_2->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{split});
    ba_3->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{split});
    //ba_4->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{split});

    osd_0->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{ba_0});
    osd_1->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{ba_1});
    osd_2->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{ba_2});
    osd_3->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{ba_3});
    //osd_4->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{ba_4});

    screen_des_0->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_0});
    screen_des_1->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_1});
    //screen_des_2->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_2});
    //screen_des_3->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_3});
    //screen_des_4->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_4});

    rtmp_des_0->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_0});
    rtmp_des_1->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_1});
    rtmp_des_2->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_2});
    rtmp_des_3->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_3});
    //rtmp_des_4->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_4});

    fake_des_2->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_2});
    fake_des_3->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{osd_3});

    /* test
    std::shared_ptr<vp_nodes::vp_primary_infer_node> primary_infer_1 = std::make_shared<vp_nodes::vp_primary_infer_node>("primary_node_1", "");
    rtsp_src_0->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{primary_infer_1});
    */

    //rtsp_src_0->start();
    //rtsp_src_1->start(); 
    //rtsp_src_2->start(); 
    //rtsp_src_3->start(); 
    //rtsp_src_4->start(); 

    //file_src_0->start();
    //file_src_1->start(); 

    //udp_src_0->start();
    //udp_src_1->start();

    vp_utils::vp_statistics_board board(std::vector<std::shared_ptr<vp_nodes::vp_node>>{udp_src_0, udp_src_1, rtsp_src_2, rtsp_src_3});
    auto i = 0;
    std::cin >> i;
    std::cout << i;
    return 0;
}

#endif