
#include <vector>
#include <iostream>
#include <memory>


#include "../nodes/vp_file_src_node.h"
#include "../nodes/vp_primary_infer_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_split_node.h"
#include "VP.h"

#if MAIN2

int main() {

    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./1.mp4");
    //auto file_src_1 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_1", 1, "./1.mp4");

    //auto primary_infer = std::make_shared<vp_nodes::vp_primary_infer_node>("primary_infer", "");

    //auto split = std::make_shared<vp_nodes::vp_split_node>("split", true);

    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    //auto screen_des_1 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_1", 1, true, vp_objects::vp_size{640, 360});


    //primary_infer->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{file_src_0});

    //split->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{primary_infer});
    screen_des_0->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{file_src_0});
    //screen_des_1->attach_to(std::vector<std::shared_ptr<vp_nodes::vp_node>>{split});

    file_src_0->start();
    //file_src_1->start();

    int a;
    std::cin >> a;

    return 0;
}

#endif
