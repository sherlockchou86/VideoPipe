#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_enet_seg_node.h"
#include "../nodes/osd/vp_seg_osd_node.h"
#include "../nodes/vp_screen_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

/*
* ## enet seg sample ##
* semantic segmentation based on ENet.
* 1 input, 2 outputs including orignal frame and mask frame.
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/enet_seg.mp4");
    auto enet_seg = std::make_shared<vp_nodes::vp_enet_seg_node>("enet_seg", "./vp_data/models/enet-cityscapes/enet-model.net");
    auto seg_osd_0 = std::make_shared<vp_nodes::vp_seg_osd_node>("seg_osd_0", "./vp_data/models/enet-cityscapes/enet-classes.txt", "./vp_data/models/enet-cityscapes/enet-colors.txt");
    auto screen_des_mask = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_mask", 0, true, vp_objects::vp_size(400, 225));
    auto screen_des_original = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_original", 0, false, vp_objects::vp_size(400, 225));

    // construct pipeline
    enet_seg->attach_to({file_src_0});
    seg_osd_0->attach_to({enet_seg});
    screen_des_mask->attach_to({seg_osd_0});
    screen_des_original->attach_to({seg_osd_0});

    file_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0});
    board.display();
}