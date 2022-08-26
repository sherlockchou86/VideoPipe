
#include "vp_track_node.h"

namespace vp_nodes {
        
    vp_track_node::vp_track_node(std::string node_name): vp_node(node_name)
    {
        this->initialized();
    }
    
    vp_track_node::~vp_track_node()
    {
    }

    std::shared_ptr<vp_objects::vp_meta> vp_track_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return meta;
    }

    std::shared_ptr<vp_objects::vp_meta> vp_track_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        /*
        if (meta->frame_index % 10 == 0) {          
            std::this_thread::sleep_for(std::chrono::milliseconds(29));
        }
        if (meta->frame_index % 100 == 0) {          
            std::this_thread::sleep_for(std::chrono::milliseconds(84));
        }
        if (meta->frame_index % 130 == 0) {          
            std::this_thread::sleep_for(std::chrono::milliseconds(60));
        }*/
        return meta;
    }
}