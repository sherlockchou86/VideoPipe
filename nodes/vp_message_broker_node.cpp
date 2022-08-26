
#include "vp_message_broker_node.h"

namespace vp_nodes {
        
    vp_message_broker_node::vp_message_broker_node(std::string node_name): vp_node(node_name) {
        this->initialized();
    }
    
    vp_message_broker_node::~vp_message_broker_node() {

    }

    std::shared_ptr<vp_objects::vp_meta> vp_message_broker_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return meta;
    }

    std::shared_ptr<vp_objects::vp_meta> vp_message_broker_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        /*
        if (meta->frame_index % 15 == 0) {          
            std::this_thread::sleep_for(std::chrono::milliseconds(28));
        }
        if (meta->frame_index % 73 == 0) {          
            std::this_thread::sleep_for(std::chrono::milliseconds(64));
        }*/
        return meta;
    }
}