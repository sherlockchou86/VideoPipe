
#include "vp_fake_des_node.h"

namespace vp_nodes {
        
    vp_fake_des_node::vp_fake_des_node(std::string node_name, 
                        int channel_index): vp_des_node(node_name, channel_index) {
        this->initialized();
    }
    
    vp_fake_des_node::~vp_fake_des_node() {
        deinitialized();
    }
}