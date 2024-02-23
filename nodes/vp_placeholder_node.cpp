
#include "vp_placeholder_node.h"

namespace vp_nodes {
        
    vp_placeholder_node::vp_placeholder_node(std::string node_name): vp_node(node_name) {
        this->initialized();
    }
    
    vp_placeholder_node::~vp_placeholder_node() {
        deinitialized();
    }
}