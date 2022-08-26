
#pragma once

#include "vp_des_node.h"

namespace vp_nodes {
    // fake des node, do nothing just a placeholder
    class vp_fake_des_node: public vp_des_node {
    private:
        /* data */
    public:
        vp_fake_des_node(std::string node_name, 
                        int channel_index);
        ~vp_fake_des_node();
    };

}