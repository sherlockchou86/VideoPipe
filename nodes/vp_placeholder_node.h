
#pragma once

#include "vp_node.h"

namespace vp_nodes {
    // placeholder node, do nothing just a placeholder in the middle of pipeline
    class vp_placeholder_node: public vp_node {
    private:
        /* data */
    public:
        vp_placeholder_node(std::string node_name);
        ~vp_placeholder_node();
    };

}