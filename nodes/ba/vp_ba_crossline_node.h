#pragma once

#include "../vp_node.h"
#include "../../objects/shapes/vp_point.h"
#include "../../objects/shapes/vp_line.h"

namespace vp_nodes {
    // behaviour analysis node for crossline
    class vp_ba_crossline_node: public vp_node 
    {
    private:
        /* data */
        int total_crossline = 0;

        // check if point at one side of line
        bool at_1_side_of_line(vp_objects::vp_point p, vp_objects::vp_line line);

        // line to check
        vp_objects::vp_line line;
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
    public:
        vp_ba_crossline_node(std::string node_name, vp_objects::vp_line line);
        ~vp_ba_crossline_node();
    };
}