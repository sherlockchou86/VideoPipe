#pragma once

#include <map>
#include "../vp_node.h"
#include "../../objects/shapes/vp_point.h"
#include "../../objects/shapes/vp_line.h"
#include "../../objects/vp_image_record_control_meta.h"
#include "../../objects/vp_video_record_control_meta.h"

namespace vp_nodes {
    // behaviour analysis node for crossline (support multi channels)
    class vp_ba_crossline_node: public vp_node 
    {
    private:
        // channel -> counter
        std::map<int, int> all_total_crossline;

        // channel -> line, 1 channel supports only 1 line at most (can be 0, which means no crossline check on this channel)
        std::map<int, vp_objects::vp_line> all_lines;

        // record params
        bool need_record_image;
        bool need_record_video;

        // check if point at one side of line
        bool at_1_side_of_line(vp_objects::vp_point p, vp_objects::vp_line line);
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
    public:
        vp_ba_crossline_node(std::string node_name, 
                            std::map<int, vp_objects::vp_line> lines,
                            bool need_record_image = true,
                            bool need_record_video = false);
        ~vp_ba_crossline_node();

        std::string to_string() override;
    };
}