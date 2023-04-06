

#include "vp_ba_crossline_node.h"

namespace vp_nodes {
    
    vp_ba_crossline_node::vp_ba_crossline_node(std::string node_name, 
                                                vp_objects::vp_line line):
                                                vp_node(node_name), line(line) {
        this->initialized();
    }
    
    vp_ba_crossline_node::~vp_ba_crossline_node() {

    }
    
    std::shared_ptr<vp_objects::vp_meta> vp_ba_crossline_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // for vp_frame_target only
        std::vector<int> involve_targets;
        for (auto& target : meta->targets) {
            auto len = target->tracks.size();
            if (len > 1 && target->track_id >= 0) {
                // check the last 2 points in tracks
                auto p1 = target->tracks[len - 1].track_point();
                auto p2 = target->tracks[len - 2].track_point();

                auto check1 = at_1_side_of_line(p1, line);
                auto check2 = at_1_side_of_line(p2, line);

                // `true and false`  or `false and true`
                // means target passed the line in current frame
                if (check1 ^ check2) {
                    total_crossline++;
                    involve_targets.push_back(target->track_id);
                }
            }
        }

        // not empty fill ba result back to frame meta
        if (involve_targets.size() > 0) {
            std::vector<vp_objects::vp_point> involve_region {line.start, line.end};
            auto ba_result = std::make_shared<vp_objects::vp_ba_result>(vp_objects::vp_ba_type::CROSSLINE, meta->channel_index, meta->frame_index, involve_targets, involve_region);
            meta->ba_results.push_back(ba_result);

            // info logo
            VP_INFO(vp_utils::string_format("[%s] crossline data [current/total]: [%d/%d]", node_name.c_str(), involve_targets.size(), total_crossline));
        }

        return meta;
    }

    bool vp_ba_crossline_node::at_1_side_of_line(vp_objects::vp_point p, vp_objects::vp_line line) {
        auto p1 = line.start;
        auto p2 = line.end;

        if (p1.x == p2.x) {
            return p.x < p1.x;
        }

        if (p1.y == p2.y) {
            return p.y < p1.y;
        }

        if (p2.x < p1.x) {
            auto tmp = p2;
            p2 = p1;
            p1 = tmp;
        }

        int ret = (p2.y - p.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p2.x - p.x);
        return ret < 0;
    }
}