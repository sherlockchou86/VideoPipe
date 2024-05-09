
#include "vp_pose_osd_node.h"

namespace vp_nodes {
    
    vp_pose_osd_node::vp_pose_osd_node(std::string node_name): vp_node(node_name) {
        populateColorPalette(colors, 100);  // generate colors
        this->initialized();
    }
    
    vp_pose_osd_node::~vp_pose_osd_node() {
        deinitialized();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_pose_osd_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // operations on osd_frame
        if (meta->osd_frame.empty()) {
            meta->osd_frame = meta->frame.clone();
        }

        // scan pose targets
        for (int i = 0; i < meta->pose_targets.size(); i++) {
            auto& pose_target = meta->pose_targets[i];

            auto nPairs = posePairs_map.at(pose_target->type).size();

            for (int j = 0; j < nPairs; j++) {
                auto& a = pose_target->key_points[posePairs_map.at(pose_target->type)[j].first];
                auto& b = pose_target->key_points[posePairs_map.at(pose_target->type)[j].second];
                // some points not detected
                if (a.x < 0 || a.y < 0 || b.x < 0 || b.y < 0) {
                    continue;
                }   
                cv::line(meta->osd_frame, cv::Point(a.x, a.y), cv::Point(b.x, b.y), colors[j], 2, cv::LINE_AA);
                cv::circle(meta->osd_frame, cv::Point(a.x, a.y), 3, colors[j], -1, cv::LINE_AA);
                cv::circle(meta->osd_frame, cv::Point(b.x, b.y), 3, colors[j], -1, cv::LINE_AA);
            }
        }

        return meta;
    }

    std::shared_ptr<vp_objects::vp_meta> vp_pose_osd_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return meta;
    }
}