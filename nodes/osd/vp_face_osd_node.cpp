
#include <opencv2/imgproc.hpp>
#include "vp_face_osd_node.h"

namespace vp_nodes {
        
    vp_face_osd_node::vp_face_osd_node(std::string node_name): vp_node(node_name) {
        this->initialized();
    }
    
    vp_face_osd_node::~vp_face_osd_node() {
        deinitialized();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_face_osd_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // operations on osd_frame
        if (meta->osd_frame.empty()) {
            meta->osd_frame = meta->frame.clone();
        }
        auto& canvas = meta->osd_frame;
        
        // scan face targets
        for(auto& i : meta->face_targets) {
            cv::rectangle(canvas, cv::Rect(i->x, i->y, i->width, i->height), cv::Scalar(0, 255, 0), 2);

            // track_id
            if (i->track_id != -1) {
                auto id = std::to_string(i->track_id);
                cv::putText(canvas, id, cv::Point(i->x, i->y), 1, 1.5, cv::Scalar(0, 0, 255));
            }

            // just handle 5 keypoints
            if (i->key_points.size() >= 5) {
                cv::circle(canvas, cv::Point(i->key_points[0].first, i->key_points[0].second), 2, cv::Scalar(255, 0, 0), 2);
                cv::circle(canvas, cv::Point(i->key_points[1].first, i->key_points[1].second), 2, cv::Scalar(0, 0, 255), 2);
                cv::circle(canvas, cv::Point(i->key_points[2].first, i->key_points[2].second), 2, cv::Scalar(0, 255, 0), 2);
                cv::circle(canvas, cv::Point(i->key_points[3].first, i->key_points[3].second), 2, cv::Scalar(255, 0, 255), 2);
                cv::circle(canvas, cv::Point(i->key_points[4].first, i->key_points[4].second), 2, cv::Scalar(0, 255, 255), 2);
            }
        }

        return meta;
    }

    std::shared_ptr<vp_objects::vp_meta> vp_face_osd_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return meta;
    }
}