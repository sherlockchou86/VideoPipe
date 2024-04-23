
#include "vp_frame_fusion_node.h"

namespace vp_nodes {
    
    vp_frame_fusion_node::vp_frame_fusion_node(std::string node_name, 
                            std::vector<vp_objects::vp_point> src_points,   // 4 calibration points of the source frame
                            std::vector<vp_objects::vp_point> des_points,   // 4 calibration points of the destination frame
                            int src_channel_index, 
                            int des_channel_index): vp_node(node_name), src_channel_index(src_channel_index), des_channel_index(des_channel_index) {
        assert(src_channel_index != des_channel_index);
        assert(src_points.size() == 4 && des_points.size() == 4);  // cv::getPerspectiveTransform(...) need 4 pairs of point

        cv::Point2f src_ps[4], des_ps[4];
        for (int i = 0; i < 4; i++) {
            src_ps[i] = cv::Point2f(src_points[i].x, src_points[i].y);
            des_ps[i] = cv::Point2f(des_points[i].x, des_points[i].y);
        }
        // get transform matrix
        trans_mat = cv::getPerspectiveTransform(src_ps, des_ps);
        this->initialized();  
    }
    
    vp_frame_fusion_node::~vp_frame_fusion_node() {
        deinitialized();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_frame_fusion_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // ignore for unrelated channels
        if (meta->channel_index != src_channel_index && meta->channel_index != des_channel_index) {
            return meta;
        }
        
        // for the source channel
        if (meta->channel_index == src_channel_index) {
            if (tmp_des == nullptr) {
                // not ready, return directly
                return meta;
            }
            else {
                // ready to fuse, put result to osd_frame
                if (tmp_des->osd_frame.empty()) {
                    tmp_des->osd_frame = tmp_des->frame.clone();
                }
                fuse(meta->frame, tmp_des->osd_frame);

                // destination channel first
                pendding_meta(tmp_des);
                tmp_des = nullptr;

                // source channel last
                return meta;
            }
        }
        
        // for the destination channel
        if (meta->channel_index == des_channel_index) {
            if (tmp_des != nullptr) {
                // push previous one to downstream first 
                pendding_meta(tmp_des);
            }
            // cache current one
            tmp_des = meta;
            // return nullptr since we need fuse next time
            return nullptr;
        }
    }

    void vp_frame_fusion_node::fuse(cv::Mat& src_canvas, cv::Mat& des_canvas) {
        // transform source image
        cv::Mat trans_result;
        cv::warpPerspective(src_canvas, trans_result, trans_mat, des_canvas.size());

        // merge pixels by weights
        cv::addWeighted(trans_result, 0.7, des_canvas, 0.3, 0, des_canvas);
    }
}