#pragma once

#include "../vp_node.h"

namespace vp_nodes {
    // fuse video frames from 2 channels based on the given calibration points.
    // only support to fuse 2 channels at the same time so far, fuse the first to second or vice versa, just fuse directly did not check the timestamp of frame.
    class vp_frame_fusion_node: public vp_node
    {
    private:
        std::shared_ptr<vp_objects::vp_frame_meta> tmp_des = nullptr;
        cv::Mat trans_mat;
        int src_channel_index = 0;
        int des_channel_index = 1;

        void fuse(cv::Mat& src_canvas, cv::Mat& des_canvas);
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
    public:
        vp_frame_fusion_node(std::string node_name, 
                            std::vector<vp_objects::vp_point> src_points,   // 4 calibration points of the source frame
                            std::vector<vp_objects::vp_point> des_points,   // 4 calibration points of the destination frame
                            int src_channel_index = 0, 
                            int des_channel_index = 1);
        ~vp_frame_fusion_node();
    };
}