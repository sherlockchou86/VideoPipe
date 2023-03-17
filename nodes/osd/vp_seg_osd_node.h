#pragma once

#include <string>
#include "../vp_node.h"

namespace vp_nodes {
    class vp_seg_osd_node: public vp_node
    {
    private:
        /* data */
        int gap = 60;
        
        // classs names of semantic segmentation
        std::vector<std::string> classes;
        // colors of semantic segmentation
        std::vector<cv::Vec3b> colors;
        void colorizeSegmentation(const cv::Mat &score, cv::Mat &segm);
        void showLegend(cv::Mat& board);

    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
    public:
        vp_seg_osd_node(std::string node_name, std::string classes_file = "", std::string colors_file = "");
        ~vp_seg_osd_node();
    };
}