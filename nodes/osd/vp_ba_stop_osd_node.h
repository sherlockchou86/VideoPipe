#pragma once

#include <map>
#include <opencv2/freetype.hpp>
#include "../vp_node.h"
#include "../../objects/shapes/vp_point.h"
#include "../../objects/shapes/vp_line.h"

namespace vp_nodes {
    // osd node for behaviour analysis of stop
    class vp_ba_stop_osd_node: public vp_node
    {
    private:
        // support chinese font
        cv::Ptr<cv::freetype::FreeType2> ft2;

        // support multi channels
        std::map<int, std::vector<int>> all_total_stop;  // channel -> stop target ids
        std::map<int, std::vector<vp_objects::vp_point>> all_stop_regions;   // channel -> stop region
        std::map<int, int> all_stop_count;  // channel -> stop count
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
    public:
        vp_ba_stop_osd_node(std::string node_name,  std::string font = "");
        ~vp_ba_stop_osd_node();
    };
}