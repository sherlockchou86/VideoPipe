

#include <opencv2/imgproc.hpp>
#include "vp_osd_node.h"

namespace vp_nodes {
        
    vp_osd_node::vp_osd_node(std::string node_name, 
                            vp_osd_option options):
                            vp_node(node_name),
                            osd_options(options) {
        this->initialized();
    }
    
    vp_osd_node::~vp_osd_node()
    {
    }
    
    std::shared_ptr<vp_objects::vp_meta> vp_osd_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return meta;
    }

    // display logic
    std::shared_ptr<vp_objects::vp_meta> vp_osd_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // operations on osd_frame
        if (meta->osd_frame.empty()) {
            meta->osd_frame = meta->frame.clone();
        }
        auto& canvas = meta->osd_frame;
        // scan targets
        for (auto& i : meta->targets) {
            auto labels_to_display = i->primary_label;
            for ( auto& label : i->secondary_labels) {
                labels_to_display += "/" + label;
            }
            
            cv::putText(canvas, labels_to_display, cv::Point(i->x, i->y), 1, 1, cv::Scalar(255, 0, 255));
            cv::rectangle(canvas, cv::Rect(i->x, i->y, i->width, i->height), cv::Scalar(255, 255, 0), 2);
        }
        return meta;
    }
}