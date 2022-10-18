
#include "vp_screen_des_node.h"
#include "../utils/vp_utils.h"

namespace vp_nodes {
    vp_screen_des_node::vp_screen_des_node(std::string node_name, 
                                            int channel_index, 
                                            bool osd,
                                            vp_objects::vp_size display_w_h):
                                            vp_des_node(node_name, channel_index),
                                            osd(osd),
                                            display_w_h(display_w_h) {
        this->gst_template = vp_utils::string_format(this->gst_template, node_name.c_str());
        VP_INFO(vp_utils::string_format("[%s] [%s]", node_name.c_str(), gst_template.c_str()));
        this->initialized();
    }
    
    vp_screen_des_node::~vp_screen_des_node() {

    }

    // re-implementation, return nullptr.
    std::shared_ptr<vp_objects::vp_meta> 
        vp_screen_des_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
            VP_DEBUG(vp_utils::string_format("[%s] received frame meta, channel_index=>%d, frame_index=>%d", node_name.c_str(), meta->channel_index, meta->frame_index));
            
            cv::Mat resize_frame;
            if (this->display_w_h.width != 0 && this->display_w_h.height != 0) {                 
                cv::resize((osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame, resize_frame, cv::Size(display_w_h.width, display_w_h.height));
            }
            else {
                resize_frame = (osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame;
            }

            if (!screen_writer.isOpened()) {
                assert(screen_writer.open(this->gst_template, cv::CAP_GSTREAMER, 0, meta->fps, {resize_frame.cols, resize_frame.rows}));
            }
            screen_writer.write(resize_frame);

            // for general works defined in base class
            return vp_des_node::handle_frame_meta(meta);
    }

    // re-implementation, return nullptr.
    std::shared_ptr<vp_objects::vp_meta> 
        vp_screen_des_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
            // for general works defined in base class
            return vp_des_node::handle_control_meta(meta);
    }
}