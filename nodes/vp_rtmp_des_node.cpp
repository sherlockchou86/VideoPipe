
#include <assert.h>
#include "vp_rtmp_des_node.h"
#include "../utils/vp_utils.h"

namespace vp_nodes {
        
    vp_rtmp_des_node::vp_rtmp_des_node(std::string node_name, 
                                        int channel_index, 
                                        std::string rtmp_url, 
                                        vp_objects::vp_size resolution_w_h, 
                                        int bitrate,
                                        bool osd):
                                        vp_des_node(node_name, channel_index),
                                        rtmp_url(rtmp_url),
                                        resolution_w_h(resolution_w_h),
                                        bitrate(bitrate),
                                        osd(osd) {
        // append channel_index to the end of rtmp_url.
        // if original rtmp_url is rtmp://192.168.77.105/live/10000 and channel_index is 10
        // then the modified rtmp_url is rtmp://192.168.77.105/live/10000_10
        this->rtmp_url = this->rtmp_url + "_" + std::to_string(channel_index);
        this->gst_template = vp_utils::string_format(this->gst_template, bitrate, this->rtmp_url.c_str());
        VP_INFO(vp_utils::string_format("[%s] [%s]", node_name.c_str(), gst_template.c_str()));
        this->initialized();
    }
    
    vp_rtmp_des_node::~vp_rtmp_des_node() {
        
    }

    // re-implementation, return nullptr.
    std::shared_ptr<vp_objects::vp_meta> 
        vp_rtmp_des_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
            VP_DEBUG(vp_utils::string_format("[%s] received frame meta, channel_index=>%d, frame_index=>%d", node_name.c_str(), meta->channel_index, meta->frame_index));
            
            cv::Mat resize_frame;
            if (this->resolution_w_h.width != 0 && this->resolution_w_h.height != 0) {                 
                cv::resize((osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame, resize_frame, cv::Size(resolution_w_h.width, resolution_w_h.height));
            }
            else {
                resize_frame = (osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame;
            }

            if (!rtmp_writer.isOpened()) {
                assert(rtmp_writer.open(this->gst_template, cv::CAP_GSTREAMER, 0, meta->fps, {resize_frame.cols, resize_frame.rows}));
            }

            rtmp_writer.write(resize_frame);

            // for general works defined in base class
            return vp_des_node::handle_frame_meta(meta);
    }

    // re-implementation, return nullptr.
    std::shared_ptr<vp_objects::vp_meta> 
        vp_rtmp_des_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
            // for general works defined in base class
            return vp_des_node::handle_control_meta(meta);
    }

    std::string vp_rtmp_des_node::to_string() {
        // just return rtmp url
        return rtmp_url;
    }
}