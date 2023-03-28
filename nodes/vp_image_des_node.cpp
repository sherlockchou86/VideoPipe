

#include "vp_image_des_node.h"


namespace vp_nodes {
        
    vp_image_des_node::vp_image_des_node(std::string node_name, 
                                    int channel_index,  
                                    std::string location, 
                                    int interval,
                                    vp_objects::vp_size resolution_w_h,
                                    bool osd):
                                    vp_des_node(node_name, channel_index),
                                    location(location),
                                    interval(interval),
                                    resolution_w_h(resolution_w_h),
                                    osd(osd) {
        // not greater than 1 minutes
        assert(interval >= 1 && interval <= 60);
        if (vp_utils::ends_with(location, ".jpeg") || vp_utils::ends_with(location, ".jpg")) {
            // save to file
            gst_template_file = vp_utils::string_format(gst_template_file, interval, location.c_str());
            to_file = true;
        }
        else if (location.find(":") != std::string::npos) {
            // push to remote
            auto parts = vp_utils::string_split(location, ':');
            assert(parts.size() == 2);
            auto host = parts[0];  // ip
            auto port = std::stoi(parts[1]);  // try to get port

            gst_template_udp = vp_utils::string_format(gst_template_udp, interval, host.c_str(), port);
            
            to_file = false;
        }
        else {
            // error
            throw "invalid location!";
        }

        this->initialized();
    }
    
    vp_image_des_node::~vp_image_des_node() {

    }

    std::shared_ptr<vp_objects::vp_meta> vp_image_des_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        VP_DEBUG(vp_utils::string_format("[%s] received frame meta, channel_index=>%d, frame_index=>%d", node_name.c_str(), meta->channel_index, meta->frame_index));
        
        cv::Mat resize_frame;
        if (this->resolution_w_h.width != 0 && this->resolution_w_h.height != 0) {                 
            cv::resize((osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame, resize_frame, cv::Size(resolution_w_h.width, resolution_w_h.height));
        }
        else {
            resize_frame = (osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame;
        }

        if (!image_writer.isOpened()) {
            if (to_file) {
                assert(image_writer.open(this->gst_template_file, cv::CAP_GSTREAMER, 0, meta->fps, {resize_frame.cols, resize_frame.rows}));
            }
            else {
                assert(image_writer.open(this->gst_template_udp, cv::CAP_GSTREAMER, 0, meta->fps, {resize_frame.cols, resize_frame.rows}));
            }
        }

        image_writer.write(resize_frame);

        // for general works defined in base class
        return vp_des_node::handle_frame_meta(meta);
    }
}