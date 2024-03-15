#pragma once

#include "vp_des_node.h"

namespace vp_nodes {
    // image des node, save image to local file or push image to remote via udp.
    class vp_image_des_node: public vp_des_node
    {
    private:
        // gstreamer template for saving image to file (jpeg encoding only, filename MUST end with 'jpg/jpeg')
        std::string gst_template_file = "appsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,framerate=1/%d ! %s ! multifilesink location=%s";
        // gstreamer template for pushing image to remote via udp （jpeg encoding only)
        std::string gst_template_udp = "appsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=I420,framerate=1/%d ! %s ! rtpjpegpay ! udpsink host=%s port=%d";
        
        // gstreamer wrapper by opencv
        cv::VideoWriter image_writer;

        // save/push ONE image every `interval` seconds
        int interval;
        // save/push image to where
        // for example, `./%d.jpg` for file (end with `jpg/jpeg`), `192.168.1.90:8000` for udp (split by `:`)
        std::string location = "./%d.jpg";

        bool osd = true;
        vp_objects::vp_size resolution_w_h;

        bool to_file = true;
        // set jpegenc as the default encoder, we can use hardware encoder instead.
        std::string gst_encoder_name = "jpegenc";
    protected:
        // re-implementation, return nullptr.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override; 

    public:
        vp_image_des_node(std::string node_name, 
                        int channel_index,  
                        std::string location = "./%d.jpg", 
                        int interval = 5,
                        vp_objects::vp_size resolution_w_h = {},
                        bool osd = true,
                        std::string gst_encoder_name = "jpegenc");
        ~vp_image_des_node();
        virtual std::string to_string() override;
    };

}