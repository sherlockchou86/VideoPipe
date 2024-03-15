
#pragma once

#include "vp_src_node.h"

namespace vp_nodes {
    // image src node, read image from local files or receive image from remote via udp.
    class vp_image_src_node: public vp_src_node
    {
    private:
        // gstreamer template for reading image from file (jpeg encoding only, filename MUST end with 'jpg/jpeg')
        std::string gst_template_file = "multifilesrc location=%s loop=%s ! jpegparse ! %s ! videorate ! video/x-raw,framerate=1/%d ! videoconvert ! appsink";
        // gstreamer template for receiving image from remote via udp （jpeg encoding only)
        std::string gst_template_udp = "udpsrc port=%d ! application/x-rtp,encoding-name=jpeg ! rtpjpegdepay ! jpegparse ! %s ! videorate ! video/x-raw,framerate=1/%d ! videoconvert ! appsink";

        cv::VideoCapture image_capture;

        // `port number` for udp mode, `directory path` for file mode. auto-detect which mode should apply according to this value.
        std::string port_or_location = "./%d.jpg";
        // frequency to read/receive image
        int interval = 1;

        // restart reading if images read completely (only used for file mode)
        bool cycle = true;

        bool from_file = true;
    protected:
        // re-implemetation
        virtual void handle_run() override;

    public:
        vp_image_src_node(std::string node_name, 
                        int channel_index, 
                        std::string port_or_location,
                        int interval = 1,
                        float resize_ratio = 1.0, 
                        bool cycle = true,
                        std::string gst_decoder_name = "jpegdec");
        ~vp_image_src_node();
        virtual std::string to_string() override;

        // set jpegdec as the default decoder, we can use hardware decoder instead.
        std::string gst_decoder_name = "jpegdec";
    };
    
}