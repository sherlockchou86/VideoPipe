#pragma once

#include <string>

#include "vp_src_node.h"

namespace vp_nodes {
    // rtmp source node, receive live video stream via rtmp protocal.
    // example:
    // rtmp://your-rtmp-server/live/streamname
    class vp_rtmp_src_node: public vp_src_node {
    private:
        /* data */
        std::string gst_template = "rtmpsrc location=%s ! flvdemux ! h264parse ! %s ! videoconvert ! appsink";
        cv::VideoCapture rtmp_capture;
    protected:
        // re-implemetation
        virtual void handle_run() override;
    public:
        vp_rtmp_src_node(std::string node_name, 
                        int channel_index, 
                        std::string rtmp_url, 
                        float resize_ratio = 1.0,
                        std::string gst_decoder_name = "avdec_h264",
                        int skip_interval = 0);
        ~vp_rtmp_src_node();

        virtual std::string to_string() override;

        std::string rtmp_url;
        // set avdec_h264 as the default decoder, we can use hardware decoder instead.
        std::string gst_decoder_name = "avdec_h264";
        // 0 means no skip
        int skip_interval = 0;
    };
}