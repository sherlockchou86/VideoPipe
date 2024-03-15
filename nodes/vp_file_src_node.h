#pragma once

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "vp_src_node.h"

namespace vp_nodes {
    // file source node, read video from local file.
    // example:
    // ../video/test.mp4
    class vp_file_src_node: public vp_src_node {
    private:
        /* data */
        std::string gst_template = "filesrc location=%s ! qtdemux ! h264parse ! %s ! videoconvert ! appsink";
        cv::VideoCapture file_capture;
    protected:
        // re-implemetation
        virtual void handle_run() override;
    public:
        vp_file_src_node(std::string node_name, 
                        int channel_index, 
                        std::string file_path, 
                        float resize_ratio = 1.0, 
                        bool cycle = true,
                        std::string gst_decoder_name = "avdec_h264",
                        int skip_interval = 0);
        ~vp_file_src_node();

        virtual std::string to_string() override;
        std::string file_path;
        bool cycle;
    
        // set avdec_h264 as the default decoder, we can use hardware decoder instead.
        std::string gst_decoder_name = "avdec_h264";
        // 0 means no skip
        int skip_interval = 0;
    };

}