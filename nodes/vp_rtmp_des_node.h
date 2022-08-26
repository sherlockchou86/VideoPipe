#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "vp_des_node.h"

namespace vp_nodes {
    // rtmp des node, push video stream via rtmp protocal.
    // example:
    // rtmp://192.168.77.105/live/10000
    class vp_rtmp_des_node: public vp_des_node
    {
    private:
        /* data */
        std::string gst_template = "appsrc ! videoconvert ! x264enc bitrate=%d ! h264parse ! flvmux ! rtmpsink location=%s";
        cv::VideoWriter rtmp_writer;
    protected:
        // re-implementation, return nullptr.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override; 
        // re-implementation, return nullptr.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
    public:
        vp_rtmp_des_node(std::string node_name, 
                        int channel_index, 
                        std::string rtmp_url, 
                        vp_objects::vp_size resolution_w_h = {}, 
                        int bitrate = 1024,
                        bool osd = true);
        ~vp_rtmp_des_node();

        virtual std::string to_string() override;
        
        std::string rtmp_url;
        // resolution for rtmp stream
        vp_objects::vp_size resolution_w_h;
        int bitrate;

        // for osd frame 
        bool osd;
    };
}