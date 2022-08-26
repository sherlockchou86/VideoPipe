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
        cv::VideoCapture file_capture;
    protected:
        // re-implemetation
        virtual void handle_run() override;
    public:
        vp_file_src_node(std::string node_name, 
                        int channel_index, 
                        std::string file_path, 
                        float resize_ratio = 1.0, 
                        bool cycle = true);
        ~vp_file_src_node();

        virtual std::string to_string() override;
        std::string file_path;
        bool cycle;
    };

}