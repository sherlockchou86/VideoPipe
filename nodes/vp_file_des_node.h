#pragma once

#include <iostream>
#include <memory>
#include <chrono>
// compile tips:
// remove experimental/ if gcc >= 8.0
#include <experimental/filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "vp_des_node.h"
#include "../objects/vp_frame_meta.h"
#include "../objects/vp_control_meta.h"
#include "../utils/vp_utils.h"


namespace vp_nodes {
    // file des node, save stream to local file
    class vp_file_des_node: public vp_des_node {
    private:
        /* data */
        std::string gst_template = "appsrc ! videoconvert ! x264enc bitrate=%d ! mp4mux ! filesink location=%s";
        cv::VideoWriter file_writer;

        int frames_already_record = -1;
        int frames_need_record = 0;

        // create file name
        std::string get_new_file_name();
    protected:
        // re-implementation, return nullptr.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override; 
        // re-implementation, return nullptr.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
    public:
        vp_file_des_node(std::string node_name, 
                        int channel_index, 
                        std::string save_dir,
                        std::string name_prefix = "",
                        int max_duration_for_single_file = 2,
                        vp_objects::vp_size resolution_w_h = {},
                        int bitrate = 1024,
                        bool osd = true);
        ~vp_file_des_node();

        // save directory
        std::string save_dir;
        // prefix of file name, format of filename: prefix_starttime.mp4
        std::string name_prefix;
        // max duration for single file (minutes)
        int max_duration_for_single_file;

        vp_objects::vp_size resolution_w_h;
        int bitrate;
        bool osd;
    };

}