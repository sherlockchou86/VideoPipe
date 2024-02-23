#include "vp_app_src_node.h"

namespace vp_nodes {
    vp_app_src_node::vp_app_src_node(std::string node_name, 
                        int channel_index):vp_src_node(node_name, channel_index, 1.0) {
        this->initialized();
    }

    vp_app_src_node::~vp_app_src_node() {
        deinitialized();
    }

    // host code acts as previous node, call vp_node::meta_flow(...)
    bool vp_app_src_node::push_frames(std::vector<cv::Mat> frames) {
        // vp_app_src_node not working
        if (!gate.is_open()) {
            VP_WARN(vp_utils::string_format("[%s] is not working!", node_name.c_str()));
            return false;
        }
        
        if (frames.size() == 0) {
            return false;
        }

        // MUST have the same size
        auto size_warn = [this]() {
            VP_WARN(vp_utils::string_format("[%s] frames to be pushed MUST have the same size!", this->node_name.c_str()));
        };
        auto w = frames[0].cols;
        auto h = frames[0].rows;
        for (auto& f: frames) {
            if (f.cols != w || f.rows != h) {
                size_warn();
                return false;
            }
        }

        if (original_height != 0 && original_height != h) {
            size_warn();
            return false;
        }

        if (original_width != 0 && original_width != w) {
            size_warn();
            return false;
        }

        // initialize video properties
        if (original_width == 0 || original_height == 0 || original_fps == 0) {    
            original_width = w;
            original_height = h;
            original_fps = 1;  // set constant value 1 for vp_app_src_node
        }
        // stream_info_hooker activated if need
        vp_stream_info stream_info {channel_index, original_fps, original_width, original_height, to_string()};
        invoke_stream_info_hooker(node_name, stream_info);

        for (auto& f: frames) {
            frame_index++;
            auto frame = f.clone();  // cv::Mat::clone() inside pipeline
            // create frame meta and meta flow like previous node
            auto in_meta = std::make_shared<vp_objects::vp_frame_meta>(frame, frame_index, channel_index, original_width, original_height, original_fps);

            vp_node::meta_flow(in_meta);
        }
        return true;
    }

    void vp_app_src_node::handle_run() {
        // call vp_node::handle_run() since we assume vp_app_src_node has virtual previous node (from host code)
        vp_node::handle_run();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_app_src_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        return vp_node::handle_frame_meta(meta);
    }

    std::shared_ptr<vp_objects::vp_meta> vp_app_src_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return vp_node::handle_control_meta(meta);
    }
}