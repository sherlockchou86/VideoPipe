
#include "vp_file_des_node.h"

namespace vp_nodes {
        
    vp_file_des_node::vp_file_des_node(std::string node_name, 
                                        int channel_index, 
                                        std::string save_dir,
                                        std::string name_prefix,
                                        int max_duration_for_single_file,
                                        vp_objects::vp_size resolution_w_h,
                                        int bite_rate,
                                        bool osd): 
                                        vp_des_node(node_name, channel_index), 
                                        save_dir(save_dir),
                                        name_prefix(name_prefix),
                                        max_duration_for_single_file(max_duration_for_single_file),
                                        resolution_w_h(resolution_w_h),
                                        bitrate(bitrate),
                                        osd(osd) {
        // compile tips:
        // remove experimental:: if gcc >= 8.0
        assert(std::experimental::filesystem::exists(save_dir));
        // max time to record
        assert(max_duration_for_single_file <= 30);
        VP_INFO(vp_utils::string_format("[%s] [%s]", node_name.c_str(), gst_template.c_str()));
        this->initialized();
    }
    
    vp_file_des_node::~vp_file_des_node() {

    }
    
    // re-implementation, return nullptr.
    std::shared_ptr<vp_objects::vp_meta> 
        vp_file_des_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
            VP_DEBUG(vp_utils::string_format("[%s] received frame meta, channel_index=>%d, frame_index=>%d", node_name.c_str(), meta->channel_index, meta->frame_index));
            
            cv::Mat resize_frame;
            if (this->resolution_w_h.width != 0 && this->resolution_w_h.height != 0) {                 
                cv::resize((osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame, resize_frame, cv::Size(resolution_w_h.width, resolution_w_h.height));
            }
            else {
                resize_frame = (osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame;
            }

            // new video file
            if (!file_writer.isOpened() ||
                frames_already_record >= frames_need_record) {
                    // total frames need to be recorded
                    frames_need_record = max_duration_for_single_file * 60 * meta->fps;
                    frames_already_record = 0;

                    if (name_prefix.empty()) {
                        name_prefix = node_name + "_" + std::to_string(meta->channel_index);
                    }

                    // close it first if it has opened
                    if (file_writer.isOpened()) {
                        /* code */
                        file_writer.release();
                    }
                    
                    auto gst_str = vp_utils::string_format(this->gst_template, bitrate, get_new_file_name().c_str());
                    assert(file_writer.open(gst_str, cv::CAP_GSTREAMER, 0, meta->fps, {resize_frame.cols, resize_frame.rows}));
            }
            
            file_writer.write(resize_frame);
            frames_already_record++;

            // for general works defined in base class
            return vp_des_node::handle_frame_meta(meta);
    }

    // re-implementation, return nullptr.
    std::shared_ptr<vp_objects::vp_meta> 
        vp_file_des_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
            return nullptr;
    }

    std::string vp_file_des_node::get_new_file_name() {
        auto stamp = std::chrono::system_clock::now().time_since_epoch().count(); 
        // compile tips:
        // remove experimental:: if gcc >= 8.0
        std::experimental::filesystem::path p1(save_dir);
        std::experimental::filesystem::path p2(name_prefix + "_" + std::to_string(stamp) + ".mp4");

        // save_dir/name_prefix_stamp.mp4
        std::experimental::filesystem::path p = p1 / p2;

        assert(!std::experimental::filesystem::exists(p));
        return p.string();
    }
}