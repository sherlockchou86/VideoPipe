#include "vp_ba_result.h"


namespace vp_objects {

    vp_ba_result::vp_ba_result(vp_ba_type type, 
                    int channel_index,
                    int frame_index,
                    std::vector<int> involve_target_ids_in_frame, 
                    std::vector<vp_objects::vp_point> involve_region_in_frame,
                    std::string ba_label,
                    std::string record_image_name,
                    std::string record_video_name):
                    type(type), channel_index(channel_index), frame_index(frame_index),
                    involve_target_ids_in_frame(involve_target_ids_in_frame),
                    involve_region_in_frame(involve_region_in_frame),
                    ba_label(ba_label),
                    record_image_name(record_image_name),
                    record_video_name(record_video_name) {
        
    }

    vp_ba_result::~vp_ba_result() {

    }

    std::string vp_ba_result::to_string() {
        return "";
    }

    std::shared_ptr<vp_ba_result> vp_ba_result::clone() {
        return std::make_shared<vp_ba_result>(*this);
    }
}