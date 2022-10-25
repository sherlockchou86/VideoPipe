

#include "vp_video_record_control_meta.h"


namespace vp_objects {
    
    vp_video_record_control_meta::vp_video_record_control_meta(int channel_index, 
                                                            std::string video_file_name_without_ext, 
                                                            int record_video_duration,
                                                            bool osd):
                                                            vp_control_meta(vp_objects::vp_control_type::VIDEO_RECORD, channel_index),
                                                            video_file_name_without_ext(video_file_name_without_ext),
                                                            record_video_duration(record_video_duration),
                                                            osd(osd) {

    }
    
    vp_video_record_control_meta::~vp_video_record_control_meta() {

    }

    std::shared_ptr<vp_meta> vp_video_record_control_meta::clone() {
        // just call copy constructor and return new pointer
        return std::make_shared<vp_video_record_control_meta>(*this);
    }
}