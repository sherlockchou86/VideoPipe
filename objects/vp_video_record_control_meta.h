
#pragma once

#include "vp_control_meta.h"

namespace vp_objects {
    class vp_video_record_control_meta: public vp_control_meta {
    private:
        /* data */
    public:
        vp_video_record_control_meta(int channel_index);
        ~vp_video_record_control_meta();


    };
    
    vp_video_record_control_meta::vp_video_record_control_meta(/* args */) {

    }
    
    vp_video_record_control_meta::~vp_video_record_control_meta()
    {
    }
    
}