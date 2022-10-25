#pragma once

#include "vp_record_task.h"

namespace vp_nodes {
    // callback when record task complete.
    class vp_record_status_hookable
    {
    protected:
        vp_record_task_complete_hooker image_record_complete_hooker;
        vp_record_task_complete_hooker video_record_complete_hooker;
    public:
        vp_record_status_hookable(/* args */) {}
        ~vp_record_status_hookable() {}

        void set_image_record_complete_hooker(vp_record_task_complete_hooker image_record_complete_hooker) {
            this->image_record_complete_hooker = image_record_complete_hooker;
        }

        void set_video_record_complete_hooker(vp_record_task_complete_hooker video_record_complete_hooker) {
            this->video_record_complete_hooker = video_record_complete_hooker;
        }
    }; 
}