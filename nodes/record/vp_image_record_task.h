
#pragma once

#include "vp_record_task.h"

namespace vp_nodes {
    // image record task, each task responsible for recording only 1 image file.
    // task works asynchronously.
    class vp_image_record_task: public vp_record_task {
    private:
        
    public:
        vp_image_record_task(std::shared_ptr<vp_objects::vp_frame_meta> frame_to_record,
                            std::string file_name_without_ext,
                            std::string save_dir,
                            bool auto_sub_dir,
                            vp_objects::vp_size resolution_w_h,
                            bool osd);
        ~vp_image_record_task();

        // record asynchronously
        void record_async();
    };

}