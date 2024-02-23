
#include "vp_image_record_task.h"

namespace vp_nodes {
    vp_image_record_task::vp_image_record_task(int channel_index,
                        std::string file_name_without_ext,
                        std::string save_dir,
                        bool auto_sub_dir,
                        bool osd,
                        vp_objects::vp_size resolution_w_h,
                        std::string host_node_name,
                        bool auto_start):
                        vp_record_task(channel_index, file_name_without_ext, save_dir, auto_sub_dir, resolution_w_h, osd, host_node_name) {
        // start automatically when initializing
        if (auto_start) {
            start();
        }   
    }

    vp_image_record_task::~vp_image_record_task() {
        stop_task();
    }

    void vp_image_record_task::record_task_run() {
        /* Below Code Run In A Separate Thread! */
        // get valid path
        auto full_record_path = get_full_record_path();
        
        while (status == vp_record_task_status::STARTED) {
             // it is a consumer
            cache_semaphore.wait();

            auto frame_to_record = frames_to_record.front();
            if (frame_to_record == nullptr) {
                //dead flag
                continue;
            }

            cv::Mat frame_data;
            // preprocess, vp_frame_meta -> cv::Mat
            preprocess(frame_to_record, frame_data);
            frames_to_record.pop_front();
            
            // write to disk
            cv::imwrite(full_record_path, frame_data);
            VP_DEBUG(vp_utils::string_format("[%s] [record] already written frame for `%s`", host_node_name.c_str(), get_full_record_path().c_str()));
            /* for image recording, just saving only 1 frame and then complete
             * refer to vp_video_record_task
             */

            vp_record_info record_info;
            record_info.record_type = vp_record_type::IMAGE;
            notify_task_complete(record_info);
            // loop end
        }
    }

    std::string vp_image_record_task::get_file_ext() {
        // save as jpg
        return ".jpg";
    }
}