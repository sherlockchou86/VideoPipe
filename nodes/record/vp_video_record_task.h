#pragma once

#include "vp_record_task.h"

namespace vp_nodes {
    // video record task, each task responsible for recording only 1 video file.
    // task works asynchronously.
    class vp_video_record_task: public vp_record_task{
    private:
        // cache frames to be recorded
        std::queue<std::shared_ptr<vp_objects::vp_frame_meta>>  frames_to_record;
        // working thread
        std::thread record_task_th;

        // video writer
        cv::VideoWriter video_writer;
        // gst template
        std::string gst_template = "appsrc ! videoconvert ! x264enc bitrate=%d ! mp4mux ! filesink location=%s";
        
        int frames_already_record = -1;
        int frames_need_record = 0;
        int bitrate;
        int record_video_duration;

        // working func
        void record_task_run();
    public:
        vp_video_record_task(std::queue<std::shared_ptr<vp_objects::vp_frame_meta>>& pre_record_frames, 
                        std::string file_name_without_ext,
                        std::string save_dir,
                        bool auto_sub_dir,
                        vp_objects::vp_size resolution_w_h,
                        int bitrate,
                        bool osd,
                        int record_video_duration);
        ~vp_video_record_task();

        // record asynchronously, just write frame to cache
        void record_async(std::shared_ptr<vp_objects::vp_frame_meta> frame_meta);

        // recording complete or not
        bool complete;
    };
}