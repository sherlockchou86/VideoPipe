#pragma once

#include "vp_record_task.h"

namespace vp_nodes {
    // video record task, each task instance responsible for recording only 1 video file.
    // create multi instances if multi videos need to be record at the same time, and maintain these tasks in a list.
    class vp_video_record_task: public vp_record_task {
    private:
        // video writer
        cv::VideoWriter video_writer;
        // gst template
        std::string gst_template = "appsrc ! videoconvert ! x264enc bitrate=%d ! mp4mux ! filesink location=%s";
        
        int frames_already_record = -1;
        int frames_need_record = 0;
        int bitrate;
        int fps;
        int record_video_duration;
        int pre_record_video_duration;

    protected:
        // define how to record video
        virtual void record_task_run() override;
        // retrive .mp4 as file extension 
        virtual std::string get_file_ext() override;
    public:
        vp_video_record_task(int channel_index, 
                        std::deque<std::shared_ptr<vp_objects::vp_frame_meta>> pre_record_frames, 
                        std::string file_name_without_ext,
                        std::string save_dir,
                        bool auto_sub_dir,
                        bool osd,
                        vp_objects::vp_size resolution_w_h,
                        int bitrate,
                        int fps,
                        int pre_record_video_duration,
                        int record_video_duration,
                        std::string host_node_name = "host_node_not_specified",
                        bool auto_start = true);
        ~vp_video_record_task();
    };
}