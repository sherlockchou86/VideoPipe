
#include "vp_video_record_task.h"
#include "../../utils/vp_utils.h"

namespace vp_nodes {
    vp_video_record_task::vp_video_record_task(int channel_index, 
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
                                                std::string host_node_name,
                                                bool auto_start):
                                                vp_record_task(channel_index, file_name_without_ext, save_dir, auto_sub_dir, resolution_w_h, osd, host_node_name),
                                                bitrate(bitrate),
                                                fps(fps),
                                                pre_record_video_duration(pre_record_video_duration),
                                                record_video_duration(record_video_duration) {
        assert(bitrate > 0);
        // transfer to inner cache
        for (auto i: pre_record_frames) {
            frames_to_record.push_back(i);
            cache_semaphore.signal();
        }

        // start automatically when initializing
        if (auto_start) {
            start();
        }
    }

    vp_video_record_task::~vp_video_record_task() {
        stop_task();
    }

    void vp_video_record_task::record_task_run() {
        /* Below Code Run In A Separate Thread! */
        // get valid path
        auto full_record_path = get_full_record_path();
        // format string used by gstreamer
        auto gst = vp_utils::string_format(gst_template, bitrate, full_record_path.c_str());
        
        // get total frames to record
        frames_need_record = (pre_record_video_duration + record_video_duration) * fps;
        frames_already_record = 0;
        // video duration at least 1 second
        assert(frames_need_record > fps * 1);

        // pop data from cache and write into file
        while (status == vp_record_task_status::STARTED) {
            // it is a consumer
            cache_semaphore.wait();

            auto frame_to_record = frames_to_record.front();
            if (frame_to_record == nullptr) {
                // dead flag
                continue;
            }
            
            cv::Mat frame_data;
            // preprocess, vp_frame_meta -> cv::Mat
            preprocess(frame_to_record, frame_data);
            frames_to_record.pop_front();

            // we open video writer in while loop, since we need width and height of frame when open a VideoWriter.
            if (!video_writer.isOpened()) {
                assert(video_writer.open(gst, cv::CAP_GSTREAMER, 0, fps, {frame_data.cols, frame_data.rows}));
            }

            // write cv::Mat to file
            video_writer.write(frame_data);
            frames_already_record++;
            VP_DEBUG(vp_utils::string_format("[%s] [record] already written %d frames for `%s`", host_node_name.c_str(), frames_already_record, get_full_record_path().c_str()));

            /* for video recording, need saving multi frames
             * refer to vp_image_record_task
             */

            // check if complete
            if (frames_already_record >= frames_need_record) {
                // here release writer at once mannually make sure the video file can be used by others.
                video_writer.release();

                vp_record_info record_info;
                record_info.record_type = vp_record_type::VIDEO;
                record_info.pre_record_video_duration = pre_record_video_duration;
                record_info.record_video_duration = record_video_duration;
                notify_task_complete(record_info);
                // loop end
            }
        }
    }


    std::string vp_video_record_task::get_file_ext() {
        // save as mp4
        return ".mp4";
    }  
}