#pragma once

#include <deque>
#include <thread>
#include <memory>
#include <functional>
// compile tips:
// remove experimental/ if gcc >= 8.0
#include <experimental/filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

#include "../../objects/vp_frame_meta.h"
#include "../../utils/vp_utils.h"
#include "../../utils/vp_semaphore.h"
#include "../../utils/logger/vp_logger.h"

namespace vp_nodes {
    // record type
    enum vp_record_type {
        IMAGE,
        VIDEO
    };

    // record information, used to notify others.
    struct vp_record_info {
        int channel_index;
        vp_record_type record_type = vp_record_type::IMAGE;
        std::string file_name_without_ext;
        std::string full_record_path;
        bool osd;

        int pre_record_video_duration = 0;   // ignore for image
        int record_video_duration = 0;       // ignore for image
    };

    // hooker for recording complete
    typedef std::function<void(int, vp_record_info)> vp_record_task_complete_hooker;

    // status of record task
    enum vp_record_task_status {
        NOSTRAT,   // task initialized but have not called start()
        STARTED,   // called start() and task is working (writing/saving data to file)
        COMPLETE   // record task is complete. task instance is un-reusable
    };

    // base class for record task (video & image), works asynchronously and mainly responsible for:
    // 1. preprocess frame before recording
    // 2. generate valid full record path, including path, name with extension
    // 3. run working thread
    // 4. notify caller when recording complete
    class vp_record_task {
    private:
        int channel_index;
        std::string file_name_without_ext;
        std::string save_dir;
        bool auto_sub_dir;
        vp_objects::vp_size resolution_w_h;
        bool osd;
        
        std::string full_record_path = "";
        vp_record_task_complete_hooker task_complete_hooker;
    protected:
        // record thread
        std::thread record_task_th;
        // record thread func, implemented by child class
        virtual void record_task_run() = 0;

        // preprocess, choose frame type (osd or not) and resize
        void preprocess(std::shared_ptr<vp_objects::vp_frame_meta>& frame_to_record, cv::Mat& data);
        // get file extension override by specific class (for example, .mp4 for video and .jpg for image)
        virtual std::string get_file_ext() = 0;

        // notify to host when task complete
        void notify_task_complete(vp_record_info record_info);

        // cache frames to be recorded (video or image)
        // 1. include pre-record frames for video
        // 2. just one frame enough for image
        std::deque<std::shared_ptr<vp_objects::vp_frame_meta>>  frames_to_record;

        // synchronize for cache
        vp_utils::vp_semaphore cache_semaphore;
        std::string host_node_name;   // the node name of host, vp_record_task is mainly used inside node.
    public:
        // status
        vp_record_task_status status = vp_record_task_status::NOSTRAT;
        // get full record path for file, include path, name with extension
        std::string get_full_record_path();
        // register hooker for recording complete
        void set_task_complete_hooker(vp_record_task_complete_hooker task_complete_hooker);
        // start task async
        void start();
   
        // append asynchronously, just write frame to cache
        void append_async(std::shared_ptr<vp_objects::vp_frame_meta> frame_meta);

        vp_record_task(int channel_index,
                        std::string file_name_without_ext, 
                        std::string save_dir, 
                        bool auto_sub_dir, 
                        vp_objects::vp_size resolution_w_h, 
                        bool osd,
                        std::string host_node_name);
        virtual ~vp_record_task();   // keep virtual since we need destruct child class via base pointer
    };

}